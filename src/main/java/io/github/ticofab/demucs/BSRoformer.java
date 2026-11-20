package io.github.ticofab.demucs;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * BS-Roformer (band-split RoPE transformer) inference, for checkpoints like BS-Roformer-SW as run by
 * audio-separator: STFT 2048/512, 62 frequency bands, 12 axial layers (a transformer along time, then one along
 * frequency) at width 256, and one mask head per stem. Only the requested stem's head is loaded.
 *
 * <p>The input spectrogram is cut into bands; each band becomes a 256-wide token per frame. The transformers mix
 * tokens across time within a band, then across bands within a frame. A per-band MLP finally turns each token into
 * a complex mask that is multiplied onto the original spectrogram, and the inverse STFT gives the stem.
 *
 * <p>Activations are token-major {@code (frames, bands, 256)}; weights are looked up by their PyTorch names.
 */
public final class BSRoformer {

  private static final int NFFT = 2048, HOP = 512, DIM = 256, HEADS = 8, HEAD_DIM = 64, FF = 1024;

  private final Map<String, float[]> w = new HashMap<>();
  private final int[] bandDims; // input width of each band: 2 (re/im) x 2 (stereo) x its bins
  private final int[] bandStarts;
  private final int features;
  private final int layers;
  private final int head;
  private final Stft stft = new Stft(NFFT, HOP, false);

  private BSRoformer(Map<String, Tensor> tensors, int head) {
    this.head = head;
    List<Integer> dims = new ArrayList<>();
    while (tensors.containsKey("band_split.to_features." + dims.size() + ".1.weight")) {
      dims.add(tensors.get("band_split.to_features." + dims.size() + ".1.weight").shape()[1]);
    }
    bandDims = dims.stream().mapToInt(Integer::intValue).toArray();
    bandStarts = new int[bandDims.length];
    int f = 0;
    for (int b = 0; b < bandDims.length; b++) {
      bandStarts[b] = f;
      f += bandDims[b];
    }
    features = f;
    int l = 0;
    while (tensors.containsKey("layers." + l + ".0.layers.0.0.to_qkv.weight")) l++;
    layers = l;
    tensors.forEach((k, v) -> {
      int[] s = v.shape();
      // Linear weights are (out, in); gemm wants (in, out).
      w.put(k, s.length == 2 && k.endsWith(".weight") ? Ops.transpose(v.data(), s[0], s[1]) : v.data());
    });
    w.values().forEach(Ops::constant);
  }

  /**
   * Loads a BS-Roformer checkpoint ({@code .ckpt}). {@code stem} is the index of the stem to separate, in the
   * model's training order (BS-Roformer-SW: bass, drums, other, vocals, guitar, piano).
   */
  public static BSRoformer load(Path checkpoint, int stem) throws IOException {
    TorchCheckpoint ck = TorchCheckpoint.read(checkpoint);
    String mine = "mask_estimators." + stem + ".";
    Map<String, Tensor> tensors = ck.tensors(k -> !k.startsWith("mask_estimators.") || k.startsWith(mine));
    if (tensors.keySet().stream().noneMatch(k -> k.startsWith(mine))) throw new IOException("no stem " + stem + " in " + checkpoint);
    if (tensors.keySet().stream().anyMatch(k -> k.contains("linear_transformer") || k.contains("hyper") || k.contains("skip"))) {
      throw new IOException("unsupported BS-Roformer variant: " + checkpoint);
    }
    return new BSRoformer(tensors, stem);
  }

  private float[] p(String name) {
    float[] v = w.get(name);
    if (v == null) throw new IllegalStateException("missing weight " + name);
    return v;
  }

  /** Separates one chunk {@code [2][n]} into the stem, {@code [2][HOP * frames-1]} (n rounded down to the hop). */
  public float[][] forward(float[][] mix) {
    // STFT of both channels, then per frame the (bin, channel, re/im) interleave the band split expects.
    float[][][] spec = {stft.forward(mix[0]), stft.forward(mix[1])}; // [ch][frame][2*bin + reim]
    int frames = spec[0].length, bins = NFFT / 2 + 1;
    int bands = bandDims.length, tokens = frames * bands;
    float[] feat = new float[frames * features];
    for (int t = 0; t < frames; t++)
      for (int f = 0; f < bins; f++)
        for (int ch = 0; ch < 2; ch++) {
          int o = t * features + (f * 2 + ch) * 2;
          feat[o] = spec[ch][t][2 * f];
          feat[o + 1] = spec[ch][t][2 * f + 1];
        }

    // Band split: per band, RMSNorm then a linear map to 256.
    float[] x = new float[tokens * DIM];
    for (int b = 0; b < bands; b++) {
      int dimIn = bandDims[b], start = bandStarts[b];
      float[] normed = new float[frames * dimIn];
      for (int t = 0; t < frames; t++) System.arraycopy(feat, t * features + start, normed, t * dimIn, dimIn);
      rmsNorm(normed, frames, dimIn, p("band_split.to_features." + b + ".0.gamma"));
      Ops.gemm(frames, DIM, dimIn, normed, 0, dimIn, p("band_split.to_features." + b + ".1.weight"), 0, DIM,
          x, b * DIM, bands * DIM, false);
      float[] bias = p("band_split.to_features." + b + ".1.bias");
      for (int t = 0; t < frames; t++) for (int d = 0; d < DIM; d++) x[(t * bands + b) * DIM + d] += bias[d];
    }

    // Axial transformer: along time within each band, then along bands within each frame.
    for (int l = 0; l < layers; l++) {
      transformer(x, tokens, "layers." + l + ".0", bands, frames, 1, bands); // sequence = band, position = frame
      transformer(x, tokens, "layers." + l + ".1", frames, bands, bands, 1); // sequence = frame, position = band
    }
    rmsNorm(x, tokens, DIM, p("final_norm.gamma"));

    // Mask head: per band, Linear -> tanh -> Linear -> GLU, giving the band's (bin, channel, re/im) mask values.
    float[] mask = new float[frames * features];
    String mh = "mask_estimators." + head + ".to_freqs.";
    for (int b = 0; b < bands; b++) {
      int dimIn = bandDims[b];
      float[] hidden = new float[frames * FF];
      Ops.gemm(frames, FF, DIM, x, b * DIM, bands * DIM, p(mh + b + ".0.0.weight"), 0, FF, hidden, 0, FF, false);
      Ops.addColBias(hidden, frames, FF, p(mh + b + ".0.0.bias"));
      IntStream.range(0, frames).parallel().forEach(t -> {
        for (int j = t * FF; j < (t + 1) * FF; j++) hidden[j] = (float) Math.tanh(hidden[j]);
      });
      float[] y = new float[frames * 2 * dimIn];
      Ops.gemm(frames, 2 * dimIn, FF, hidden, 0, FF, p(mh + b + ".0.2.weight"), 0, 2 * dimIn, y, 0, 2 * dimIn, false);
      Ops.addColBias(y, frames, 2 * dimIn, p(mh + b + ".0.2.bias"));
      int start = bandStarts[b];
      for (int t = 0; t < frames; t++) {
        int yo = t * 2 * dimIn;
        for (int j = 0; j < dimIn; j++) {
          mask[t * features + start + j] = (float) (y[yo + j] / (1 + Math.exp(-y[yo + dimIn + j])));
        }
      }
    }

    // Complex mask times the input spectrogram, then the inverse STFT per channel.
    float[][] out = new float[2][];
    IntStream.range(0, 2).parallel().forEach(ch -> {
      float[][] masked = new float[frames][2 * bins];
      for (int t = 0; t < frames; t++)
        for (int f = 0; f < bins; f++) {
          int o = t * features + (f * 2 + ch) * 2;
          float mr = mask[o], mi = mask[o + 1];
          float sr = spec[ch][t][2 * f], si = spec[ch][t][2 * f + 1];
          masked[t][2 * f] = sr * mr - si * mi;
          masked[t][2 * f + 1] = sr * mi + si * mr;
        }
      out[ch] = stft.inverse(masked, HOP * (frames - 1));
    });
    return out;
  }

  /**
   * One transformer (attention + feed-forward, both residual) over sequences laid out in the token-major x: token
   * of (sequence s, position i) is {@code s * seqStride + i * posStride}.
   */
  private void transformer(float[] x, int tokens, String name, int sequences, int length, int seqStride, int posStride) {
    String a = name + ".layers.0.0", ff = name + ".layers.0.1";
    // Attention.
    float[] xn = x.clone();
    rmsNorm(xn, tokens, DIM, p(a + ".norm.gamma"));
    int inner = HEADS * HEAD_DIM;
    float[] qkv = new float[tokens * 3 * inner];
    Ops.gemm(tokens, 3 * inner, DIM, xn, 0, DIM, p(a + ".to_qkv.weight"), 0, 3 * inner, qkv, 0, 3 * inner, false);
    // Rotary angles depend only on (position, pair): tabulate cos and sin once.
    float[] freqs = p(a + ".rotary_embed.freqs");
    int pairs = HEAD_DIM / 2;
    float[] cos = new float[length * pairs], sin = new float[length * pairs];
    for (int i = 0; i < length; i++)
      for (int j = 0; j < pairs; j++) {
        float angle = i * freqs[j]; // float32, as torch computes it
        cos[i * pairs + j] = (float) Math.cos(angle);
        sin[i * pairs + j] = (float) Math.sin(angle);
      }
    // Gather into (sequence, position, inner) with rotary embeddings applied to q and k.
    int n = sequences * length;
    float[] q = new float[n * inner], k = new float[n * inner], v = new float[n * inner];
    IntStream.range(0, sequences).parallel().forEach(s -> {
      for (int i = 0; i < length; i++) {
        int tok = s * seqStride + i * posStride, src = tok * 3 * inner, dst = (s * length + i) * inner;
        System.arraycopy(qkv, src, q, dst, inner);
        System.arraycopy(qkv, src + inner, k, dst, inner);
        System.arraycopy(qkv, src + 2 * inner, v, dst, inner);
        rope(q, dst, i, cos, sin);
        rope(k, dst, i, cos, sin);
      }
    });
    float[] ctx = new float[n * inner];
    if (!Ops.batchedAttention(sequences, length, HEADS, HEAD_DIM, q, k, v, ctx)) {
      attentionCpu(sequences, length, q, k, v, ctx);
    }
    // Per-head sigmoid gates, then back to token order.
    float[] gates = new float[tokens * HEADS];
    Ops.gemm(tokens, HEADS, DIM, xn, 0, DIM, p(a + ".to_gates.weight"), 0, HEADS, gates, 0, HEADS, false);
    float[] gb = p(a + ".to_gates.bias");
    float[] merged = new float[tokens * inner];
    IntStream.range(0, sequences).parallel().forEach(s -> {
      for (int i = 0; i < length; i++) {
        int tok = s * seqStride + i * posStride, src = (s * length + i) * inner;
        for (int h = 0; h < HEADS; h++) {
          float g = (float) (1 / (1 + Math.exp(-(gates[tok * HEADS + h] + gb[h]))));
          for (int d = 0; d < HEAD_DIM; d++) merged[tok * inner + h * HEAD_DIM + d] = ctx[src + h * HEAD_DIM + d] * g;
        }
      }
    });
    Ops.gemm(tokens, DIM, inner, merged, 0, inner, p(a + ".to_out.0.weight"), 0, DIM, x, 0, DIM, true);

    // Feed-forward.
    float[] h = x.clone();
    rmsNorm(h, tokens, DIM, p(ff + ".net.0.gamma"));
    float[] hidden = new float[tokens * FF];
    Ops.gemm(tokens, FF, DIM, h, 0, DIM, p(ff + ".net.1.weight"), 0, FF, hidden, 0, FF, false);
    Ops.addColBias(hidden, tokens, FF, p(ff + ".net.1.bias"));
    Ops.gelu(hidden);
    float[] y = new float[tokens * DIM];
    Ops.gemm(tokens, DIM, FF, hidden, 0, FF, p(ff + ".net.4.weight"), 0, DIM, y, 0, DIM, false);
    Ops.addColBias(y, tokens, DIM, p(ff + ".net.4.bias"));
    IntStream.range(0, tokens).parallel().forEach(t -> {
      for (int d = t * DIM; d < (t + 1) * DIM; d++) x[d] += y[d];
    });
  }

  // softmax(q k^T / sqrt(d)) v for every (sequence, head), on (sequence, position, heads*headDim) arrays.
  private static void attentionCpu(int sequences, int length, float[] q, float[] k, float[] v, float[] out) {
    int inner = HEADS * HEAD_DIM;
    float scale = (float) (1 / Math.sqrt(HEAD_DIM));
    IntStream.range(0, sequences * HEADS).parallel().forEach(job -> {
      int s = job / HEADS, h = job % HEADS, base = s * length * inner + h * HEAD_DIM;
      float[] kt = new float[HEAD_DIM * length];
      for (int i = 0; i < length; i++)
        for (int d = 0; d < HEAD_DIM; d++) kt[d * length + i] = k[base + i * inner + d];
      float[] scores = new float[length * length];
      Ops.gemmSerial(length, length, HEAD_DIM, q, base, inner, kt, 0, length, scores, 0, length);
      for (int i = 0; i < length; i++) {
        int o = i * length;
        float max = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < length; j++) max = Math.max(max, scores[o + j] *= scale);
        double sum = 0;
        for (int j = 0; j < length; j++) sum += scores[o + j] = (float) Math.exp(scores[o + j] - max);
        float inv = (float) (1 / sum);
        for (int j = 0; j < length; j++) scores[o + j] *= inv;
      }
      Ops.gemmSerial(length, HEAD_DIM, length, scores, 0, length, v, base, inner, out, base, inner);
    });
  }

  // Interleaved rotary embedding on each head of a token: pairs (2i, 2i+1) rotate by position * freqs[i].
  private static void rope(float[] x, int off, int pos, float[] cos, float[] sin) {
    int pairs = HEAD_DIM / 2;
    for (int h = 0; h < HEADS; h++) {
      int o = off + h * HEAD_DIM;
      for (int i = 0; i < pairs; i++) {
        float c = cos[pos * pairs + i], s = sin[pos * pairs + i];
        float a = x[o + 2 * i], b = x[o + 2 * i + 1];
        x[o + 2 * i] = a * c - b * s;
        x[o + 2 * i + 1] = b * c + a * s;
      }
    }
  }

  // RMSNorm as in BS-Roformer: x / max(||x||, 1e-12) * sqrt(dim) * gamma, per row, in place.
  private static void rmsNorm(float[] x, int rows, int dim, float[] gamma) {
    double scale = Math.sqrt(dim);
    IntStream.range(0, rows).parallel().forEach(r -> {
      int o = r * dim;
      double sq = 0;
      for (int d = 0; d < dim; d++) sq += (double) x[o + d] * x[o + d];
      float f = (float) (scale / Math.max(Math.sqrt(sq), 1e-12));
      for (int d = 0; d < dim; d++) x[o + d] *= f * gamma[d];
    });
  }
}
