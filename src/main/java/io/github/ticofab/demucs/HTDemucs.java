package io.github.ticofab.demucs;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * One Demucs v4 hybrid transformer network ({@code HTDemucs}), inference only, for the configuration used by the
 * released {@code htdemucs} / {@code htdemucs_ft} checkpoints: depth 4, 48 channels, CaC spectrogram branch, a
 * 5-layer cross-domain transformer at 512 channels.
 *
 * <p>The network is a U-Net with two branches: one works on the spectrogram (frequency × time, real and imaginary
 * parts as channels), the other on the raw waveform. Each branch has 4 encoder levels that shrink the signal by 4
 * and double the channels; the two meet in a transformer that lets each attend to the other; 4 decoder levels then
 * expand back, using the encoders' outputs as skip connections. The result is the sum of both branches.
 *
 * <p>Activations are flat row-major {@code float[]}s: {@code (channels, freqs, frames)} for the spectrogram branch,
 * {@code (channels, samples)} for the waveform branch, {@code (tokens, 512)} inside the transformer. Weights are
 * looked up by their PyTorch names, so the code can be read side by side with Demucs' {@code htdemucs.py}.
 *
 * <p>{@link #forward} processes exactly one training-length segment (7.8 s at 44.1 kHz); {@link Separator} does the
 * splitting of whole songs.
 */
public final class HTDemucs {

  public static final int SAMPLE_RATE = 44100;
  /** 39/5 s at 44.1 kHz, the training segment length. */
  public static final int SEGMENT = 343_980;

  private static final int NFFT = 4096, HOP = 1024, DEPTH = 4, HEADS = 8, LAYERS = 5;

  private final List<String> sources;
  private final Map<String, float[]> w;
  private final Map<String, int[]> shapes = new HashMap<>();
  private final Stft stft = new Stft(NFFT, HOP);

  private HTDemucs(List<String> sources, Map<String, Tensor> tensors) {
    this.sources = sources;
    this.w = new HashMap<>();
    tensors.forEach((k, v) -> {
      w.put(k, v.data());
      shapes.put(k, v.shape());
    });
    prepareWeights();
    w.values().forEach(Ops::constant);
  }

  /** Loads a Demucs v4 checkpoint ({@code .th}), e.g. {@code d12395a8-e57c48e6.th} (htdemucs_ft's bass model). */
  public static HTDemucs load(Path checkpoint) throws IOException {
    TorchCheckpoint ck = TorchCheckpoint.read(checkpoint);
    Map<?, ?> root = (Map<?, ?>) ck.root();
    Map<?, ?> kwargs = (Map<?, ?>) root.get("kwargs");
    checkConfig(kwargs);
    @SuppressWarnings("unchecked")
    List<String> sources = (List<String>) kwargs.get("sources");
    return new HTDemucs(List.copyOf(sources), ck.tensors("state"));
  }

  private static void checkConfig(Map<?, ?> kw) throws IOException {
    Object[][] expected = {
        {"audio_channels", 2}, {"channels", 48}, {"growth", 2}, {"nfft", 4096}, {"depth", 4}, {"cac", true},
        {"wiener_iters", 0}, {"kernel_size", 8}, {"stride", 4}, {"context", 1}, {"context_enc", 0},
        {"norm_starts", 4}, {"dconv_mode", 3}, {"dconv_depth", 2}, {"dconv_comp", 8}, {"bottom_channels", 512},
        {"t_layers", 5}, {"t_heads", 8}, {"t_norm_in", true}, {"t_norm_first", true}, {"t_norm_out", true},
        {"t_layer_scale", true}, {"t_gelu", true}, {"t_cross_first", false}, {"t_emb", "sin"},
        {"t_sparse_self_attn", false}, {"t_sparse_cross_attn", false}, {"t_sin_random_shift", 0},
        {"samplerate", 44100}, {"rewrite", true},
    };
    for (Object[] e : expected) {
      Object v = kw.get(e[0]);
      if (!e[1].equals(v)) throw new IOException("unsupported model config: " + e[0] + " = " + v + " (expected " + e[1] + ")");
    }
    if (((Number) kw.get("freq_emb")).doubleValue() != 0.2) throw new IOException("unsupported freq_emb");
    if (!((List<?>) kw.get("multi_freqs")).isEmpty()) throw new IOException("unsupported multi_freqs");
  }

  /** Where the large matrix multiplies run: an accelerator's name (e.g. {@code metal}), or {@code java}. */
  public static String backend() {
    return Ops.backendName();
  }

  /** Source names, in output order (e.g. drums, bass, other, vocals). */
  public List<String> sources() {
    return sources;
  }

  // Linear layers and transposed convolutions want transposed weights for gemm; do it once.
  private void prepareWeights() {
    for (String k : List.copyOf(w.keySet())) {
      int[] s = shapes.get(k);
      if (k.endsWith("conv_tr.weight")) {
        int cin = s[0], cout = s[1], kernel = s[2];
        float[] src = w.get(k), dst = new float[src.length];
        for (int ci = 0; ci < cin; ci++)
          for (int co = 0; co < cout; co++)
            for (int kk = 0; kk < kernel; kk++) dst[(co * kernel + kk) * cin + ci] = src[(ci * cout + co) * kernel + kk];
        w.put(k + ".t", dst);
      } else if (k.startsWith("crosstransformer.") && (k.endsWith("linear1.weight") || k.endsWith("linear2.weight")
          || k.endsWith("out_proj.weight"))) {
        w.put(k + ".t", Ops.transpose(w.get(k), s[0], s[1]));
      } else if (k.endsWith("in_proj_weight")) {
        int d = s[1];
        float[] src = w.get(k);
        for (int part = 0; part < 3; part++) {
          float[] slice = new float[d * d];
          System.arraycopy(src, part * d * d, slice, 0, d * d);
          w.put(k + ".t" + part, Ops.transpose(slice, d, d));
        }
      }
    }
  }

  /** A weight by its PyTorch state-dict name, e.g. {@code encoder.0.conv.weight}. */
  private float[] p(String name) {
    float[] v = w.get(name);
    if (v == null) throw new IllegalStateException("missing weight " + name);
    return v;
  }

  /**
   * Separates one segment. {@code mix} is {@code [2][SEGMENT]} (left, right); returns
   * {@code [sources][2][SEGMENT]}.
   */
  public float[][][] forward(float[][] mix) {
    int length = SEGMENT;
    if (mix.length != 2 || mix[0].length != length || mix[1].length != length) {
      throw new IllegalArgumentException("expected a stereo segment of " + length + " samples");
    }

    // Spectrogram branch input: CaC channels [L.re, L.im, R.re, R.im] of (2048 bins, le frames).
    int le = (length + HOP - 1) / HOP;
    int pad = HOP / 2 * 3;
    int bins = NFFT / 2;
    float[] x = new float[4 * bins * le];
    for (int ch = 0; ch < 2; ch++) {
      float[] padded = Stft.reflectPad(mix[ch], 0, length, pad, pad + le * HOP - length);
      float[][] z = stft.forward(padded); // le + 4 frames; Demucs keeps the le in the middle
      for (int t = 0; t < le; t++) {
        float[] frame = z[t + 2];
        for (int f = 0; f < bins; f++) {
          x[((ch * 2) * bins + f) * le + t] = frame[2 * f];
          x[((ch * 2 + 1) * bins + f) * le + t] = frame[2 * f + 1];
        }
      }
    }
    double[] ms = meanStd(x);
    float mean = (float) ms[0], std = (float) ms[1];
    normalize(x, mean, std);

    // Waveform branch input.
    float[] xt = new float[2 * length];
    System.arraycopy(mix[0], 0, xt, 0, length);
    System.arraycopy(mix[1], 0, xt, length, length);
    double[] mst = meanStd(xt);
    float meant = (float) mst[0], stdt = (float) mst[1];
    normalize(xt, meant, stdt);

    // Encoders. Each level: strided conv (length / 4), GELU, DConv residual, 1x1 "rewrite" conv + GLU. Outputs are
    // kept for the decoders' skip connections.
    Deque<float[]> saved = new ArrayDeque<>(), savedT = new ArrayDeque<>();
    Deque<Integer> lengthsT = new ArrayDeque<>();
    int c = 4, ct = 2, fr = bins, lt = length;
    for (int i = 0; i < DEPTH; i++) {
      String e = "encoder." + i, te = "tencoder." + i;
      int cout = 48 << i;
      // time branch
      lengthsT.push(lt);
      int padded = lt % 4 == 0 ? lt : lt + 4 - lt % 4;
      float[] xin = lt == padded ? xt : padTime(xt, ct, lt, padded);
      xt = Ops.conv1d(xin, ct, padded, p(te + ".conv.weight"), p(te + ".conv.bias"), cout, 8, 4, 2, 1);
      lt = padded / 4;
      Ops.gelu(xt);
      xt = dconv(xt, 1, cout, lt, te + ".dconv");
      xt = Ops.glu(Ops.conv1d(xt, cout, lt, p(te + ".rewrite.weight"), p(te + ".rewrite.bias"), 2 * cout, 1, 1, 0, 1),
          2 * cout, lt);
      savedT.push(xt);
      ct = cout;
      // frequency branch
      x = Ops.convFreq(x, c, fr, le, p(e + ".conv.weight"), p(e + ".conv.bias"), cout, 8, 4, 2);
      fr /= 4;
      Ops.gelu(x);
      x = dconvFreq(x, cout, fr, le, e + ".dconv");
      x = Ops.glu(Ops.conv1d(x, cout, fr * le, p(e + ".rewrite.weight"), p(e + ".rewrite.bias"), 2 * cout, 1, 1, 0, 1),
          2 * cout, fr * le);
      c = cout;
      if (i == 0) {
        float[] emb = p("freq_emb.embedding.weight"); // (512 freqs, 48 channels), times scale 10, times 0.2
        for (int ch = 0; ch < c; ch++)
          for (int f = 0; f < fr; f++) {
            float v = 0.2f * 10f * emb[f * c + ch];
            int o = (ch * fr + f) * le;
            for (int t = 0; t < le; t++) x[o + t] += v;
          }
      }
      saved.push(x);
    }

    // Bottleneck: 384 -> 512 channels, cross-domain transformer, back to 384.
    int tc = 512;
    x = Ops.conv1d(x, c, fr * le, p("channel_upsampler.weight"), p("channel_upsampler.bias"), tc, 1, 1, 0, 1);
    xt = Ops.conv1d(xt, ct, lt, p("channel_upsampler_t.weight"), p("channel_upsampler_t.bias"), tc, 1, 1, 0, 1);
    float[][] both = crossTransformer(x, fr, le, xt, lt);
    x = Ops.conv1d(both[0], tc, fr * le, p("channel_downsampler.weight"), p("channel_downsampler.bias"), c, 1, 1, 0, 1);
    xt = Ops.conv1d(both[1], tc, lt, p("channel_downsampler_t.weight"), p("channel_downsampler_t.bias"), ct, 1, 1, 0, 1);

    // Decoders, deepest first. Each level mirrors an encoder: add the skip, rewrite conv (3x3 on the spectrogram,
    // k=3 on the waveform) + GLU, DConv, transposed conv (length * 4), crop the padding, GELU except at the end.
    for (int i = 0; i < DEPTH; i++) {
      int level = DEPTH - 1 - i;
      boolean last = i == DEPTH - 1;
      String d = "decoder." + i, td = "tdecoder." + i;
      int cin = 48 << level;
      int cout = last ? 4 * sources.size() : cin / 2;
      int coutT = last ? 2 * sources.size() : cin / 2;
      // frequency branch
      add(x, saved.pop());
      x = Ops.glu(Ops.conv3x3(x, cin, fr, le, p(d + ".rewrite.weight"), p(d + ".rewrite.bias"), 2 * cin), 2 * cin,
          fr * le);
      x = dconvFreq(x, cin, fr, le, d + ".dconv");
      float[] y = Ops.convTranspose(x, cin, fr, le, p(d + ".conv_tr.weight.t"), p(d + ".conv_tr.bias"), cout, 8, 4);
      int frOut = fr * 4;
      x = cropFreq(y, cout, (fr - 1) * 4 + 8, le, 2, frOut);
      fr = frOut;
      if (!last) Ops.gelu(x);
      // time branch
      add(xt, savedT.pop());
      xt = Ops.glu(Ops.conv1d(xt, cin, lt, p(td + ".rewrite.weight"), p(td + ".rewrite.bias"), 2 * cin, 3, 1, 1, 1),
          2 * cin, lt);
      xt = dconv(xt, 1, cin, lt, td + ".dconv");
      float[] yt = Ops.convTranspose(xt, cin, lt, 1, p(td + ".conv_tr.weight.t"), p(td + ".conv_tr.bias"), coutT, 8, 4);
      int target = lengthsT.pop();
      xt = cropTime(yt, coutT, (lt - 1) * 4 + 8, 2, target);
      lt = target;
      if (!last) Ops.gelu(xt);
    }

    // Outputs: spectrogram branch through the iSTFT, plus the waveform branch.
    int s = sources.size();
    float[] spec = x, wav = xt;
    float[][][] out = new float[s][2][];
    IntStream.range(0, s * 2).parallel().forEach(sc -> {
      int src = sc / 2, ch = sc % 2;
      int reCh = src * 4 + ch * 2, imCh = reCh + 1;
      float[][] frames = new float[le + 4][2 * (bins + 1)];
      for (int t = 0; t < le; t++) {
        float[] frame = frames[t + 2];
        for (int f = 0; f < bins; f++) {
          frame[2 * f] = spec[(reCh * bins + f) * le + t] * std + mean;
          frame[2 * f + 1] = spec[(imCh * bins + f) * le + t] * std + mean;
        }
      }
      int padIs = HOP / 2 * 3;
      int lenIs = HOP * ((length + HOP - 1) / HOP) + 2 * padIs;
      float[] wave = stft.inverse(frames, lenIs);
      float[] res = new float[length];
      int to = (src * 2 + ch) * length;
      for (int n = 0; n < length; n++) res[n] = wave[padIs + n] + wav[to + n] * stdt + meant;
      out[src][ch] = res;
    });
    return out;
  }

  // Cross-domain transformer on (512, fr, le) and (512, lt); returns both in the same layouts.
  private float[][] crossTransformer(float[] x, int fr, int le, float[] xt, int lt) {
    int dim = 512, tokens = fr * le;
    // (c, fr, t1) -> tokens (t1 fr), c
    float[] a = new float[tokens * dim];
    for (int ch = 0; ch < dim; ch++)
      for (int f = 0; f < fr; f++)
        for (int t = 0; t < le; t++) a[(t * fr + f) * dim + ch] = x[(ch * fr + f) * le + t];
    float[] b = Ops.transpose(xt, dim, lt);
    a = Ops.layerNorm(a, tokens, dim, p("crosstransformer.norm_in.weight"), p("crosstransformer.norm_in.bias"));
    b = Ops.layerNorm(b, lt, dim, p("crosstransformer.norm_in_t.weight"), p("crosstransformer.norm_in_t.bias"));
    add2dSinEmbedding(a, dim, fr, le);
    addSinEmbedding(b, dim, lt);
    for (int i = 0; i < LAYERS; i++) {
      String l = "crosstransformer.layers." + i, lt2 = "crosstransformer.layers_t." + i;
      if (i % 2 == 0) {
        a = selfLayer(a, tokens, l);
        b = selfLayer(b, lt, lt2);
      } else {
        float[] oldA = a;
        a = crossLayer(a, tokens, b, lt, l);
        b = crossLayer(b, lt, oldA, tokens, lt2);
      }
    }
    float[] xo = new float[dim * tokens];
    for (int t = 0; t < le; t++)
      for (int f = 0; f < fr; f++)
        for (int ch = 0; ch < dim; ch++) xo[(ch * fr + f) * le + t] = a[(t * fr + f) * dim + ch];
    return new float[][] {xo, Ops.transpose(b, lt, dim)};
  }

  /** Pre-norm transformer layer: self-attention and feed-forward, each scaled and added back, then GroupNorm. */
  private float[] selfLayer(float[] x, int n, String l) {
    int dim = 512;
    float[] h = Ops.layerNorm(x, n, dim, p(l + ".norm1.weight"), p(l + ".norm1.bias"));
    float[] att = attention(h, n, h, n, l + ".self_attn");
    addScaled(x, att, p(l + ".gamma_1.scale"), dim);
    h = Ops.layerNorm(x, n, dim, p(l + ".norm2.weight"), p(l + ".norm2.bias"));
    addScaled(x, feedForward(h, n, l), p(l + ".gamma_2.scale"), dim);
    Ops.groupNorm1Tokens(x, n, dim, p(l + ".norm_out.weight"), p(l + ".norm_out.bias"));
    return x;
  }

  /** Like {@link #selfLayer}, but the queries come from one branch and the keys/values from the other. */
  private float[] crossLayer(float[] q, int nq, float[] k, int nk, String l) {
    int dim = 512;
    float[] x = q.clone();
    float[] hq = Ops.layerNorm(q, nq, dim, p(l + ".norm1.weight"), p(l + ".norm1.bias"));
    float[] hk = Ops.layerNorm(k, nk, dim, p(l + ".norm2.weight"), p(l + ".norm2.bias"));
    addScaled(x, attention(hq, nq, hk, nk, l + ".cross_attn"), p(l + ".gamma_1.scale"), dim);
    float[] h = Ops.layerNorm(x, nq, dim, p(l + ".norm3.weight"), p(l + ".norm3.bias"));
    addScaled(x, feedForward(h, nq, l), p(l + ".gamma_2.scale"), dim);
    Ops.groupNorm1Tokens(x, nq, dim, p(l + ".norm_out.weight"), p(l + ".norm_out.bias"));
    return x;
  }

  // nn.MultiheadAttention (packed in_proj, 8 heads of 64), queries from q, keys and values from kv.
  private float[] attention(float[] q, int nq, float[] kv, int nk, String a) {
    int dim = 512, hd = dim / HEADS;
    float[] bias = p(a + ".in_proj_bias");
    float[] qp = new float[nq * dim], kp = new float[nk * dim], vp = new float[nk * dim];
    Ops.gemm(nq, dim, dim, q, 0, dim, p(a + ".in_proj_weight.t0"), 0, dim, qp, 0, dim, false);
    Ops.gemm(nk, dim, dim, kv, 0, dim, p(a + ".in_proj_weight.t1"), 0, dim, kp, 0, dim, false);
    Ops.gemm(nk, dim, dim, kv, 0, dim, p(a + ".in_proj_weight.t2"), 0, dim, vp, 0, dim, false);
    float scale = (float) (1 / Math.sqrt(hd));
    IntStream.range(0, nq).parallel().forEach(t -> {
      for (int j = 0; j < dim; j++) qp[t * dim + j] = (qp[t * dim + j] + bias[j]) * scale;
    });
    IntStream.range(0, nk).parallel().forEach(t -> {
      for (int j = 0; j < dim; j++) {
        kp[t * dim + j] += bias[dim + j];
        vp[t * dim + j] += bias[2 * dim + j];
      }
    });
    float[] ctx = new float[nq * dim];
    if (!Ops.attention(nq, nk, HEADS, hd, qp, kp, vp, ctx)) attentionCpu(nq, nk, qp, kp, vp, ctx);
    float[] out = new float[nq * dim];
    Ops.gemm(nq, dim, dim, ctx, 0, dim, p(a + ".out_proj.weight.t"), 0, dim, out, 0, dim, false);
    Ops.addColBias(out, nq, dim, p(a + ".out_proj.bias"));
    return out;
  }

  private static void attentionCpu(int nq, int nk, float[] qp, float[] kp, float[] vp, float[] ctx) {
    int dim = 512, hd = dim / HEADS;
    float[] scores = new float[nq * nk];
    for (int h = 0; h < HEADS; h++) {
      int ho = h * hd;
      float[] kt = new float[hd * nk]; // (hd, nk)
      for (int t = 0; t < nk; t++) for (int j = 0; j < hd; j++) kt[j * nk + t] = kp[t * dim + ho + j];
      Ops.gemm(nq, nk, hd, qp, ho, dim, kt, 0, nk, scores, 0, nk, false);
      Ops.softmaxRows(scores, nq, nk);
      Ops.gemm(nq, hd, nk, scores, 0, nk, vp, ho, dim, ctx, ho, dim, false);
    }
  }

  private float[] feedForward(float[] x, int n, String l) {
    int dim = 512, hidden = 2048;
    float[] h = new float[n * hidden];
    Ops.gemm(n, hidden, dim, x, 0, dim, p(l + ".linear1.weight.t"), 0, hidden, h, 0, hidden, false);
    Ops.addColBias(h, n, hidden, p(l + ".linear1.bias"));
    Ops.gelu(h);
    float[] out = new float[n * dim];
    Ops.gemm(n, dim, hidden, h, 0, hidden, p(l + ".linear2.weight.t"), 0, dim, out, 0, dim, false);
    Ops.addColBias(out, n, dim, p(l + ".linear2.bias"));
    return out;
  }

  // x += scale[c] * y over token-major (n, dim).
  private static void addScaled(float[] x, float[] y, float[] scale, int dim) {
    int n = x.length / dim;
    IntStream.range(0, n).parallel().forEach(t -> {
      int o = t * dim;
      for (int c = 0; c < dim; c++) x[o + c] += scale[c] * y[o + c];
    });
  }

  // DConv over (batch, channels, len): two residual blocks of dilated conv, GN, GELU, 1x1 conv, GN, GLU, LayerScale.
  private float[] dconv(float[] x, int batch, int channels, int len, String name) {
    int hidden = channels / 8;
    for (int d = 0; d < 2; d++) {
      String l = name + ".layers." + d;
      int dil = 1 << d;
      float[] h = batch == 1
          ? Ops.conv1d(x, channels, len, p(l + ".0.weight"), p(l + ".0.bias"), hidden, 3, 1, dil, dil)
          : Ops.conv1dBatched(x, batch, channels, len, p(l + ".0.weight"), p(l + ".0.bias"), hidden, 3, dil, dil);
      Ops.groupNorm1(h, batch, hidden, len, p(l + ".1.weight"), p(l + ".1.bias"));
      Ops.gelu(h);
      float[] g = batch == 1
          ? Ops.conv1d(h, hidden, len, p(l + ".3.weight"), p(l + ".3.bias"), 2 * channels, 1, 1, 0, 1)
          : Ops.conv1dBatched(h, batch, hidden, len, p(l + ".3.weight"), p(l + ".3.bias"), 2 * channels, 1, 0, 1);
      Ops.groupNorm1(g, batch, 2 * channels, len, p(l + ".4.weight"), p(l + ".4.bias"));
      float[] scale = p(l + ".6.scale");
      int per = channels * len;
      if (Ops.gatedResidual(x, g, scale, batch, channels, len)) continue;
      IntStream.range(0, batch).parallel().forEach(bi -> {
        int go = bi * 2 * per, xo = bi * per;
        for (int c = 0; c < channels; c++) {
          float sc = scale[c];
          for (int t = 0; t < len; t++) {
            int i = c * len + t;
            float a = g[go + i], b = g[go + per + i];
            x[xo + i] += sc * (float) (a / (1 + Math.exp(-b)));
          }
        }
      });
    }
    return x;
  }

  // DConv on the frequency branch runs along time, independently for every frequency row.
  private float[] dconvFreq(float[] x, int channels, int fr, int le, String name) {
    float[] rows = new float[x.length]; // (fr, channels, le)
    for (int c = 0; c < channels; c++)
      for (int f = 0; f < fr; f++) System.arraycopy(x, (c * fr + f) * le, rows, (f * channels + c) * le, le);
    rows = dconv(rows, fr, channels, le, name);
    float[] out = new float[x.length];
    for (int c = 0; c < channels; c++)
      for (int f = 0; f < fr; f++) System.arraycopy(rows, (f * channels + c) * le, out, (c * fr + f) * le, le);
    return out;
  }

  private static float[] padTime(float[] x, int channels, int len, int padded) {
    float[] out = new float[channels * padded];
    for (int c = 0; c < channels; c++) System.arraycopy(x, c * len, out, c * padded, len);
    return out;
  }

  private static float[] cropFreq(float[] y, int channels, int frIn, int le, int from, int count) {
    float[] out = new float[channels * count * le];
    for (int c = 0; c < channels; c++) System.arraycopy(y, (c * frIn + from) * le, out, c * count * le, count * le);
    return out;
  }

  private static float[] cropTime(float[] y, int channels, int lenIn, int from, int count) {
    float[] out = new float[channels * count];
    for (int c = 0; c < channels; c++) System.arraycopy(y, c * lenIn + from, out, c * count, count);
    return out;
  }

  private static void add(float[] x, float[] y) {
    IntStream.range(0, (x.length + 65535) / 65536).parallel().forEach(blk -> {
      int end = Math.min(x.length, (blk + 1) * 65536);
      for (int i = blk * 65536; i < end; i++) x[i] += y[i];
    });
  }

  // Mean and unbiased standard deviation, as torch.mean / torch.std.
  private static double[] meanStd(float[] x) {
    double sum = 0;
    for (float v : x) sum += v;
    double mean = sum / x.length, sq = 0;
    for (float v : x) sq += (v - mean) * (v - mean);
    return new double[] {mean, Math.sqrt(sq / (x.length - 1))};
  }

  private static void normalize(float[] x, float mean, float std) {
    float inv = 1 / (1e-5f + std);
    for (int i = 0; i < x.length; i++) x[i] = (x[i] - mean) * inv;
  }

  // create_2d_sin_embedding(512, height=fr, width=le), added to tokens (t1 fr).
  private static void add2dSinEmbedding(float[] a, int dim, int fr, int le) {
    int half = dim / 2;
    double[] div = new double[half / 2];
    for (int j = 0; j < half / 2; j++) div[j] = Math.exp(2.0 * j * -(Math.log(10000.0) / half));
    for (int t = 0; t < le; t++)
      for (int f = 0; f < fr; f++) {
        int o = (t * fr + f) * dim;
        for (int j = 0; j < half / 2; j++) {
          a[o + 2 * j] += (float) Math.sin(t * div[j]);
          a[o + 2 * j + 1] += (float) Math.cos(t * div[j]);
          a[o + half + 2 * j] += (float) Math.sin(f * div[j]);
          a[o + half + 2 * j + 1] += (float) Math.cos(f * div[j]);
        }
      }
  }

  // create_sin_embedding(length, 512): [cos(phase), sin(phase)], phase = pos / 10000^(i / (half - 1)).
  private static void addSinEmbedding(float[] b, int dim, int len) {
    int half = dim / 2;
    for (int t = 0; t < len; t++) {
      int o = t * dim;
      for (int i = 0; i < half; i++) {
        double phase = t / Math.pow(10000.0, (double) i / (half - 1));
        b[o + i] += (float) Math.cos(phase);
        b[o + half + i] += (float) Math.sin(phase);
      }
    }
  }
}
