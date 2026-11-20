package io.github.ticofab.demucs;

import java.util.List;
import java.util.Random;
import java.util.function.DoubleConsumer;

/**
 * Separates whole songs with one {@link HTDemucs} network, following Demucs' {@code apply_model}: normalize by the
 * song's mean and standard deviation, average {@code shifts} randomly time-shifted passes, and split each pass into
 * segments with 25% overlap blended by triangular weights.
 */
public final class Separator {

  /** Maximum random time shift: half a second. */
  public static final int MAX_SHIFT = HTDemucs.SAMPLE_RATE / 2;

  private final HTDemucs model;
  private final int shifts;
  private final Random random;

  /**
   * @param shifts number of randomly shifted passes to average (Demucs' default in audio-separator is 2; 0 or 1 means
   *               a single pass, 0 without any shift)
   * @param random source of the shift offsets; pass a seeded one for reproducible output
   */
  public Separator(HTDemucs model, int shifts, Random random) {
    this.model = model;
    this.shifts = shifts;
    this.random = random;
  }

  /** The model's source names, in the order {@link #separate} returns them. */
  public List<String> sources() {
    return model.sources();
  }

  /**
   * Separates a stereo song {@code [2][n]} at 44.1 kHz into {@code [sources][2][n]}. {@code progress} receives the
   * completed fraction (0 to 1) after each segment.
   */
  public float[][][] separate(float[][] mix, DoubleConsumer progress) {
    int n = mix[0].length;
    // Whole-song normalization on the mono average (unbiased std, as torch.std).
    double sum = 0;
    for (int i = 0; i < n; i++) sum += (mix[0][i] + mix[1][i]) * 0.5;
    double mean = sum / n, sq = 0;
    for (int i = 0; i < n; i++) {
      double d = (mix[0][i] + mix[1][i]) * 0.5 - mean;
      sq += d * d;
    }
    double std = Math.max(Math.sqrt(sq / Math.max(1, n - 1)), 1.1920929e-07);
    float[][] norm = new float[2][n];
    for (int c = 0; c < 2; c++) for (int i = 0; i < n; i++) norm[c][i] = (float) ((mix[c][i] - mean) / std);

    int s = model.sources().size();
    int passes = Math.max(1, shifts);
    int segmentsPerPass = (int) Math.ceil((double) (n + MAX_SHIFT) / stride());
    int[] done = {0};
    int total = passes * segmentsPerPass;
    DoubleConsumer step = ignored -> progress.accept(Math.min(1.0, (double) ++done[0] / total));

    float[][][] out = new float[s][2][n];
    // Shifting the input by a random fraction of a second and averaging makes the result less dependent on where the
    // segment boundaries fall; it is Demucs' "shift trick".
    if (shifts == 0) {
      accumulate(split(norm, 0, n, step), out, 0, 1f);
    } else {
      // Pad by MAX_SHIFT on both sides; each pass reads [offset, offset + n + MAX_SHIFT - offset).
      float[][] padded = new float[2][n + 2 * MAX_SHIFT];
      for (int c = 0; c < 2; c++) System.arraycopy(norm[c], 0, padded[c], MAX_SHIFT, n);
      Chunk whole = new Chunk(padded, 0, padded[0].length);
      for (int k = 0; k < shifts; k++) {
        int offset = random.nextInt(MAX_SHIFT + 1);
        int len = n + MAX_SHIFT - offset;
        float[][][] res = split(whole.sub(offset, len), len, step);
        accumulate(res, out, MAX_SHIFT - offset, 1f / shifts);
      }
    }
    for (float[][] src : out)
      for (float[] ch : src)
        for (int i = 0; i < n; i++) ch[i] = (float) (ch[i] * std + mean);
    return out;
  }

  private static int stride() {
    return (int) (0.75 * HTDemucs.SEGMENT);
  }

  // out[..., i] += weight * res[..., from + i]
  private static void accumulate(float[][][] res, float[][][] out, int from, float weight) {
    for (int s = 0; s < out.length; s++)
      for (int c = 0; c < 2; c++)
        for (int i = 0; i < out[s][c].length; i++) out[s][c][i] += weight * res[s][c][from + i];
  }

  private float[][][] split(float[][] mix, int offset, int length, DoubleConsumer step) {
    return split(new Chunk(mix, offset, length), length, step);
  }

  // Overlapping segments with triangular weights, as apply_model(split=True, overlap=0.25).
  private float[][][] split(Chunk chunk, int length, DoubleConsumer step) {
    int segment = HTDemucs.SEGMENT, stride = stride(), s = model.sources().size();
    float[] weight = new float[segment];
    int halfSeg = segment / 2;
    for (int i = 0; i < halfSeg; i++) weight[i] = (i + 1f) / halfSeg;
    for (int i = halfSeg; i < segment; i++) weight[i] = (float) (segment - i) / halfSeg;
    float[][][] out = new float[s][2][length];
    float[] sumWeight = new float[length];
    for (int offset = 0; offset < length; offset += stride) {
      Chunk piece = chunk.sub(offset, Math.min(segment, length - offset));
      float[][][] res = model.forward(piece.padded(segment));
      int n = piece.length, trim = (segment - n) / 2;
      for (int src = 0; src < s; src++)
        for (int c = 0; c < 2; c++)
          for (int i = 0; i < n; i++) out[src][c][offset + i] += weight[i] * res[src][c][trim + i];
      for (int i = 0; i < n; i++) sumWeight[offset + i] += weight[i];
      step.accept(0);
    }
    for (float[][] src : out) for (float[] ch : src) for (int i = 0; i < length; i++) ch[i] /= sumWeight[i];
    return out;
  }

  /** A window onto an underlying signal; padding takes real samples from the signal where they exist (TensorChunk). */
  private record Chunk(float[][] signal, int offset, int length) {
    Chunk sub(int start, int len) {
      return new Chunk(signal, offset + start, len);
    }

    float[][] padded(int target) {
      int delta = target - length;
      int start = offset - delta / 2;
      float[][] out = new float[2][target];
      int total = signal[0].length;
      for (int c = 0; c < 2; c++) {
        int from = Math.max(0, start), to = Math.min(total, start + target);
        if (to > from) System.arraycopy(signal[c], from, out[c], from - start, to - from);
      }
      return out;
    }
  }
}
