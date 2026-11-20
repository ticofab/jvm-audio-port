package io.github.ticofab.demucs;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleConsumer;

/**
 * Separates whole songs with a {@link BSRoformer}, as audio-separator's MDXC path does: peak-limit the mix to 0.9,
 * run overlapping chunks (the last one aligned to the song's end), blend them with a Hamming-weighted average, and
 * peak-limit the stem to 0.9.
 */
public final class RoformerSeparator implements StemSeparator {

  /** 512 × (801 − 1) samples, 9.3 s: audio-separator's chunk for dim_t 801. */
  public static final int CHUNK = 409_600;
  /** For songs under 10 s, audio-separator uses dim_t 256: 512 × 255 samples. */
  public static final int SHORT_CHUNK = 130_560;
  private static final int OVERLAP = 2;
  private static final float PEAK = 0.9f;

  private final BSRoformer model;

  public RoformerSeparator(BSRoformer model) {
    this.model = model;
  }

  /** Separates a stereo song {@code [2][n]} at 44.1 kHz into the model's stem, {@code [2][n]}. */
  @Override
  public float[][] separate(float[][] input, DoubleConsumer progress) {
    int n = input[0].length;
    float[][] mix = {input[0].clone(), input[1].clone()};
    limitPeak(mix);
    int chunk = n < 10 * HTDemucs.SAMPLE_RATE ? SHORT_CHUNK : CHUNK;
    int step = chunk / OVERLAP;
    float[] window = new float[chunk];
    for (int i = 0; i < chunk; i++) window[i] = (float) (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (chunk - 1)));

    List<Integer> starts = chunkStarts(n, chunk, step);
    float[][] result = new float[2][n];
    float[] counter = new float[n];
    for (int c = 0; c < starts.size(); c++) {
      int start = starts.get(c), len = Math.min(chunk, n - start);
      float[][] part = {new float[len], new float[len]};
      for (int ch = 0; ch < 2; ch++) System.arraycopy(mix[ch], start, part[ch], 0, len);
      float[][] out = model.forward(part);
      int safe = Math.min(len, out[0].length);
      for (int ch = 0; ch < 2; ch++) for (int i = 0; i < safe; i++) result[ch][start + i] += out[ch][i] * window[i];
      for (int i = 0; i < safe; i++) counter[start + i] += window[i];
      progress.accept((c + 1.0) / starts.size());
    }
    for (int ch = 0; ch < 2; ch++) for (int i = 0; i < n; i++) result[ch][i] /= Math.max(counter[i], 1e-10f);
    limitPeak(result);
    return result;
  }

  // 0, step, 2·step, … until a chunk would reach the end; that one is replaced by the chunk ending at the song's end.
  static List<Integer> chunkStarts(int length, int chunk, int step) {
    List<Integer> starts = new ArrayList<>();
    for (int offset = 0; offset < length; offset += step) {
      if (offset + chunk >= length) {
        int tail = Math.max(length - chunk, 0);
        if (starts.isEmpty() || starts.getLast() != tail) starts.add(tail);
        break;
      }
      starts.add(offset);
    }
    return starts;
  }

  // Scales everything down so the peak is 0.9, if it is above; never amplifies.
  private static void limitPeak(float[][] audio) {
    float peak = 0;
    for (float[] ch : audio) for (float v : ch) peak = Math.max(peak, Math.abs(v));
    if (peak > PEAK) {
      float g = PEAK / peak;
      for (float[] ch : audio) for (int i = 0; i < ch.length; i++) ch[i] *= g;
    }
  }
}
