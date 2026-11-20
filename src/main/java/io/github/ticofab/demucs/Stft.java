package io.github.ticofab.demucs;

import java.util.stream.IntStream;

/**
 * Short-time Fourier transform: cuts the signal into overlapping windowed frames and takes each frame's spectrum,
 * giving the (frequency × time) picture the spectrogram branch works on; the inverse puts frames back together.
 *
 * <p>STFT/iSTFT matching {@code torch.stft}/{@code torch.istft} with a periodic Hann window, {@code center=True},
 * reflect padding, and {@code normalized} either way, for a power-of-two {@code nfft}.
 */
final class Stft {

  private final int nfft;
  private final int hop;
  private final float[] window;
  private final double[] cos;
  private final double[] sin;
  private final int[] bitReverse;
  private final boolean normalized;

  /** {@code normalized=true} as in Demucs. */
  Stft(int nfft, int hop) {
    this(nfft, hop, true);
  }

  Stft(int nfft, int hop, boolean normalized) {
    this.normalized = normalized;
    if (Integer.bitCount(nfft) != 1) throw new IllegalArgumentException("nfft must be a power of two: " + nfft);
    this.nfft = nfft;
    this.hop = hop;
    window = new float[nfft];
    for (int i = 0; i < nfft; i++) window[i] = (float) (0.5 - 0.5 * Math.cos(2 * Math.PI * i / nfft));
    cos = new double[nfft / 2];
    sin = new double[nfft / 2];
    for (int i = 0; i < nfft / 2; i++) {
      cos[i] = Math.cos(2 * Math.PI * i / nfft);
      sin[i] = Math.sin(2 * Math.PI * i / nfft);
    }
    int bits = Integer.numberOfTrailingZeros(nfft);
    bitReverse = new int[nfft];
    for (int i = 0; i < nfft; i++) bitReverse[i] = Integer.reverse(i) >>> (32 - bits);
  }

  /** Reflect padding as in {@code F.pad(mode="reflect")} (the edge sample is not repeated). */
  static float[] reflectPad(float[] x, int off, int len, int left, int right) {
    float[] out = new float[left + len + right];
    System.arraycopy(x, off, out, left, len);
    for (int i = 0; i < left; i++) out[left - 1 - i] = x[off + 1 + i];
    for (int i = 0; i < right; i++) out[left + len + i] = x[off + len - 2 - i];
    return out;
  }

  /**
   * Spectrogram of one channel: {@code [frames][2*(nfft/2+1)]} as interleaved (re, im), frames over the centred,
   * reflect-padded signal, scaled by {@code 1/sqrt(nfft)}.
   */
  float[][] forward(float[] signal) {
    float[] x = reflectPad(signal, 0, signal.length, nfft / 2, nfft / 2);
    int frames = 1 + (x.length - nfft) / hop;
    int bins = nfft / 2 + 1;
    float scale = normalized ? (float) (1 / Math.sqrt(nfft)) : 1f;
    float[][] out = new float[frames][];
    IntStream.range(0, frames).parallel().forEach(f -> {
      double[] re = new double[nfft], im = new double[nfft];
      for (int i = 0; i < nfft; i++) re[bitReverse[i]] = x[f * hop + i] * window[i];
      fft(re, im, false);
      float[] z = new float[2 * bins];
      for (int k = 0; k < bins; k++) {
        z[2 * k] = (float) (re[k] * scale);
        z[2 * k + 1] = (float) (im[k] * scale);
      }
      out[f] = z;
    });
    return out;
  }

  /**
   * Inverse of {@link #forward}: overlap-adds windowed inverse FFTs, divides by the summed squared window, drops the
   * centre padding and returns {@code length} samples.
   */
  float[] inverse(float[][] spec, int length) {
    int frames = spec.length;
    int bins = nfft / 2 + 1;
    int full = nfft + hop * (frames - 1);
    float[][] chunks = new float[frames][];
    double scale = (normalized ? Math.sqrt(nfft) : 1) / nfft; // undo the normalization, then the inverse DFT's 1/N
    IntStream.range(0, frames).parallel().forEach(f -> {
      double[] re = new double[nfft], im = new double[nfft];
      float[] z = spec[f];
      for (int k = 0; k < bins; k++) {
        double r = z[2 * k], i = (k == 0 || k == bins - 1) ? 0 : z[2 * k + 1];
        re[bitReverse[k]] = r;
        im[bitReverse[k]] = i;
        if (k > 0 && k < bins - 1) {
          re[bitReverse[nfft - k]] = r;
          im[bitReverse[nfft - k]] = -i;
        }
      }
      fft(re, im, true);
      float[] y = new float[nfft];
      for (int n = 0; n < nfft; n++) y[n] = (float) (re[n] * scale * window[n]);
      chunks[f] = y;
    });
    double[] acc = new double[full], env = new double[full];
    for (int f = 0; f < frames; f++) {
      int o = f * hop;
      for (int n = 0; n < nfft; n++) {
        acc[o + n] += chunks[f][n];
        env[o + n] += (double) window[n] * window[n];
      }
    }
    float[] out = new float[length];
    int start = nfft / 2;
    for (int i = 0; i < length && start + i < full; i++) {
      double e = env[start + i];
      out[i] = (float) (e > 1e-11 ? acc[start + i] / e : 0);
    }
    return out;
  }

  /** In-place iterative radix-2 FFT on bit-reversed input; {@code inverse} flips the twiddle sign (no 1/N). */
  private void fft(double[] re, double[] im, boolean inverse) {
    for (int size = 2; size <= nfft; size <<= 1) {
      int half = size / 2, step = nfft / size;
      for (int s = 0; s < nfft; s += size) {
        for (int j = 0; j < half; j++) {
          double wr = cos[j * step], wi = inverse ? sin[j * step] : -sin[j * step];
          int a = s + j, b = a + half;
          double tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }
}
