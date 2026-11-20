package io.github.ticofab.demucs;

import java.util.stream.IntStream;

/**
 * Blends two stems at a crossover frequency: the lows of one and the highs of the other, split by a zero-phase
 * Butterworth lowpass. Matches scipy's {@code butter(order, hz, "low", fs=..., output="sos")} and
 * {@code sosfiltfilt} (odd padding, steady-state initial conditions), so results equal splitter's former Python.
 */
public final class Crossover {

  private Crossover() {}

  /**
   * {@code lowpass(low) + (high - lowpass(high))} per channel: complementary, so the bands sum back without phase
   * smear. Inputs are {@code [channels][frames]}; the output has the shorter length.
   */
  public static float[][] blend(float[][] low, float[][] high, double hz, int sampleRate, int order) {
    double[][] sos = butterLowpass(order, hz, sampleRate);
    int n = Math.min(low[0].length, high[0].length);
    float[][] out = new float[low.length][];
    IntStream.range(0, low.length).parallel().forEach(ch -> {
      double[] l = new double[n], h = new double[n];
      for (int i = 0; i < n; i++) {
        l[i] = low[ch][i];
        h[i] = high[ch][i];
      }
      double[] ll = sosfiltfilt(sos, l), hl = sosfiltfilt(sos, h);
      float[] res = new float[n];
      for (int i = 0; i < n; i++) res[i] = (float) (ll[i] + (h[i] - hl[i]));
      out[ch] = res;
    });
    return out;
  }

  /**
   * Digital Butterworth lowpass as second-order sections {@code [b0, b1, b2, 1, a1, a2]}, designed like scipy: analog
   * prototype, prewarped bilinear transform, and sections ordered with the poles nearest the unit circle last and
   * the gain in the first. Even orders only.
   */
  static double[][] butterLowpass(int order, double hz, double fs) {
    if (order <= 0 || order % 2 != 0) throw new IllegalArgumentException("even order expected: " + order);
    double wn = 2 * hz / fs;
    double warped = 4 * Math.tan(Math.PI * wn / 2); // scipy works at fs = 2, so 2 * fs = 4
    int sections = order / 2;
    // Upper-half-plane analog poles -exp(i·π·m / 2N), m = 1, 3, ..., N-1, scaled to the cutoff, then z = (4 + p) / (4 - p).
    double[][] poles = new double[sections][2];
    double gainRe = Math.pow(warped, order), gainIm = 0;
    for (int s = 0; s < sections; s++) {
      double angle = Math.PI * (2 * s + 1) / (2.0 * order);
      double pr = -Math.cos(angle) * warped, pi = -Math.sin(angle) * warped;
      double nr = 4 + pr, ni = pi, dr = 4 - pr, di = -pi, d = dr * dr + di * di;
      poles[s][0] = (nr * dr + ni * di) / d;
      poles[s][1] = (ni * dr - nr * di) / d;
      // The gain divides by prod(4 - p) over all poles: this one and its conjugate together contribute |4 - p|².
      gainRe /= d;
    }
    // scipy fills sections from the pole nearest the unit circle and then reverses: nearest goes last.
    java.util.Arrays.sort(poles, (x, y) -> Double.compare(
        Math.abs(1 - Math.hypot(y[0], y[1])), Math.abs(1 - Math.hypot(x[0], x[1]))));
    double[][] sos = new double[sections][];
    for (int s = 0; s < sections; s++) {
      double g = s == 0 ? gainRe : 1;
      double re = poles[s][0], im = poles[s][1];
      sos[s] = new double[] {g, 2 * g, g, 1, -2 * re, re * re + im * im}; // zeros at -1, -1
    }
    return sos;
  }

  /** Forward-backward filtering with scipy's defaults: odd extension by {@code 3 * (2 * sections + 1)} samples. */
  static double[] sosfiltfilt(double[][] sos, double[] x) {
    int pad = 3 * (2 * sos.length + 1);
    int n = x.length;
    if (n <= pad) throw new IllegalArgumentException("signal too short for filtfilt: " + n);
    double[] ext = new double[n + 2 * pad];
    for (int i = 0; i < pad; i++) {
      ext[i] = 2 * x[0] - x[pad - i];
      ext[pad + n + i] = 2 * x[n - 1] - x[n - 2 - i];
    }
    System.arraycopy(x, 0, ext, pad, n);
    double[][] zi = sosfiltZi(sos);
    double[] y = sosfilt(sos, ext, zi, ext[0]);
    reverse(y);
    y = sosfilt(sos, y, zi, y[0]);
    reverse(y);
    double[] out = new double[n];
    System.arraycopy(y, pad, out, 0, n);
    return out;
  }

  // Cascade of transposed direct-form II sections, starting from the steady state for a constant input x0.
  private static double[] sosfilt(double[][] sos, double[] x, double[][] zi, double x0) {
    double[] y = x.clone();
    for (int s = 0; s < sos.length; s++) {
      double b0 = sos[s][0], b1 = sos[s][1], b2 = sos[s][2], a1 = sos[s][4], a2 = sos[s][5];
      double z0 = zi[s][0] * x0, z1 = zi[s][1] * x0;
      for (int i = 0; i < y.length; i++) {
        double in = y[i], out = b0 * in + z0;
        z0 = b1 * in - a1 * out + z1;
        z1 = b2 * in - a2 * out;
        y[i] = out;
      }
    }
    return y;
  }

  // scipy.signal.sosfilt_zi: each section's lfilter_zi, scaled by the DC gain of the sections before it.
  private static double[][] sosfiltZi(double[][] sos) {
    double[][] zi = new double[sos.length][2];
    double scale = 1;
    for (int s = 0; s < sos.length; s++) {
      double b0 = sos[s][0], b1 = sos[s][1], b2 = sos[s][2], a1 = sos[s][4], a2 = sos[s][5];
      // Solve [[1 + a1, -1], [a2, 1]] · zi = [b1 - a1·b0, b2 - a2·b0].
      double r0 = b1 - a1 * b0, r1 = b2 - a2 * b0;
      double det = (1 + a1) + a2;
      double z0 = (r0 + r1) / det;
      double z1 = r1 - a2 * z0;
      zi[s][0] = scale * z0;
      zi[s][1] = scale * z1;
      scale *= (b0 + b1 + b2) / (1 + a1 + a2);
    }
    return zi;
  }

  private static void reverse(double[] a) {
    for (int i = 0, j = a.length - 1; i < j; i++, j--) {
      double t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }
}
