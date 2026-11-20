package io.github.ticofab.demucs;

import io.github.ticofab.demucs.spi.Accelerator;
import java.util.ServiceLoader;
import java.util.stream.IntStream;

/**
 * The numeric kernels HTDemucs needs, on plain row-major {@code float[]} arrays.
 *
 * <p>Everything heavy goes through {@link #gemm}: convolutions via im2col, attention and linear layers. Its inner
 * loop runs over contiguous columns so the JIT can vectorize it, and work is split across cores.
 */
final class Ops {

  private Ops() {}

  /** The accelerator in use, or null for pure Java. */
  static final Accelerator BACKEND = loadBackend();

  private static Accelerator loadBackend() {
    String wanted = System.getProperty("demucs.backend", "");
    if (wanted.equals("java")) return null;
    for (Accelerator b : ServiceLoader.load(Accelerator.class)) {
      if ((wanted.isEmpty() || wanted.equals(b.name())) && b.available()) return b;
    }
    if (!wanted.isEmpty()) throw new IllegalStateException("demucs.backend=" + wanted + " is not available");
    return null;
  }

  /** Name of the backend running the large matrix multiplies: a backend's name, or {@code java}. */
  static String backendName() {
    return BACKEND == null ? "java" : BACKEND.name();
  }

  /** Attention on the accelerator if it supports it; false means the caller computes it. */
  static boolean attention(int nq, int nk, int heads, int headDim, float[] q, float[] k, float[] v, float[] out) {
    return BACKEND != null && BACKEND.attention(nq, nk, heads, headDim, q, k, v, out);
  }

  /** {@link io.github.ticofab.demucs.spi.Accelerator#batchedAttention} on the accelerator, if it takes it. */
  static boolean batchedAttention(int batch, int length, int heads, int headDim, float[] q, float[] k, float[] v,
                                  float[] out) {
    return BACKEND != null && BACKEND.batchedAttention(batch, length, heads, headDim, q, k, v, out);
  }

  /** {@link io.github.ticofab.demucs.spi.Accelerator#gatedResidual} on the accelerator, if it takes it. */
  static boolean gatedResidual(float[] x, float[] g, float[] scale, int batch, int channels, int len) {
    return BACKEND != null && x.length >= BACKEND.minElements()
        && BACKEND.gatedResidual(x, g, scale, batch, channels, len);
  }

  /** The accelerator's device API, or null when there is none. */
  static io.github.ticofab.demucs.spi.Device device() {
    return BACKEND == null ? null : BACKEND.device();
  }

  static void constant(float[] array) {
    if (BACKEND != null) BACKEND.constant(array);
  }

  private static final int COL_TILE = 1024;
  private static final int K_BLOCK = 256;
  private static final int ROW_BLOCK = 4;

  /**
   * {@code C = A·B} (or {@code C += A·B} when {@code accumulate}), with A {@code m×k}, B {@code k×n}, C {@code m×n},
   * all row-major with the given leading dimensions and offsets.
   */
  static void gemm(int m, int n, int k,
                   float[] a, int aOff, int lda,
                   float[] b, int bOff, int ldb,
                   float[] c, int cOff, int ldc,
                   boolean accumulate) {
    int rowBlocks = (m + ROW_BLOCK - 1) / ROW_BLOCK;
    int colTiles = (n + COL_TILE - 1) / COL_TILE;
    long flops = 2L * m * n * k;
    if (BACKEND != null && flops >= BACKEND.minFlops()) {
      BACKEND.gemm(m, n, k, a, aOff, lda, b, bOff, ldb, c, cOff, ldc, accumulate);
      return;
    }
    IntStream tasks = IntStream.range(0, rowBlocks * colTiles);
    if (flops > 4_000_000L) tasks = tasks.parallel();
    tasks.forEach(task -> {
      int i0 = (task / colTiles) * ROW_BLOCK, i1 = Math.min(m, i0 + ROW_BLOCK);
      int j0 = (task % colTiles) * COL_TILE, j1 = Math.min(n, j0 + COL_TILE);
      gemmTile(i0, i1, j0, j1, k, a, aOff, lda, b, bOff, ldb, c, cOff, ldc, accumulate);
    });
  }

  /** {@link #gemm} on the calling thread only, for callers that parallelize over many small products. */
  static void gemmSerial(int m, int n, int k, float[] a, int aOff, int lda, float[] b, int bOff, int ldb,
                         float[] c, int cOff, int ldc) {
    gemmTile(0, m, 0, n, k, a, aOff, lda, b, bOff, ldb, c, cOff, ldc, false);
  }

  // One tile of C. The k loop is unrolled by 4 so each pass over a row of C does four multiply-adds per load and
  // store; the innermost j loop runs over contiguous memory, which is what the JIT turns into SIMD instructions.
  private static void gemmTile(int i0, int i1, int j0, int j1, int k,
                               float[] a, int aOff, int lda,
                               float[] b, int bOff, int ldb,
                               float[] c, int cOff, int ldc,
                               boolean accumulate) {
    if (!accumulate) {
      for (int i = i0; i < i1; i++) java.util.Arrays.fill(c, cOff + i * ldc + j0, cOff + i * ldc + j1, 0f);
    }
    for (int p0 = 0; p0 < k; p0 += K_BLOCK) {
      int p1 = Math.min(k, p0 + K_BLOCK);
      for (int i = i0; i < i1; i++) {
        int ci = cOff + i * ldc;
        int ai = aOff + i * lda;
        int p = p0;
        for (; p + 3 < p1; p += 4) {
          float a0 = a[ai + p], a1 = a[ai + p + 1], a2 = a[ai + p + 2], a3 = a[ai + p + 3];
          int b0 = bOff + p * ldb, b1 = b0 + ldb, b2 = b1 + ldb, b3 = b2 + ldb;
          for (int j = j0; j < j1; j++) {
            c[ci + j] += a0 * b[b0 + j] + a1 * b[b1 + j] + a2 * b[b2 + j] + a3 * b[b3 + j];
          }
        }
        for (; p < p1; p++) {
          float a0 = a[ai + p];
          int b0 = bOff + p * ldb;
          for (int j = j0; j < j1; j++) c[ci + j] += a0 * b[b0 + j];
        }
      }
    }
  }

  /** Adds {@code bias[r]} to every element of row {@code r} of a {@code rows×cols} matrix. */
  static void addRowBias(float[] x, int rows, int cols, float[] bias) {
    for (int r = 0; r < rows; r++) {
      float v = bias[r];
      int o = r * cols;
      for (int j = 0; j < cols; j++) x[o + j] += v;
    }
  }

  /** Adds {@code bias[j]} to column {@code j} of every row of a {@code rows×cols} matrix. */
  static void addColBias(float[] x, int rows, int cols, float[] bias) {
    for (int r = 0; r < rows; r++) {
      int o = r * cols;
      for (int j = 0; j < cols; j++) x[o + j] += bias[j];
    }
  }

  /**
   * 1D convolution: x {@code (cin, len)} with weight {@code (cout, cin, kernel)}, zero padding, stride and dilation.
   * Returns {@code (cout, outLen)}.
   */
  static float[] conv1d(float[] x, int cin, int len, float[] w, float[] bias, int cout, int kernel,
                        int stride, int pad, int dilation) {
    int outLen = (len + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    float[] out = new float[cout * outLen];
    if (kernel == 1 && stride == 1 && pad == 0) {
      gemm(cout, outLen, cin, w, 0, cin, x, 0, len, out, 0, outLen, false);
    } else {
      int rows = cin * kernel;
      float[] col = new float[rows * outLen];
      IntStream.range(0, rows).parallel().forEach(r -> {
        int ci = r / kernel, kk = r % kernel;
        int src = ci * len, dst = r * outLen;
        for (int l = 0; l < outLen; l++) {
          int t = l * stride + kk * dilation - pad;
          col[dst + l] = t >= 0 && t < len ? x[src + t] : 0f;
        }
      });
      gemm(cout, outLen, rows, w, 0, rows, col, 0, outLen, out, 0, outLen, false);
    }
    if (bias != null) addRowBias(out, cout, outLen, bias);
    return out;
  }

  /**
   * The same 1D convolution applied independently to {@code batch} rows of {@code (cin, len)}, in parallel over the
   * batch (each one is small).
   */
  static float[] conv1dBatched(float[] x, int batch, int cin, int len, float[] w, float[] bias, int cout, int kernel,
                               int pad, int dilation) {
    int outLen = len + 2 * pad - dilation * (kernel - 1);
    float[] out = new float[batch * cout * outLen];
    IntStream.range(0, batch).parallel().forEach(bi -> {
      int rows = cin * kernel;
      float[] col = new float[rows * outLen];
      for (int r = 0; r < rows; r++) {
        int ci = r / kernel, kk = r % kernel;
        int src = bi * cin * len + ci * len, dst = r * outLen;
        for (int l = 0; l < outLen; l++) {
          int t = l + kk * dilation - pad;
          col[dst + l] = t >= 0 && t < len ? x[src + t] : 0f;
        }
      }
      int o = bi * cout * outLen;
      gemmTile(0, cout, 0, outLen, rows, w, 0, rows, col, 0, outLen, out, o, outLen, false);
      for (int c = 0; c < cout; c++) {
        float v = bias[c];
        for (int l = 0; l < outLen; l++) out[o + c * outLen + l] += v;
      }
    });
    return out;
  }

  /**
   * Convolution along the frequency axis of {@code (cin, freqs, time)}, kernel {@code [kernel, 1]}, stride
   * {@code [stride, 1]}, zero padding {@code [pad, 0]}. Returns {@code (cout, outFreqs, time)}.
   */
  static float[] convFreq(float[] x, int cin, int freqs, int time, float[] w, float[] bias, int cout, int kernel,
                          int stride, int pad) {
    int outF = (freqs + 2 * pad - kernel) / stride + 1;
    int n = outF * time, rows = cin * kernel;
    float[] col = new float[rows * n];
    IntStream.range(0, rows).parallel().forEach(r -> {
      int ci = r / kernel, kk = r % kernel;
      for (int f = 0; f < outF; f++) {
        int sf = f * stride + kk - pad;
        int dst = r * n + f * time;
        if (sf < 0 || sf >= freqs) java.util.Arrays.fill(col, dst, dst + time, 0f);
        else System.arraycopy(x, (ci * freqs + sf) * time, col, dst, time);
      }
    });
    float[] out = new float[cout * n];
    gemm(cout, n, rows, w, 0, rows, col, 0, n, out, 0, n, false);
    addRowBias(out, cout, n, bias);
    return out;
  }

  /** 3×3 convolution with padding 1 over {@code (cin, freqs, time)}, tiled over frequency to bound memory. */
  static float[] conv3x3(float[] x, int cin, int freqs, int time, float[] w, float[] bias, int cout) {
    int rows = cin * 9, total = freqs * time;
    float[] out = new float[cout * total];
    int tileF = Math.max(1, Math.min(freqs, 8_000_000 / (rows * time)));
    for (int f0 = 0; f0 < freqs; f0 += tileF) {
      int f1 = Math.min(freqs, f0 + tileF), fs = f0;
      int n = (f1 - f0) * time;
      float[] col = new float[rows * n];
      IntStream.range(0, rows).parallel().forEach(r -> {
        int ci = r / 9, kf = (r % 9) / 3 - 1, kt = r % 3 - 1;
        for (int f = fs; f < f1; f++) {
          int sf = f + kf, dst = r * n + (f - fs) * time;
          if (sf < 0 || sf >= freqs) {
            java.util.Arrays.fill(col, dst, dst + time, 0f);
            continue;
          }
          int src = (ci * freqs + sf) * time;
          for (int t = 0; t < time; t++) {
            int st = t + kt;
            col[dst + t] = st >= 0 && st < time ? x[src + st] : 0f;
          }
        }
      });
      gemm(cout, n, rows, w, 0, rows, col, 0, n, out, f0 * time, total, false);
    }
    addRowBias(out, cout, total, bias);
    return out;
  }

  /**
   * Transposed convolution along the first spatial axis: x {@code (cin, len, inner)}, weight {@code (cin, cout,
   * kernel)} pre-transposed to {@code wt = (cout*kernel, cin)}. Returns {@code (cout, (len-1)*stride+kernel, inner)}.
   */
  static float[] convTranspose(float[] x, int cin, int len, int inner, float[] wt, float[] bias, int cout,
                               int kernel, int stride) {
    int n = len * inner;
    float[] z = new float[cout * kernel * n];
    gemm(cout * kernel, n, cin, wt, 0, cin, x, 0, n, z, 0, n, false);
    int outLen = (len - 1) * stride + kernel;
    float[] out = new float[cout * outLen * inner];
    IntStream.range(0, cout).parallel().forEach(co -> {
      int o = co * outLen * inner;
      for (int kk = 0; kk < kernel; kk++) {
        int zr = (co * kernel + kk) * n;
        for (int l = 0; l < len; l++) {
          int dst = o + (l * stride + kk) * inner, src = zr + l * inner;
          for (int i = 0; i < inner; i++) out[dst + i] += z[src + i];
        }
      }
      float v = bias[co];
      for (int i = 0; i < outLen * inner; i++) out[o + i] += v;
    });
    return out;
  }

  /** Transposes a {@code (rows, cols)} matrix into a new {@code (cols, rows)} one. */
  static float[] transpose(float[] x, int rows, int cols) {
    float[] out = new float[rows * cols];
    for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) out[c * rows + r] = x[r * cols + c];
    return out;
  }

  /** {@code GroupNorm(1, channels)} over each of {@code batch} items of {@code (channels, len)}; biased variance. */
  static void groupNorm1(float[] x, int batch, int channels, int len, float[] w, float[] b) {
    IntStream range = IntStream.range(0, batch);
    if (batch > 1) range = range.parallel();
    range.forEach(bi -> {
      int o = bi * channels * len, n = channels * len;
      double sum = 0, sq = 0;
      for (int i = 0; i < n; i++) {
        double v = x[o + i];
        sum += v;
        sq += v * v;
      }
      double mean = sum / n, var = Math.max(0, sq / n - mean * mean);
      float inv = (float) (1 / Math.sqrt(var + 1e-5)), m = (float) mean;
      for (int c = 0; c < channels; c++) {
        float scale = inv * w[c], shift = b[c] - m * scale;
        int co = o + c * len;
        for (int i = 0; i < len; i++) x[co + i] = x[co + i] * scale + shift;
      }
    });
  }

  /** {@code GroupNorm(1, channels)} on token-major {@code (tokens, channels)} data: one mean/var over everything. */
  static void groupNorm1Tokens(float[] x, int tokens, int channels, float[] w, float[] b) {
    double sum = 0, sq = 0;
    for (float v : x) {
      sum += v;
      sq += (double) v * v;
    }
    int n = tokens * channels;
    double mean = sum / n, var = Math.max(0, sq / n - mean * mean);
    float inv = (float) (1 / Math.sqrt(var + 1e-5)), m = (float) mean;
    IntStream.range(0, tokens).parallel().forEach(t -> {
      int o = t * channels;
      for (int c = 0; c < channels; c++) x[o + c] = (x[o + c] - m) * inv * w[c] + b[c];
    });
  }

  /** LayerNorm over the last axis of {@code (tokens, channels)}, into a new array. */
  static float[] layerNorm(float[] x, int tokens, int channels, float[] w, float[] b) {
    float[] out = new float[x.length];
    IntStream.range(0, tokens).parallel().forEach(t -> {
      int o = t * channels;
      double sum = 0, sq = 0;
      for (int c = 0; c < channels; c++) {
        double v = x[o + c];
        sum += v;
        sq += v * v;
      }
      double mean = sum / channels, var = Math.max(0, sq / channels - mean * mean);
      float inv = (float) (1 / Math.sqrt(var + 1e-5)), m = (float) mean;
      for (int c = 0; c < channels; c++) out[o + c] = (x[o + c] - m) * inv * w[c] + b[c];
    });
    return out;
  }

  /** Exact (erf-based) GELU, in place. */
  static void gelu(float[] x) {
    if (BACKEND != null && x.length >= BACKEND.minElements() && BACKEND.gelu(x)) return;
    IntStream.range(0, (x.length + 65535) / 65536).parallel().forEach(blk -> {
      int end = Math.min(x.length, (blk + 1) * 65536);
      for (int i = blk * 65536; i < end; i++) {
        float v = x[i];
        x[i] = (float) (0.5 * v * (1 + erf(v * 0.7071067811865476)));
      }
    });
  }

  /** erf via the Numerical Recipes erfc approximation (fractional error below 1.2e-7 everywhere). */
  static double erf(double x) {
    double z = Math.abs(x);
    double t = 1 / (1 + 0.5 * z);
    double erfc = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
        + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
        + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - erfc : erfc - 1;
  }

  /** GLU over the first (channel) axis of {@code (2c, inner)}: {@code a · sigmoid(b)}, giving {@code (c, inner)}. */
  static float[] glu(float[] x, int channels2, int inner) {
    int half = channels2 / 2 * inner;
    float[] out = new float[half];
    if (BACKEND != null && half >= BACKEND.minElements() && BACKEND.glu(x, out)) return out;
    IntStream.range(0, (half + 65535) / 65536).parallel().forEach(blk -> {
      int end = Math.min(half, (blk + 1) * 65536);
      for (int i = blk * 65536; i < end; i++) out[i] = (float) (x[i] / (1 + Math.exp(-x[half + i])));
    });
    return out;
  }

  /** Row-wise softmax of a {@code (rows, cols)} matrix, in place. */
  static void softmaxRows(float[] x, int rows, int cols) {
    IntStream.range(0, rows).parallel().forEach(r -> {
      int o = r * cols;
      float max = Float.NEGATIVE_INFINITY;
      for (int j = 0; j < cols; j++) max = Math.max(max, x[o + j]);
      double sum = 0;
      for (int j = 0; j < cols; j++) {
        float e = (float) Math.exp(x[o + j] - max);
        x[o + j] = e;
        sum += e;
      }
      float inv = (float) (1 / sum);
      for (int j = 0; j < cols; j++) x[o + j] *= inv;
    });
  }
}
