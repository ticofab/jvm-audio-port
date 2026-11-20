package io.github.ticofab.demucs.spi;

/**
 * Optional, from {@link Accelerator#device()}: buffers that stay on the accelerator, and operations on them that are
 * queued rather than run one by one. A model can keep a whole stack of layers on the device this way and pay for one
 * upload and one download, instead of a round trip per operation.
 *
 * <p>Operations take effect in order; {@link #download} waits for everything queued before it. Offsets and sizes
 * are in floats; matrices are row-major. Buffers are single-use scratch unless obtained from {@link #constant}.
 */
public interface Device {

  /** A buffer on the device. */
  interface Buffer {}

  /** A buffer holding {@code data}. */
  Buffer upload(float[] data);

  /** The device copy of a weight array marked with {@link Accelerator#constant}; never released by the caller. */
  Buffer constant(float[] weights);

  /** A zeroed buffer of {@code floats} floats. */
  Buffer allocate(long floats);

  /** Waits for all queued work, then copies the first {@code count} floats of {@code buffer} into {@code dst}. */
  void download(Buffer buffer, float[] dst, int count);

  /** Returns a scratch buffer once all queued work using it is done. */
  void release(Buffer buffer);

  /** {@code C = A·B}, or {@code C += A·B} when {@code accumulate}. */
  void gemm(int m, int n, int k, Buffer a, int lda, Buffer b, int ldb, Buffer c, int ldc, boolean accumulate);

  /**
   * Attention over {@code batch} sequences of {@code length} tokens, inputs and output {@code (batch, length,
   * heads*headDim)}, scaled by {@code 1/sqrt(headDim)}.
   */
  void attention(int batch, int length, int heads, int headDim, Buffer q, Buffer k, Buffer v, Buffer out);

  /** Row-wise {@code out = x / max(||x||, 1e-12) * sqrt(dim) * gamma} over {@code (rows, dim)}. */
  void rmsNorm(Buffer x, Buffer out, int rows, int dim, Buffer gamma);

  /**
   * Splits token-major {@code qkv (tokens, 3*heads*headDim)} into {@code q, k, v} laid out {@code (sequence,
   * position, heads*headDim)}, where token = {@code s * seqStride + i * posStride}, applying interleaved rotary
   * embeddings to q and k from tables {@code cos, sin (length, headDim/2)}.
   */
  void ropeSplit(Buffer qkv, Buffer q, Buffer k, Buffer v, int sequences, int length, int seqStride, int posStride,
                 int heads, int headDim, Buffer cos, Buffer sin);

  /**
   * The inverse layout move of {@link #ropeSplit} for the attention output, multiplying each head by
   * {@code sigmoid(gates[token, h] + gateBias[h])}: {@code out (tokens, heads*headDim)}.
   */
  void gateMerge(Buffer ctx, Buffer gates, Buffer gateBias, Buffer out, int sequences, int length, int seqStride,
                 int posStride, int heads, int headDim);

  /** {@code x[r, c] += bias[c]} over {@code (rows, cols)}, then exact GELU if {@code gelu}. */
  void addBias(Buffer x, Buffer bias, int rows, int cols, boolean gelu);
}
