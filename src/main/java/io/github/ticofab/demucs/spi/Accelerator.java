package io.github.ticofab.demucs.spi;

/**
 * An optional hardware accelerator, discovered with {@link java.util.ServiceLoader}: matrix multiplies (~95% of the
 * model's work) and, optionally, whole attention blocks. Without one, everything runs in pure Java on the CPU.
 *
 * <p>Select with the system property {@code demucs.backend}: a backend's {@link #name()}, or {@code java} to force
 * the pure-Java path. By default the first available backend is used.
 */
public interface Accelerator {

  /** Short name, e.g. {@code metal}. */
  String name();

  /** Whether this backend can run here (right OS, a usable device). Called once; must not throw. */
  boolean available();

  /** Problems smaller than this many FLOPs stay on the CPU, where they are cheaper than a round trip. */
  default long minFlops() {
    return 100_000_000L;
  }

  /**
   * Marks an array as constant (a weight): it never changes, so the backend may keep a copy on the device. Arrays
   * not marked are copied on every call.
   */
  default void constant(float[] array) {}

  /**
   * {@code C = A·B}, or {@code C += A·B} when {@code accumulate}; row-major with leading dimensions and offsets (in
   * elements). Only the {@code m×n} block of C may change.
   */
  void gemm(int m, int n, int k,
            float[] a, int aOff, int lda,
            float[] b, int bOff, int ldb,
            float[] c, int cOff, int ldc,
            boolean accumulate);

  /**
   * Multi-head scaled dot-product attention: for each head {@code h}, {@code out_h = softmax(q_h·k_hᵀ)·v_h}, where
   * {@code x_h} is columns {@code [h*headDim, (h+1)*headDim)} of the token-major {@code (tokens, heads*headDim)}
   * matrix. {@code q} is already scaled. Returns false if unsupported, and the caller does it on the CPU.
   */
  default boolean attention(int nq, int nk, int heads, int headDim, float[] q, float[] k, float[] v, float[] out) {
    return false;
  }

  /** Elementwise work smaller than this many elements stays on the CPU. */
  default int minElements() {
    return 1 << 20;
  }

  /** Exact (erf) GELU in place. Returns false if unsupported. */
  default boolean gelu(float[] x) {
    return false;
  }

  /** GLU over the channel axis of {@code (2c, inner)}: {@code out = a·sigmoid(b)}. Returns false if unsupported. */
  default boolean glu(float[] x, float[] out) {
    return false;
  }

  /**
   * Demucs' gated residual: for {@code batch} items of {@code (2·channels, len)} in {@code g}, with halves a and b,
   * {@code x += scale[c]·a·sigmoid(b)} over {@code (batch, channels, len)}. Returns false if unsupported.
   */
  default boolean gatedResidual(float[] x, float[] g, float[] scale, int batch, int channels, int len) {
    return false;
  }
}
