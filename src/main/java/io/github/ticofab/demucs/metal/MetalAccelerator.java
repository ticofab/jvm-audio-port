package io.github.ticofab.demucs.metal;

import io.github.ticofab.demucs.spi.Accelerator;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The Apple GPU through Metal Performance Shaders: matrix multiplies ({@code MPSMatrixMultiplication}) and whole
 * attention blocks (plus {@code MPSMatrixSoftMax}), using shared (unified) memory. Weights marked
 * {@link #constant constant} are uploaded once and stay on the GPU.
 */
public final class MetalAccelerator implements Accelerator {

  private static final long FLOAT32 = 0x10000000L | 32; // MPSDataTypeFloat32
  private static final long STORAGE_SHARED = 0; // MTLResourceStorageModeShared

  // Elementwise kernels, compiled from source when the backend starts. erf is not in the Metal standard library, so
  // it is the same Numerical Recipes approximation the CPU path uses.
  private static final String KERNELS = """
      #include <metal_stdlib>
      using namespace metal;

      static float erf_nr(float x) {
        float z = fabs(x);
        float t = 1.0f / (1.0f + 0.5f * z);
        float erfc = t * precise::exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f
            + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f
            + t * (-0.82215223f + t * 0.17087277f)))))))));
        return x >= 0.0f ? 1.0f - erfc : erfc - 1.0f;
      }

      static float sigmoid(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

      kernel void gelu(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + erf_nr(v * 0.70710678118654752f));
      }

      kernel void glu(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                      constant uint& half_n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= half_n) return;
        out[i] = x[i] * sigmoid(x[half_n + i]);
      }

      kernel void gated_residual(device float* x [[buffer(0)]], device const float* g [[buffer(1)]],
                                 device const float* scale [[buffer(2)]], constant uint* dims [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
        uint channels = dims[0], len = dims[1], n = dims[2];
        if (i >= n) return;
        uint per = channels * len, b = i / per, r = i % per, c = r / len;
        uint go = b * 2 * per;
        x[i] += scale[c] * g[go + r] * sigmoid(g[go + per + r]);
      }
      """;

  private static final java.lang.foreign.MemoryLayout MTL_SIZE =
      java.lang.foreign.MemoryLayout.structLayout(JAVA_LONG, JAVA_LONG, JAVA_LONG);

  private MemorySegment device = MemorySegment.NULL;
  private MemorySegment queue;
  private MemorySegment geluPipeline, gluPipeline, gatedPipeline;
  private final Map<float[], MemorySegment> constants = new IdentityHashMap<>();
  // Temporary buffers are recycled by power-of-two size: allocating (and zeroing) large buffers on every call costs
  // more than the work on small layers.
  private final Map<Long, java.util.ArrayDeque<MemorySegment>> pool = new java.util.HashMap<>();
  private final Map<MemorySegment, Long> pooledSize = new java.util.HashMap<>();

  @Override
  public String name() {
    return "metal";
  }

  // Objective-C autorelease pools belong to one OS thread, and callers may be virtual threads that move between
  // carrier threads; so every Metal call runs on this one platform thread (the GPU serializes the work anyway).
  private final java.util.concurrent.ExecutorService gpuThread = java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
    Thread t = new Thread(r, "demucs-metal");
    t.setDaemon(true);
    return t;
  });

  private <T> T onGpu(java.util.concurrent.Callable<T> work) {
    try {
      return gpuThread.submit(work).get();
    } catch (java.util.concurrent.ExecutionException e) {
      if (e.getCause() instanceof RuntimeException r) throw r;
      if (e.getCause() instanceof Error err) throw err;
      throw new RuntimeException(e.getCause());
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException(e);
    }
  }

  @Override
  public boolean available() {
    try {
      return onGpu(this::availableOnGpu);
    } catch (Throwable t) {
      return false;
    }
  }

  @Override
  public void constant(float[] array) {
    onGpu(() -> {
      constantOnGpu(array);
      return null;
    });
  }

  @Override
  public void gemm(int m, int n, int k, float[] a, int aOff, int lda, float[] b, int bOff, int ldb,
                   float[] c, int cOff, int ldc, boolean accumulate) {
    onGpu(() -> {
      gemmOnGpu(m, n, k, a, aOff, lda, b, bOff, ldb, c, cOff, ldc, accumulate);
      return null;
    });
  }

  @Override
  public boolean attention(int nq, int nk, int heads, int headDim, float[] q, float[] k, float[] v, float[] out) {
    return onGpu(() -> attentionOnGpu(nq, nk, heads, headDim, q, k, v, out));
  }

  @Override
  public boolean gelu(float[] x) {
    return onGpu(() -> geluOnGpu(x));
  }

  @Override
  public boolean glu(float[] x, float[] out) {
    return onGpu(() -> gluOnGpu(x, out));
  }

  @Override
  public boolean gatedResidual(float[] x, float[] g, float[] scale, int batch, int channels, int len) {
    return onGpu(() -> gatedResidualOnGpu(x, g, scale, batch, channels, len));
  }

  private boolean availableOnGpu() {
    if (!System.getProperty("os.name", "").startsWith("Mac")) return false;
    try {
      SymbolLookup metal = SymbolLookup.libraryLookup("/System/Library/Frameworks/Metal.framework/Metal", ObjC.ARENA);
      // Loading the framework registers the MPS classes with the Objective-C runtime.
      SymbolLookup.libraryLookup(
          "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders", ObjC.ARENA);
      MethodHandle create = ObjC.function(metal, "MTLCreateSystemDefaultDevice", FunctionDescriptor.of(ADDRESS));
      device = (MemorySegment) create.invokeExact();
      if (device.equals(MemorySegment.NULL)) return false;
      queue = ObjC.id(device, "newCommandQueue");
      if (queue.equals(MemorySegment.NULL)) return false;
      MemorySegment pool = ObjC.poolPush();
      try {
        MemorySegment error = ObjC.ARENA.allocate(ADDRESS);
        MemorySegment library = (MemorySegment) ObjC.send(ADDRESS, device, "newLibraryWithSource:options:error:",
            List.of(ADDRESS, ADDRESS, ADDRESS), nsString(KERNELS), MemorySegment.NULL, error);
        if (library.equals(MemorySegment.NULL)) {
          System.err.println("demucs-jvm-metal: kernel compilation failed: " + describe(error.get(ADDRESS, 0)));
          return false;
        }
        geluPipeline = pipeline(library, "gelu");
        gluPipeline = pipeline(library, "glu");
        gatedPipeline = pipeline(library, "gated_residual");
      } finally {
        ObjC.poolPop(pool);
      }
      return true;
    } catch (Throwable t) {
      return false;
    }
  }

  private void constantOnGpu(float[] array) {
    if (!constants.containsKey(array) && array.length > 0) {
      MemorySegment buf = newBuffer((long) array.length * 4);
      MemorySegment.copy(array, 0, contents(buf, array.length), JAVA_FLOAT, 0, array.length);
      constants.put(array, buf);
    }
  }

  private void gemmOnGpu(int m, int n, int k,
                                float[] a, int aOff, int lda,
                                float[] b, int bOff, int ldb,
                                float[] c, int cOff, int ldc,
                                boolean accumulate) {
    MemorySegment pool = ObjC.poolPush();
    MemorySegment tmpA = MemorySegment.NULL, tmpB = MemorySegment.NULL, bufC = MemorySegment.NULL;
    MemorySegment matA = MemorySegment.NULL, matB = MemorySegment.NULL, matC = MemorySegment.NULL, op = MemorySegment.NULL;
    try {
      int lenA = (m - 1) * lda + k, lenB = (k - 1) * ldb + n, lenC = (m - 1) * ldc + n;
      MemorySegment bufA = constants.get(a);
      long offA = (long) aOff * 4;
      if (bufA == null) {
        bufA = tmpA = upload(a, aOff, lenA);
        offA = 0;
      }
      MemorySegment bufB = constants.get(b);
      long offB = (long) bOff * 4;
      if (bufB == null) {
        bufB = tmpB = upload(b, bOff, lenB);
        offB = 0;
      }
      // C's region is copied in whenever rows are strided, so columns outside the m×n block come back unchanged.
      boolean copyIn = accumulate || ldc != n;
      bufC = copyIn ? upload(c, cOff, lenC) : temp((long) lenC * 4);

      matA = matrix(bufA, offA, m, k, lda);
      matB = matrix(bufB, offB, k, n, ldb);
      matC = matrix(bufC, 0, m, n, ldc);
      op = (MemorySegment) ObjC.send(ADDRESS, ObjC.id(ObjC.cls("MPSMatrixMultiplication"), "alloc"),
          "initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:",
          List.of(ADDRESS, JAVA_BOOLEAN, JAVA_BOOLEAN, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_DOUBLE, JAVA_DOUBLE),
          device, false, false, (long) m, (long) n, (long) k, 1.0, accumulate ? 1.0 : 0.0);
      MemorySegment commands = ObjC.id(queue, "commandBuffer");
      ObjC.send(null, op, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:",
          List.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS), commands, matA, matB, matC);
      run(commands);

      MemorySegment out = contents(bufC, lenC);
      if (ldc == n) {
        copyOut(out, 0, c, cOff, lenC);
      } else {
        for (int i = 0; i < m; i++) MemorySegment.copy(out, JAVA_FLOAT, (long) i * ldc * 4, c, cOff + i * ldc, n);
      }
    } finally {
      for (MemorySegment o : new MemorySegment[] {op, matA, matB, matC}) ObjC.release(o);
      for (MemorySegment o : new MemorySegment[] {tmpA, tmpB, bufC}) if (!o.equals(MemorySegment.NULL)) recycle(o);
      ObjC.poolPop(pool);
    }
  }

  private boolean attentionOnGpu(int nq, int nk, int heads, int headDim, float[] q, float[] k, float[] v,
                                        float[] out) {
    int dim = heads * headDim;
    MemorySegment pool = ObjC.poolPush();
    java.util.List<MemorySegment> owned = new java.util.ArrayList<>(), temps = new java.util.ArrayList<>();
    try {
      MemorySegment bq = keep(temps, upload(q, 0, nq * dim));
      MemorySegment bk = keep(temps, upload(k, 0, nk * dim));
      MemorySegment bv = keep(temps, upload(v, 0, nk * dim));
      MemorySegment bo = keep(temps, temp((long) nq * dim * 4));
      MemorySegment scores = keep(temps, temp((long) nq * nk * 4));
      MemorySegment probs = keep(temps, temp((long) nq * nk * 4));
      MemorySegment qk = keep(owned, multiplication(false, true, nq, nk, headDim));
      MemorySegment pv = keep(owned, multiplication(false, false, nq, headDim, nk));
      MemorySegment softmax = keep(owned, (MemorySegment) ObjC.send(ADDRESS,
          ObjC.id(ObjC.cls("MPSMatrixSoftMax"), "alloc"), "initWithDevice:", List.of(ADDRESS), device));
      MemorySegment matS = keep(owned, matrix(scores, 0, nq, nk, nk));
      MemorySegment matP = keep(owned, matrix(probs, 0, nq, nk, nk));
      MemorySegment commands = ObjC.id(queue, "commandBuffer");
      List<java.lang.foreign.MemoryLayout> four = List.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS);
      for (int h = 0; h < heads; h++) {
        long off = (long) h * headDim * 4;
        MemorySegment matQ = keep(owned, matrix(bq, off, nq, headDim, dim));
        MemorySegment matK = keep(owned, matrix(bk, off, nk, headDim, dim));
        MemorySegment matV = keep(owned, matrix(bv, off, nk, headDim, dim));
        MemorySegment matO = keep(owned, matrix(bo, off, nq, headDim, dim));
        ObjC.send(null, qk, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", four, commands, matQ, matK, matS);
        ObjC.send(null, softmax, "encodeToCommandBuffer:inputMatrix:resultMatrix:", List.of(ADDRESS, ADDRESS, ADDRESS),
            commands, matS, matP);
        ObjC.send(null, pv, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", four, commands, matP, matV, matO);
      }
      run(commands);
      copyOut(contents(bo, (long) nq * dim), 0, out, 0, nq * dim);
      return true;
    } finally {
      owned.forEach(ObjC::release);
      temps.forEach(this::recycle);
      ObjC.poolPop(pool);
    }
  }

  private boolean geluOnGpu(float[] x) {
    MemorySegment pool = ObjC.poolPush();
    MemorySegment buf = upload(x, 0, x.length);
    try {
      dispatch(geluPipeline, x.length, new MemorySegment[] {buf}, new int[] {x.length});
      copyOut(contents(buf, x.length), 0, x, 0, x.length);
      return true;
    } finally {
      recycle(buf);
      ObjC.poolPop(pool);
    }
  }

  private boolean gluOnGpu(float[] x, float[] out) {
    MemorySegment pool = ObjC.poolPush();
    MemorySegment in = upload(x, 0, 2 * out.length), res = temp((long) out.length * 4);
    try {
      dispatch(gluPipeline, out.length, new MemorySegment[] {in, res}, new int[] {out.length});
      copyOut(contents(res, out.length), 0, out, 0, out.length);
      return true;
    } finally {
      recycle(in);
      recycle(res);
      ObjC.poolPop(pool);
    }
  }

  private boolean gatedResidualOnGpu(float[] x, float[] g, float[] scale, int batch, int channels, int len) {
    int n = batch * channels * len;
    MemorySegment pool = ObjC.poolPush();
    MemorySegment bx = upload(x, 0, n), bg = upload(g, 0, 2 * n), bs = upload(scale, 0, channels);
    try {
      dispatch(gatedPipeline, n, new MemorySegment[] {bx, bg, bs}, new int[] {channels, len, n});
      copyOut(contents(bx, n), 0, x, 0, n);
      return true;
    } finally {
      for (MemorySegment b : new MemorySegment[] {bx, bg, bs}) recycle(b);
      ObjC.poolPop(pool);
    }
  }

  // One compute kernel over n threads: buffers at indices 0.., then the ints as one small constant buffer.
  private void dispatch(MemorySegment pipeline, int n, MemorySegment[] buffers, int[] ints) {
    MemorySegment commands = ObjC.id(queue, "commandBuffer");
    MemorySegment encoder = ObjC.id(commands, "computeCommandEncoder");
    ObjC.send(null, encoder, "setComputePipelineState:", List.of(ADDRESS), pipeline);
    for (int i = 0; i < buffers.length; i++) {
      ObjC.send(null, encoder, "setBuffer:offset:atIndex:", List.of(ADDRESS, JAVA_LONG, JAVA_LONG), buffers[i], 0L, (long) i);
    }
    try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
      MemorySegment args = arena.allocateFrom(JAVA_INT, ints);
      ObjC.send(null, encoder, "setBytes:length:atIndex:", List.of(ADDRESS, JAVA_LONG, JAVA_LONG), args,
          (long) ints.length * 4, (long) buffers.length);
      long width = Math.min(256, (long) ObjC.send(JAVA_LONG, pipeline, "maxTotalThreadsPerThreadgroup", List.of()));
      MemorySegment grid = arena.allocate(MTL_SIZE), group = arena.allocate(MTL_SIZE);
      grid.setAtIndex(JAVA_LONG, 0, n);
      grid.setAtIndex(JAVA_LONG, 1, 1);
      grid.setAtIndex(JAVA_LONG, 2, 1);
      group.setAtIndex(JAVA_LONG, 0, width);
      group.setAtIndex(JAVA_LONG, 1, 1);
      group.setAtIndex(JAVA_LONG, 2, 1);
      ObjC.send(null, encoder, "dispatchThreads:threadsPerThreadgroup:", List.of(MTL_SIZE, MTL_SIZE), grid, group);
    }
    ObjC.call(encoder, "endEncoding");
    run(commands);
  }

  private MemorySegment pipeline(MemorySegment library, String function) {
    MemorySegment fn = (MemorySegment) ObjC.send(ADDRESS, library, "newFunctionWithName:", List.of(ADDRESS),
        nsString(function));
    MemorySegment error = ObjC.ARENA.allocate(ADDRESS);
    MemorySegment p = (MemorySegment) ObjC.send(ADDRESS, device, "newComputePipelineStateWithFunction:error:",
        List.of(ADDRESS, ADDRESS), fn, error);
    if (p.equals(MemorySegment.NULL)) throw new IllegalStateException("pipeline " + function + ": " + describe(error.get(ADDRESS, 0)));
    return p;
  }

  private static MemorySegment nsString(String s) {
    return (MemorySegment) ObjC.send(ADDRESS, ObjC.cls("NSString"), "stringWithUTF8String:", List.of(ADDRESS),
        ObjC.ARENA.allocateFrom(s));
  }

  private static String describe(MemorySegment nsError) {
    if (nsError.equals(MemorySegment.NULL)) return "unknown error";
    MemorySegment text = ObjC.id(ObjC.id(nsError, "localizedDescription"), "UTF8String");
    return text.reinterpret(Long.MAX_VALUE).getString(0);
  }

  private static MemorySegment keep(java.util.List<MemorySegment> owned, MemorySegment object) {
    owned.add(object);
    return object;
  }

  private MemorySegment multiplication(boolean transposeLeft, boolean transposeRight, int rows, int cols, int inner) {
    return (MemorySegment) ObjC.send(ADDRESS, ObjC.id(ObjC.cls("MPSMatrixMultiplication"), "alloc"),
        "initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:",
        List.of(ADDRESS, JAVA_BOOLEAN, JAVA_BOOLEAN, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_DOUBLE, JAVA_DOUBLE),
        device, transposeLeft, transposeRight, (long) rows, (long) cols, (long) inner, 1.0, 0.0);
  }

  private static void run(MemorySegment commands) {
    ObjC.call(commands, "commit");
    ObjC.call(commands, "waitUntilCompleted");
    long status = (long) ObjC.send(JAVA_LONG, commands, "status", List.of());
    if (status != 4) throw new IllegalStateException("Metal command buffer failed, status " + status); // 4 = completed
  }

  /** An MPS view of a row-major matrix inside a buffer: no copy, so a head's columns can be addressed in place. */
  private MemorySegment matrix(MemorySegment buffer, long offsetBytes, int rows, int cols, int ld) {
    MemorySegment desc = (MemorySegment) ObjC.send(ADDRESS, ObjC.cls("MPSMatrixDescriptor"),
        "matrixDescriptorWithRows:columns:rowBytes:dataType:", List.of(JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT),
        (long) rows, (long) cols, (long) ld * 4, (int) FLOAT32);
    return (MemorySegment) ObjC.send(ADDRESS, ObjC.id(ObjC.cls("MPSMatrix"), "alloc"),
        "initWithBuffer:offset:descriptor:", List.of(ADDRESS, JAVA_LONG, ADDRESS), buffer, offsetBytes, desc);
  }

  /** Copies {@code src[off, off+len)} into a (pooled) shared buffer that the GPU reads directly. */
  private MemorySegment upload(float[] src, int off, int len) {
    MemorySegment buf = temp((long) len * 4);
    copyIn(src, off, contents(buf, len), len);
    return buf;
  }

  private MemorySegment temp(long bytes) {
    long size = Long.highestOneBit(Math.max(bytes, 1 << 16) - 1) << 1;
    java.util.ArrayDeque<MemorySegment> free = pool.get(size);
    MemorySegment buf = free == null || free.isEmpty() ? newBuffer(size) : free.pop();
    pooledSize.put(buf, size);
    return buf;
  }

  private void recycle(MemorySegment buf) {
    Long size = pooledSize.remove(buf);
    if (size == null) ObjC.release(buf);
    else pool.computeIfAbsent(size, s -> new java.util.ArrayDeque<>()).push(buf);
  }

  // Large copies are split across cores; a single memcpy thread is far below the memory bandwidth.
  private static final int PARALLEL_COPY = 1 << 20;

  private static void copyIn(float[] src, int off, MemorySegment dst, int len) {
    if (len < PARALLEL_COPY) {
      MemorySegment.copy(src, off, dst, JAVA_FLOAT, 0, len);
      return;
    }
    int chunk = PARALLEL_COPY / 4;
    java.util.stream.IntStream.range(0, (len + chunk - 1) / chunk).parallel().forEach(i -> {
      int from = i * chunk, n = Math.min(chunk, len - from);
      MemorySegment.copy(src, off + from, dst, JAVA_FLOAT, (long) from * 4, n);
    });
  }

  private static void copyOut(MemorySegment src, long srcFloat, float[] dst, int off, int len) {
    if (len < PARALLEL_COPY) {
      MemorySegment.copy(src, JAVA_FLOAT, srcFloat * 4, dst, off, len);
      return;
    }
    int chunk = PARALLEL_COPY / 4;
    java.util.stream.IntStream.range(0, (len + chunk - 1) / chunk).parallel().forEach(i -> {
      int from = i * chunk, n = Math.min(chunk, len - from);
      MemorySegment.copy(src, JAVA_FLOAT, (srcFloat + from) * 4, dst, off + from, n);
    });
  }

  private MemorySegment newBuffer(long bytes) {
    MemorySegment buf = (MemorySegment) ObjC.send(ADDRESS, device, "newBufferWithLength:options:",
        List.of(JAVA_LONG, JAVA_LONG), Math.max(bytes, 4), STORAGE_SHARED);
    if (buf.equals(MemorySegment.NULL)) throw new OutOfMemoryError("Metal could not allocate " + bytes + " bytes");
    return buf;
  }

  private static MemorySegment contents(MemorySegment buffer, long floats) {
    return ObjC.id(buffer, "contents").reinterpret(floats * 4);
  }
}
