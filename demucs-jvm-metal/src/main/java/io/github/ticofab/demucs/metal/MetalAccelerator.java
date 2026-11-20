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
 * attention blocks (with a softmax kernel of our own: MPS's was most of attention's time), using shared (unified)
 * memory. Weights marked
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

      // Sum (or max) over a threadgroup: within each SIMD group, then across them through threadgroup memory.
      static float group_reduce(float v, bool is_max, threadgroup float* shared, uint lane, uint simd, uint groups) {
        v = is_max ? simd_max(v) : simd_sum(v);
        if (lane == 0) shared[simd] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
          float w = lane < groups ? shared[lane] : (is_max ? -INFINITY : 0.0f);
          w = is_max ? simd_max(w) : simd_sum(w);
          if (lane == 0) shared[0] = w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float r = shared[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return r;
      }

      // One threadgroup per row; dims: rows, dim.
      kernel void rms_norm(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                           device const float* gamma [[buffer(2)]], constant uint* dims [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                           uint tpg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
        threadgroup float shared[32];
        uint dim = dims[1];
        device const float* xr = x + (ulong) row * dim;
        float sq = 0.0f;
        for (uint d = tid; d < dim; d += tpg) sq += xr[d] * xr[d];
        sq = group_reduce(sq, false, shared, lane, simd, (tpg + 31) / 32);
        float f = sqrt((float) dim) / max(sqrt(sq), 1e-12f);
        device float* o = out + (ulong) row * dim;
        for (uint d = tid; d < dim; d += tpg) o[d] = xr[d] * f * gamma[d];
      }

      // In-place softmax of each row, one threadgroup per row; dims: cols.
      kernel void softmax_rows(device float* x [[buffer(0)]], constant uint* dims [[buffer(1)]],
                               uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                               uint tpg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
        threadgroup float shared[32];
        uint cols = dims[0], groups = (tpg + 31) / 32;
        device float* r = x + (ulong) row * cols;
        float m = -INFINITY;
        for (uint j = tid; j < cols; j += tpg) m = max(m, r[j]);
        m = group_reduce(m, true, shared, lane, simd, groups);
        float sum = 0.0f;
        for (uint j = tid; j < cols; j += tpg) {
          float e = precise::exp(r[j] - m);
          r[j] = e;
          sum += e;
        }
        float inv = 1.0f / group_reduce(sum, false, shared, lane, simd, groups);
        for (uint j = tid; j < cols; j += tpg) r[j] *= inv;
      }

      // dims: sequences, length, seqStride, posStride, heads, headDim
      kernel void rope_split(device const float* qkv [[buffer(0)]], device float* q [[buffer(1)]],
                             device float* k [[buffer(2)]], device float* v [[buffer(3)]],
                             device const float* cosT [[buffer(4)]], device const float* sinT [[buffer(5)]],
                             constant uint* dims [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
        uint length = dims[1], seqStride = dims[2], posStride = dims[3], headDim = dims[5];
        uint inner = dims[4] * headDim;
        if (gid >= dims[0] * length * inner) return;
        uint n = gid / inner, c = gid % inner, s = n / length, i = n % length;
        device const float* src = qkv + (ulong) (s * seqStride + i * posStride) * 3 * inner;
        uint d = c % headDim, pairs = headDim / 2, t = i * pairs + d / 2;
        float cs = cosT[t], sn = sinT[t];
        bool even = (d & 1) == 0;
        uint a = even ? c : c - 1;
        float qa = src[a], qb = src[a + 1], ka = src[inner + a], kb = src[inner + a + 1];
        q[gid] = even ? qa * cs - qb * sn : qb * cs + qa * sn;
        k[gid] = even ? ka * cs - kb * sn : kb * cs + ka * sn;
        v[gid] = src[2 * inner + c];
      }

      // dims: sequences, length, seqStride, posStride, heads, headDim
      kernel void gate_merge(device const float* ctx [[buffer(0)]], device const float* gates [[buffer(1)]],
                             device const float* gateBias [[buffer(2)]], device float* out [[buffer(3)]],
                             constant uint* dims [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
        uint length = dims[1], seqStride = dims[2], posStride = dims[3], heads = dims[4], headDim = dims[5];
        uint inner = heads * headDim;
        if (gid >= dims[0] * length * inner) return;
        uint n = gid / inner, c = gid % inner, s = n / length, i = n % length;
        uint tok = s * seqStride + i * posStride, h = c / headDim;
        out[(ulong) tok * inner + c] = ctx[gid] * sigmoid(gates[tok * heads + h] + gateBias[h]);
      }

      // dims: rows, cols, gelu
      kernel void add_bias(device float* x [[buffer(0)]], device const float* bias [[buffer(1)]],
                           constant uint* dims [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
        if (gid >= dims[0] * dims[1]) return;
        float v = x[gid] + bias[gid % dims[1]];
        x[gid] = dims[2] != 0 ? 0.5f * v * (1.0f + erf_nr(v * 0.70710678118654752f)) : v;
      }
      """;

  private static final java.lang.foreign.MemoryLayout MTL_SIZE =
      java.lang.foreign.MemoryLayout.structLayout(JAVA_LONG, JAVA_LONG, JAVA_LONG);

  private MemorySegment device = MemorySegment.NULL;
  private MemorySegment queue;
  private MemorySegment geluPipeline, gluPipeline, gatedPipeline, rmsPipeline, ropePipeline, gatePipeline, biasPipeline,
      softmaxPipeline;
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
  public boolean batchedAttention(int batch, int length, int heads, int headDim, float[] q, float[] k, float[] v,
                                  float[] out) {
    return onGpu(() -> batchedAttentionOnGpu(batch, length, heads, headDim, q, k, v, out));
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
        rmsPipeline = pipeline(library, "rms_norm");
        ropePipeline = pipeline(library, "rope_split");
        gatePipeline = pipeline(library, "gate_merge");
        biasPipeline = pipeline(library, "add_bias");
        softmaxPipeline = pipeline(library, "softmax_rows");
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
      // MPS treats a matrix as rows × rowBytes, so every buffer covers whole rows, including the last one's padding
      // up to the leading dimension: buffers sized only to the last element let MPS run past their end.
      MemorySegment bufA = (long) aOff + (long) m * lda <= a.length ? constants.get(a) : null;
      long offA = (long) aOff * 4;
      if (bufA == null) {
        bufA = tmpA = upload(a, aOff, lenA, (long) m * lda);
        offA = 0;
      }
      MemorySegment bufB = (long) bOff + (long) k * ldb <= b.length ? constants.get(b) : null;
      long offB = (long) bOff * 4;
      if (bufB == null) {
        bufB = tmpB = upload(b, bOff, lenB, (long) k * ldb);
        offB = 0;
      }
      // C's region is copied in whenever rows are strided, so columns outside the m×n block come back unchanged.
      boolean copyIn = accumulate || ldc != n;
      bufC = copyIn ? upload(c, cOff, lenC, (long) m * ldc) : output((long) m * ldc * 4);

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
      // Head views start up to dim columns into the buffer, so each buffer gets one extra row.
      MemorySegment bq = keep(temps, upload(q, 0, nq * dim, (long) (nq + 1) * dim));
      MemorySegment bk = keep(temps, upload(k, 0, nk * dim, (long) (nk + 1) * dim));
      MemorySegment bv = keep(temps, upload(v, 0, nk * dim, (long) (nk + 1) * dim));
      MemorySegment bo = keep(temps, output((long) (nq + 1) * dim * 4));
      MemorySegment scores = keep(temps, output((long) nq * nk * 4));
      MemorySegment qk = keep(owned, multiplication(false, true, nq, nk, headDim));
      MemorySegment pv = keep(owned, multiplication(false, false, nq, headDim, nk));
      MemorySegment matS = keep(owned, matrix(scores, 0, nq, nk, nk));
      MemorySegment commands = ObjC.id(queue, "commandBuffer");
      List<java.lang.foreign.MemoryLayout> four = List.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS);
      for (int h = 0; h < heads; h++) {
        long off = (long) h * headDim * 4;
        MemorySegment matQ = keep(owned, matrix(bq, off, nq, headDim, dim));
        MemorySegment matK = keep(owned, matrix(bk, off, nk, headDim, dim));
        MemorySegment matV = keep(owned, matrix(bv, off, nk, headDim, dim));
        MemorySegment matO = keep(owned, matrix(bo, off, nq, headDim, dim));
        ObjC.send(null, qk, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", four, commands, matQ, matK, matS);
        encodeSoftmax(commands, scores, nq, nk);
        ObjC.send(null, pv, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", four, commands, matS, matV, matO);
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
    MemorySegment in = upload(x, 0, 2 * out.length), res = output((long) out.length * 4);
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
    encodeKernel(commands, pipeline, n, buffers, ints);
    run(commands);
  }

  private void encodeKernel(MemorySegment commands, MemorySegment pipeline, int n, MemorySegment[] buffers, int[] ints) {
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
  }

  // One threadgroup of up to 256 threads per row, for the row-reduction kernels.
  private void encodeRows(MemorySegment commands, MemorySegment pipeline, int rows, int threads, MemorySegment[] buffers,
                          int[] ints) {
    MemorySegment encoder = ObjC.id(commands, "computeCommandEncoder");
    ObjC.send(null, encoder, "setComputePipelineState:", List.of(ADDRESS), pipeline);
    for (int i = 0; i < buffers.length; i++) {
      ObjC.send(null, encoder, "setBuffer:offset:atIndex:", List.of(ADDRESS, JAVA_LONG, JAVA_LONG), buffers[i], 0L, (long) i);
    }
    try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
      MemorySegment args = arena.allocateFrom(JAVA_INT, ints);
      ObjC.send(null, encoder, "setBytes:length:atIndex:", List.of(ADDRESS, JAVA_LONG, JAVA_LONG), args,
          (long) ints.length * 4, (long) buffers.length);
      long width = Math.min(Math.min(256, threads),
          (long) ObjC.send(JAVA_LONG, pipeline, "maxTotalThreadsPerThreadgroup", List.of()));
      width = Math.max(32, (width + 31) / 32 * 32);
      MemorySegment grid = arena.allocate(MTL_SIZE), group = arena.allocate(MTL_SIZE);
      grid.setAtIndex(JAVA_LONG, 0, rows);
      grid.setAtIndex(JAVA_LONG, 1, 1);
      grid.setAtIndex(JAVA_LONG, 2, 1);
      group.setAtIndex(JAVA_LONG, 0, width);
      group.setAtIndex(JAVA_LONG, 1, 1);
      group.setAtIndex(JAVA_LONG, 2, 1);
      ObjC.send(null, encoder, "dispatchThreadgroups:threadsPerThreadgroup:", List.of(MTL_SIZE, MTL_SIZE), grid, group);
    }
    ObjC.call(encoder, "endEncoding");
  }

  private void encodeSoftmax(MemorySegment commands, MemorySegment scores, int rows, int cols) {
    encodeRows(commands, softmaxPipeline, rows, cols, new MemorySegment[] {scores}, new int[] {cols});
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

  // Each head is one batched MPS product over all sequences: the descriptor's matrixBytes steps from one sequence
  // to the next, and the byte offset picks the head's columns.
  private boolean batchedAttentionOnGpu(int batch, int length, int heads, int headDim, float[] q, float[] k,
                                        float[] v, float[] out) {
    int inner = heads * headDim, n = batch * length * inner;
    long scoreBytes = (long) batch * length * length * 4;
    MemorySegment pool = ObjC.poolPush();
    java.util.List<MemorySegment> owned = new java.util.ArrayList<>(), temps = new java.util.ArrayList<>();
    try {
      // Head views start up to inner columns into the buffer, so each buffer gets one extra row.
      long padded = (long) n + inner;
      MemorySegment bq = keep(temps, upload(q, 0, n, padded)), bk = keep(temps, upload(k, 0, n, padded));
      MemorySegment bv = keep(temps, upload(v, 0, n, padded));
      MemorySegment bo = keep(temps, output(padded * 4));
      MemorySegment scores = keep(temps, output(scoreBytes));
      MemorySegment qk = keep(owned, multiplication(false, true, length, length, headDim, 1 / Math.sqrt(headDim)));
      MemorySegment pv = keep(owned, multiplication(false, false, length, headDim, length, 1.0));
      long seqBytes = (long) length * inner * 4;
      MemorySegment matS = keep(owned, batchMatrix(scores, 0, length, length, length * 4L, (long) length * length * 4, batch));
      MemorySegment commands = ObjC.id(queue, "commandBuffer");
      List<java.lang.foreign.MemoryLayout> four = List.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS);
      for (int h = 0; h < heads; h++) {
        long off = (long) h * headDim * 4;
        MemorySegment matQ = keep(owned, batchMatrix(bq, off, length, headDim, inner * 4L, seqBytes, batch));
        MemorySegment matK = keep(owned, batchMatrix(bk, off, length, headDim, inner * 4L, seqBytes, batch));
        MemorySegment matV = keep(owned, batchMatrix(bv, off, length, headDim, inner * 4L, seqBytes, batch));
        MemorySegment matO = keep(owned, batchMatrix(bo, off, length, headDim, inner * 4L, seqBytes, batch));
        ObjC.send(null, qk, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", four, commands, matQ, matK, matS);
        encodeSoftmax(commands, scores, batch * length, length);
        ObjC.send(null, pv, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", four, commands, matS, matV, matO);
      }
      run(commands);
      copyOut(contents(bo, n), 0, out, 0, n);
      return true;
    } finally {
      owned.forEach(ObjC::release);
      temps.forEach(this::recycle);
      ObjC.poolPop(pool);
    }
  }

  private MemorySegment batchMatrix(MemorySegment buffer, long offsetBytes, int rows, int cols, long rowBytes,
                                    long matrixBytes, int matrices) {
    MemorySegment desc = (MemorySegment) ObjC.send(ADDRESS, ObjC.cls("MPSMatrixDescriptor"),
        "matrixDescriptorWithRows:columns:matrices:rowBytes:matrixBytes:dataType:",
        List.of(JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT),
        (long) rows, (long) cols, (long) matrices, rowBytes, matrixBytes, (int) FLOAT32);
    return (MemorySegment) ObjC.send(ADDRESS, ObjC.id(ObjC.cls("MPSMatrix"), "alloc"),
        "initWithBuffer:offset:descriptor:", List.of(ADDRESS, JAVA_LONG, ADDRESS), buffer, offsetBytes, desc);
  }

  private static MemorySegment keep(java.util.List<MemorySegment> owned, MemorySegment object) {
    owned.add(object);
    return object;
  }

  private MemorySegment multiplication(boolean transposeLeft, boolean transposeRight, int rows, int cols, int inner) {
    return multiplication(transposeLeft, transposeRight, rows, cols, inner, 1.0);
  }

  private MemorySegment multiplication(boolean transposeLeft, boolean transposeRight, int rows, int cols, int inner,
                                       double alpha) {
    return (MemorySegment) ObjC.send(ADDRESS, ObjC.id(ObjC.cls("MPSMatrixMultiplication"), "alloc"),
        "initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:",
        List.of(ADDRESS, JAVA_BOOLEAN, JAVA_BOOLEAN, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_DOUBLE, JAVA_DOUBLE),
        device, transposeLeft, transposeRight, (long) rows, (long) cols, (long) inner, alpha, 0.0);
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
    return upload(src, off, len, len);
  }

  /** Like {@link #upload(float[], int, int)}, in a buffer of at least {@code capacity} floats. */
  private MemorySegment upload(float[] src, int off, int len, long capacity) {
    MemorySegment buf = temp(Math.max(len, capacity) * 4);
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

  /**
   * A pooled buffer that a kernel only writes, zeroed first: a recycled buffer holds the previous user's data, and
   * MPS still reads an output it is told to overwrite (β = 0 computes 0·C, and 0·NaN is NaN).
   */
  private MemorySegment output(long bytes) {
    MemorySegment buf = temp(bytes);
    ObjC.id(buf, "contents").reinterpret(bytes).fill((byte) 0);
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

  // MPS kernels work in tiles and can read past the end of a matrix (masking those values by multiplying with zero);
  // whatever memory follows a buffer then leaks into results, and 0 × NaN is NaN. So every buffer gets a zeroed
  // margin at the end.
  private static final long MARGIN = 64 * 1024;

  private MemorySegment newBuffer(long bytes) {
    MemorySegment buf = (MemorySegment) ObjC.send(ADDRESS, device, "newBufferWithLength:options:",
        List.of(JAVA_LONG, JAVA_LONG), Math.max(bytes, 4) + MARGIN, STORAGE_SHARED);
    if (buf.equals(MemorySegment.NULL)) throw new OutOfMemoryError("Metal could not allocate " + bytes + " bytes");
    return buf;
  }

  private static MemorySegment contents(MemorySegment buffer, long floats) {
    return ObjC.id(buffer, "contents").reinterpret(floats * 4);
  }

  private final MetalDevice deviceApi = new MetalDevice();

  @Override
  public io.github.ticofab.demucs.spi.Device device() {
    return deviceApi;
  }

  /**
   * The {@link io.github.ticofab.demucs.spi.Device} API: every operation is encoded into one open command buffer,
   * which is committed only when a result is downloaded. Objects the queued work refers to (MPS matrices and
   * kernels, scratch buffers) are released or recycled only after that.
   */
  private final class MetalDevice implements io.github.ticofab.demucs.spi.Device {

    private record Buf(MemorySegment buffer, long floats) implements Buffer {}

    private MemorySegment commands = MemorySegment.NULL;
    private MemorySegment autoreleasePool = MemorySegment.NULL;
    private final java.util.List<MemorySegment> releaseAfter = new java.util.ArrayList<>();
    private final java.util.List<MemorySegment> recycleAfter = new java.util.ArrayList<>();
    private final Map<float[], Buf> constantBufs = new IdentityHashMap<>();

    private MemorySegment open() {
      if (commands.equals(MemorySegment.NULL)) {
        autoreleasePool = ObjC.poolPush();
        commands = ObjC.id(queue, "commandBuffer");
      }
      return commands;
    }

    // Runs on the GPU thread: commits what is queued, waits, then frees what the work referred to.
    private void flush() {
      if (commands.equals(MemorySegment.NULL)) return;
      try {
        run(commands);
      } finally {
        releaseAfter.forEach(ObjC::release);
        releaseAfter.clear();
        recycleAfter.forEach(MetalAccelerator.this::recycle);
        recycleAfter.clear();
        commands = MemorySegment.NULL;
        ObjC.poolPop(autoreleasePool);
      }
    }

    private MemorySegment seg(Buffer b) {
      return ((Buf) b).buffer();
    }

    private MemorySegment later(MemorySegment object) {
      releaseAfter.add(object);
      return object;
    }

    @Override
    public Buffer upload(float[] data) {
      return onGpu(() -> new Buf(MetalAccelerator.this.upload(data, 0, data.length), data.length));
    }

    @Override
    public Buffer constant(float[] weights) {
      return onGpu(() -> constantBufs.computeIfAbsent(weights, w -> {
        MemorySegment b = constants.get(w);
        if (b == null) throw new IllegalArgumentException("not a constant array");
        return new Buf(b, w.length);
      }));
    }

    @Override
    public Buffer allocate(long floats) {
      return onGpu(() -> new Buf(output(floats * 4), floats));
    }

    @Override
    public void download(Buffer buffer, float[] dst, int count) {
      onGpu(() -> {
        flush();
        copyOut(contents(seg(buffer), count), 0, dst, 0, count);
        return null;
      });
    }

    @Override
    public void release(Buffer buffer) {
      onGpu(() -> {
        if (constantBufs.containsValue(buffer)) return null;
        if (commands.equals(MemorySegment.NULL)) recycle(seg(buffer));
        else recycleAfter.add(seg(buffer));
        return null;
      });
    }

    @Override
    public void gemm(int m, int n, int k, Buffer a, int lda, Buffer b, int ldb, Buffer c, int ldc, boolean accumulate) {
      onGpu(() -> {
        MemorySegment cb = open();
        MemorySegment ma = later(matrix(seg(a), 0, m, k, lda)), mb = later(matrix(seg(b), 0, k, n, ldb));
        MemorySegment mc = later(matrix(seg(c), 0, m, n, ldc));
        MemorySegment op = later((MemorySegment) ObjC.send(ADDRESS, ObjC.id(ObjC.cls("MPSMatrixMultiplication"), "alloc"),
            "initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:",
            List.of(ADDRESS, JAVA_BOOLEAN, JAVA_BOOLEAN, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_DOUBLE, JAVA_DOUBLE),
            device, false, false, (long) m, (long) n, (long) k, 1.0, accumulate ? 1.0 : 0.0));
        ObjC.send(null, op, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:",
            List.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS), cb, ma, mb, mc);
        return null;
      });
    }

    @Override
    public void attention(int batch, int length, int heads, int headDim, Buffer q, Buffer k, Buffer v, Buffer out) {
      onGpu(() -> {
        MemorySegment cb = open();
        int inner = heads * headDim;
        long scoreBytes = (long) batch * length * length * 4;
        MemorySegment scores = output(scoreBytes);
        recycleAfter.add(scores);
        MemorySegment qk = later(multiplication(false, true, length, length, headDim, 1 / Math.sqrt(headDim)));
        MemorySegment pv = later(multiplication(false, false, length, headDim, length, 1.0));
        long seqBytes = (long) length * inner * 4, square = (long) length * length * 4;
        MemorySegment matS = later(batchMatrix(scores, 0, length, length, length * 4L, square, batch));
        List<java.lang.foreign.MemoryLayout> four = List.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS);
        for (int h = 0; h < heads; h++) {
          long off = (long) h * headDim * 4;
          MemorySegment mq = later(batchMatrix(seg(q), off, length, headDim, inner * 4L, seqBytes, batch));
          MemorySegment mk = later(batchMatrix(seg(k), off, length, headDim, inner * 4L, seqBytes, batch));
          MemorySegment mv = later(batchMatrix(seg(v), off, length, headDim, inner * 4L, seqBytes, batch));
          MemorySegment mo = later(batchMatrix(seg(out), off, length, headDim, inner * 4L, seqBytes, batch));
          ObjC.send(null, qk, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", four, cb, mq, mk, matS);
          encodeSoftmax(cb, scores, batch * length, length);
          ObjC.send(null, pv, "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:", four, cb, matS, mv, mo);
        }
        return null;
      });
    }

    @Override
    public void rmsNorm(Buffer x, Buffer out, int rows, int dim, Buffer gamma) {
      onGpu(() -> {
        encodeRows(open(), rmsPipeline, rows, dim, new MemorySegment[] {seg(x), seg(out), seg(gamma)}, new int[] {rows, dim});
        return null;
      });
    }

    @Override
    public void ropeSplit(Buffer qkv, Buffer q, Buffer k, Buffer v, int sequences, int length, int seqStride,
                          int posStride, int heads, int headDim, Buffer cos, Buffer sin) {
      kernel(ropePipeline, sequences * length * heads * headDim, new Buffer[] {qkv, q, k, v, cos, sin},
          new int[] {sequences, length, seqStride, posStride, heads, headDim});
    }

    @Override
    public void gateMerge(Buffer ctx, Buffer gates, Buffer gateBias, Buffer out, int sequences, int length,
                          int seqStride, int posStride, int heads, int headDim) {
      kernel(gatePipeline, sequences * length * heads * headDim, new Buffer[] {ctx, gates, gateBias, out},
          new int[] {sequences, length, seqStride, posStride, heads, headDim});
    }

    @Override
    public void addBias(Buffer x, Buffer bias, int rows, int cols, boolean gelu) {
      kernel(biasPipeline, rows * cols, new Buffer[] {x, bias}, new int[] {rows, cols, gelu ? 1 : 0});
    }

    private void kernel(MemorySegment pipeline, int threads, Buffer[] buffers, int[] ints) {
      onGpu(() -> {
        MemorySegment[] segs = new MemorySegment[buffers.length];
        for (int i = 0; i < buffers.length; i++) segs[i] = seg(buffers[i]);
        encodeKernel(open(), pipeline, threads, segs, ints);
        return null;
      });
    }
  }
}
