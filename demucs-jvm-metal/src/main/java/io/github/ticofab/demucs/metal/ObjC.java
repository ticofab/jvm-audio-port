package io.github.ticofab.demucs.metal;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * Just enough of the Objective-C runtime, through the FFM API, to drive Metal: classes, selectors and
 * {@code objc_msgSend} with an explicit signature per call (on arm64 it must be called with the callee's exact
 * C signature, never as a variadic).
 */
final class ObjC {

  private ObjC() {}

  static final Linker LINKER = Linker.nativeLinker();
  static final Arena ARENA = Arena.global();
  private static final SymbolLookup LIBOBJC = SymbolLookup.libraryLookup("/usr/lib/libobjc.A.dylib", ARENA);
  private static final MemorySegment MSG_SEND = LIBOBJC.find("objc_msgSend").orElseThrow();
  private static final MethodHandle GET_CLASS = function(LIBOBJC, "objc_getClass", FunctionDescriptor.of(ADDRESS, ADDRESS));
  private static final MethodHandle REGISTER_SEL = function(LIBOBJC, "sel_registerName", FunctionDescriptor.of(ADDRESS, ADDRESS));
  private static final MethodHandle POOL_PUSH = function(LIBOBJC, "objc_autoreleasePoolPush", FunctionDescriptor.of(ADDRESS));
  private static final MethodHandle POOL_POP = function(LIBOBJC, "objc_autoreleasePoolPop", FunctionDescriptor.ofVoid(ADDRESS));

  private static final Map<String, MemorySegment> SELECTORS = new ConcurrentHashMap<>();
  private static final Map<FunctionDescriptor, MethodHandle> SENDERS = new ConcurrentHashMap<>();

  static MethodHandle function(SymbolLookup lib, String name, FunctionDescriptor fd) {
    return LINKER.downcallHandle(lib.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), fd);
  }

  static MemorySegment cls(String name) {
    try {
      MemorySegment c = (MemorySegment) GET_CLASS.invokeExact(ARENA.allocateFrom(name));
      if (c.equals(MemorySegment.NULL)) throw new IllegalStateException("Objective-C class not found: " + name);
      return c;
    } catch (Throwable t) {
      throw rethrow(t);
    }
  }

  static MemorySegment sel(String name) {
    return SELECTORS.computeIfAbsent(name, n -> {
      try {
        return (MemorySegment) REGISTER_SEL.invokeExact(ARENA.allocateFrom(n));
      } catch (Throwable t) {
        throw rethrow(t);
      }
    });
  }

  /**
   * {@code [receiver selector args...]} with C signature {@code ret(id, SEL, argLayouts...)}; {@code ret} null means
   * void.
   */
  static Object send(MemoryLayout ret, MemorySegment receiver, String selector, List<MemoryLayout> argLayouts,
                     Object... args) {
    MemoryLayout[] all = new MemoryLayout[argLayouts.size() + 2];
    all[0] = ADDRESS;
    all[1] = ADDRESS;
    for (int i = 0; i < argLayouts.size(); i++) all[i + 2] = argLayouts.get(i);
    FunctionDescriptor fd = ret == null ? FunctionDescriptor.ofVoid(all) : FunctionDescriptor.of(ret, all);
    MethodHandle h = SENDERS.computeIfAbsent(fd, d -> LINKER.downcallHandle(MSG_SEND, d));
    Object[] full = new Object[args.length + 2];
    full[0] = receiver;
    full[1] = sel(selector);
    System.arraycopy(args, 0, full, 2, args.length);
    try {
      return h.invokeWithArguments(full);
    } catch (Throwable t) {
      throw rethrow(t);
    }
  }

  /** {@code [receiver selector]} returning an object. */
  static MemorySegment id(MemorySegment receiver, String selector) {
    return (MemorySegment) send(ADDRESS, receiver, selector, List.of());
  }

  /** {@code [receiver selector]} returning nothing. */
  static void call(MemorySegment receiver, String selector) {
    send(null, receiver, selector, List.of());
  }

  static void release(MemorySegment object) {
    if (!object.equals(MemorySegment.NULL)) call(object, "release");
  }

  static MemorySegment poolPush() {
    try {
      return (MemorySegment) POOL_PUSH.invokeExact();
    } catch (Throwable t) {
      throw rethrow(t);
    }
  }

  static void poolPop(MemorySegment pool) {
    try {
      POOL_POP.invokeExact(pool);
    } catch (Throwable t) {
      throw rethrow(t);
    }
  }

  private static RuntimeException rethrow(Throwable t) {
    if (t instanceof RuntimeException r) return r;
    if (t instanceof Error e) throw e;
    return new RuntimeException(t);
  }
}
