package io.github.ticofab.demucs;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Reads a PyTorch checkpoint ({@code torch.save} zip: a pickle plus raw tensor storages) without Python.
 *
 * <p>The pickle is interpreted by a small stack machine that never executes anything: Python globals become inert
 * {@link Global} references, and only the two constructions a checkpoint needs are understood
 * ({@code torch._utils._rebuild_tensor_v2} and {@code collections.OrderedDict}).
 */
public final class TorchCheckpoint {

  /** A Python global ({@code module.name}), kept as a name only. */
  public record Global(String module, String name) {}

  /** A call we don't interpret (e.g. numpy scalars in training metadata). */
  public record Opaque(Global callable, Object args) {}

  private record StorageRef(String dtype, String key) {}

  private record TensorRef(StorageRef storage, int offset, int[] shape, int[] stride) {}

  private final Object root;
  private final ZipFile zip;
  private final String prefix;

  private TorchCheckpoint(ZipFile zip, String prefix, Object root) {
    this.zip = zip;
    this.prefix = prefix;
    this.root = root;
  }

  public static TorchCheckpoint read(Path file) throws IOException {
    ZipFile zip = new ZipFile(file.toFile());
    ZipEntry pkl = zip.stream()
        .filter(e -> e.getName().endsWith("/data.pkl"))
        .findFirst()
        .orElseThrow(() -> new IOException("not a PyTorch checkpoint (no data.pkl): " + file));
    String prefix = pkl.getName().substring(0, pkl.getName().length() - "data.pkl".length());
    try (InputStream in = zip.getInputStream(pkl)) {
      return new TorchCheckpoint(zip, prefix, new Unpickler(in.readAllBytes()).load());
    }
  }

  /** The unpickled top-level object: maps, lists, Object[] tuples, strings, numbers, {@link Global}s. */
  public Object root() {
    return root;
  }

  /** All tensors found under {@code path} (a dict of name to tensor, e.g. {@code "state"}), widened to fp32. */
  public Map<String, Tensor> tensors(String... path) throws IOException {
    Object node = root;
    for (String key : path) node = ((Map<?, ?>) node).get(key);
    Map<String, Tensor> out = new LinkedHashMap<>();
    for (Map.Entry<?, ?> e : ((Map<?, ?>) node).entrySet()) {
      if (e.getValue() instanceof TensorRef ref) out.put((String) e.getKey(), load(ref));
    }
    return out;
  }

  private Tensor load(TensorRef ref) throws IOException {
    int numel = 1;
    for (int d : ref.shape) numel *= d;
    int[] expected = new int[ref.shape.length];
    for (int i = ref.shape.length - 1, s = 1; i >= 0; i--) {
      expected[i] = s;
      s *= ref.shape[i];
    }
    if (numel > 1 && !Arrays.equals(expected, ref.stride)) {
      throw new IOException("non-contiguous tensor in checkpoint: " + Arrays.toString(ref.stride));
    }
    ZipEntry entry = zip.getEntry(prefix + "data/" + ref.storage.key);
    byte[] bytes;
    try (InputStream in = zip.getInputStream(entry)) {
      bytes = in.readAllBytes();
    }
    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    float[] data = new float[numel];
    switch (ref.storage.dtype) {
      case "HalfStorage" -> {
        for (int i = 0; i < numel; i++) data[i] = Float.float16ToFloat(buf.getShort(2 * (ref.offset + i)));
      }
      case "FloatStorage" -> {
        for (int i = 0; i < numel; i++) data[i] = buf.getFloat(4 * (ref.offset + i));
      }
      case "BFloat16Storage" -> {
        for (int i = 0; i < numel; i++) data[i] = Float.intBitsToFloat(buf.getShort(2 * (ref.offset + i)) << 16);
      }
      default -> throw new IOException("unsupported storage type " + ref.storage.dtype);
    }
    return new Tensor(data, ref.shape);
  }

  /** The subset of the pickle protocol (up to 2, plus a few later opcodes) that torch.save emits. */
  private static final class Unpickler {
    private static final Object MARK = new Object();

    private final ByteBuffer in;
    private final Deque<Object> stack = new ArrayDeque<>();
    private final Map<Integer, Object> memo = new HashMap<>();

    Unpickler(byte[] bytes) {
      in = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    Object load() throws IOException {
      while (true) {
        int op = in.get() & 0xff;
        switch (op) {
          case 0x80 -> in.get(); // PROTO
          case 0x95 -> in.getLong(); // FRAME
          case '}' -> stack.push(new LinkedHashMap<>());
          case ']' -> stack.push(new ArrayList<>());
          case ')' -> stack.push(new Object[0]);
          case '(' -> stack.push(MARK);
          case 'N' -> stack.push(NONE);
          case 0x88 -> stack.push(Boolean.TRUE);
          case 0x89 -> stack.push(Boolean.FALSE);
          case 'J' -> stack.push(in.getInt());
          case 'K' -> stack.push(in.get() & 0xff);
          case 'M' -> stack.push(in.getShort() & 0xffff);
          case 0x8a -> stack.push(long1());
          case 'G' -> stack.push(Double.longBitsToDouble(Long.reverseBytes(in.getLong()))); // big-endian
          case 'X' -> stack.push(string(in.getInt()));
          case 0x8c -> stack.push(string(in.get() & 0xff)); // SHORT_BINUNICODE
          case 'T' -> stack.push(string(in.getInt())); // BINSTRING
          case 'U' -> stack.push(string(in.get() & 0xff)); // SHORT_BINSTRING
          case 'B' -> stack.push(bytes(in.getInt()));
          case 'C' -> stack.push(bytes(in.get() & 0xff));
          case 'c' -> stack.push(new Global(line(), line()));
          case 0x93 -> { // STACK_GLOBAL
            String name = (String) stack.pop();
            stack.push(new Global((String) stack.pop(), name));
          }
          case 't' -> stack.push(popMark().toArray());
          case 0x85 -> stack.push(new Object[] {stack.pop()});
          case 0x86 -> {
            Object b = stack.pop();
            stack.push(new Object[] {stack.pop(), b});
          }
          case 0x87 -> {
            Object c = stack.pop(), b = stack.pop();
            stack.push(new Object[] {stack.pop(), b, c});
          }
          case 'l' -> stack.push(new ArrayList<>(popMark()));
          case 'd' -> {
            List<Object> items = popMark();
            Map<Object, Object> m = new LinkedHashMap<>();
            for (int i = 0; i < items.size(); i += 2) m.put(items.get(i), items.get(i + 1));
            stack.push(m);
          }
          case 'q' -> memo.put(in.get() & 0xff, stack.peek());
          case 'r' -> memo.put(in.getInt(), stack.peek());
          case 0x94 -> memo.put(memo.size(), stack.peek()); // MEMOIZE
          case 'h' -> stack.push(memo.get(in.get() & 0xff));
          case 'j' -> stack.push(memo.get(in.getInt()));
          case 'Q' -> stack.push(persistent((Object[]) stack.pop()));
          case 'R' -> {
            Object args = stack.pop();
            stack.push(reduce((Global) stack.pop(), (Object[]) args));
          }
          case 0x81 -> { // NEWOBJ
            Object args = stack.pop();
            stack.push(reduce((Global) stack.pop(), (Object[]) args));
          }
          case 's' -> {
            Object v = stack.pop(), k = stack.pop();
            asMap(stack.peek()).put(k, v);
          }
          case 'u' -> {
            List<Object> items = popMark();
            Map<Object, Object> m = asMap(stack.peek());
            for (int i = 0; i < items.size(); i += 2) m.put(items.get(i), items.get(i + 1));
          }
          case 'a' -> {
            Object v = stack.pop();
            asList(stack.peek()).add(v);
          }
          case 'e' -> {
            List<Object> items = popMark();
            asList(stack.peek()).addAll(items);
          }
          case 'b' -> { // BUILD: only dict state is applied; anything else is metadata we ignore
            Object state = stack.pop();
            if (stack.peek() instanceof Map<?, ?> && state instanceof Map<?, ?> s) asMap(stack.peek()).putAll(s);
          }
          case '.' -> {
            return stack.pop();
          }
          default -> throw new IOException(String.format("unsupported pickle opcode 0x%02x at %d", op, in.position() - 1));
        }
      }
    }

    private static final Object NONE = new Object() {
      @Override
      public String toString() {
        return "None";
      }
    };

    private Object reduce(Global f, Object[] args) {
      String name = f.module + "." + f.name;
      return switch (name) {
        case "torch._utils._rebuild_tensor_v2" ->
            new TensorRef((StorageRef) args[0], (Integer) args[1], ints(args[2]), ints(args[3]));
        case "collections.OrderedDict" -> new LinkedHashMap<>();
        default -> new Opaque(f, args);
      };
    }

    private Object persistent(Object[] pid) {
      // ('storage', torch.<Type>Storage, key, location, numel)
      return new StorageRef(((Global) pid[1]).name(), (String) pid[2]);
    }

    private static int[] ints(Object tuple) {
      Object[] t = (Object[]) tuple;
      int[] out = new int[t.length];
      for (int i = 0; i < t.length; i++) out[i] = ((Number) t[i]).intValue();
      return out;
    }

    @SuppressWarnings("unchecked")
    private static Map<Object, Object> asMap(Object o) {
      return (Map<Object, Object>) o;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> asList(Object o) {
      return (List<Object>) o;
    }

    private List<Object> popMark() {
      List<Object> items = new ArrayList<>();
      while (stack.peek() != MARK) items.addFirst(stack.pop());
      stack.pop();
      return items;
    }

    private Object long1() {
      int n = in.get() & 0xff;
      long v = 0;
      for (int i = 0; i < n; i++) v |= (long) (in.get() & 0xff) << (8 * i);
      if (n > 0 && n < 8 && (v & (1L << (8 * n - 1))) != 0) v -= 1L << (8 * n);
      return v;
    }

    private String string(int n) {
      return new String(bytes(n), StandardCharsets.UTF_8);
    }

    private byte[] bytes(int n) {
      byte[] b = new byte[n];
      in.get(b);
      return b;
    }

    private String line() {
      StringBuilder sb = new StringBuilder();
      for (byte b; (b = in.get()) != '\n'; ) sb.append((char) b);
      return sb.toString();
    }
  }
}
