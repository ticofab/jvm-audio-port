# demucs-jvm

Pure-Java inference for music source separation: [Demucs](https://github.com/adefossez/demucs) v4 (HTDemucs) and
BS-Roformer. No Python, no native libraries, no dependencies: it reads the original PyTorch checkpoints directly and
runs on the CPU on any JVM 25+, or on a GPU through an optional backend.

```java
HTDemucs model = HTDemucs.load(Path.of("d12395a8-e57c48e6.th")); // htdemucs_ft's bass network
Separator separator = new Separator(model, 2, new Random());      // 2 shifted passes, like Demucs' default
float[][][] stems = separator.separate(stereo, progress -> {});   // [sources][2][frames]
float[][] bass = stems[model.sources().indexOf("bass")];
```

BS-Roformer (e.g. BS-Roformer-SW; only the requested stem's head is loaded):

```java
BSRoformer roformer = BSRoformer.load(Path.of("BS-Roformer-SW.ckpt"), 0); // stem 0 = bass
float[][] bass = new RoformerSeparator(roformer).separate(stereo, progress -> {});
```

As a Reactive Streams component (`java.util.concurrent.Flow`), usable from Pekko/Akka Streams (`JavaFlowSupport`),
Reactor (`JdkFlowAdapter`), RxJava and others:

```java
var processor = new SeparationProcessor(separator.stem("bass"), 44_100, executor, progress -> {});
// subscribe it to a Publisher<float[]> of interleaved stereo frames; it publishes the bass the same way
```

It currently collects the whole input before separating (Demucs normalizes by whole-song statistics), then emits
the result with backpressure.

## Accuracy

Against PyTorch (CPU, fp32) on *L'impresa eccezionale*, htdemucs_ft bass network:

| Test | SNR vs PyTorch |
|------|---------------:|
| One 7.8 s segment, `HTDemucs.forward`, bass | 121.7 dB (drums 111.4, other 116.4, vocals 123.7) |
| 30 s clip, full `Separator` (no shifts), bass | 120.5 dB |
| BS-Roformer-SW, one 9.3 s chunk, bass | 122.7 dB |

That is float rounding: for scale, Demucs' own random shifts make two PyTorch runs differ by 25–32 dB.

## Speed and hardware backends

Pure Java runs everywhere and is the default: ~3.9 s per 7.8 s segment on an Apple M1 Max (all cores), about half
of PyTorch's CPU speed.

Like PyTorch's device backends, accelerators are optional artifacts, discovered at runtime through
`io.github.ticofab.demucs.spi.Accelerator` (`ServiceLoader`). Add one to the classpath and it is used when its
hardware is present; `-Ddemucs.backend=java` forces the CPU path.

| Backend | Artifact | Hardware | Demucs segment time (M1 Max) |
|---------|----------|----------|----------------------:|
| `java` (built in) | `demucs-jvm` | any CPU | 3.9 s (BS-Roformer: 35 s per 9.3 s chunk) |
| `metal` | `demucs-jvm-metal` | Apple GPU (Metal Performance Shaders) | 1.1 s (BS-Roformer: 5.1 s) |

`demucs-jvm-metal` is pure Java too: it calls Metal through the FFM API and the Objective-C runtime, so nothing
native is shipped. Run with `--enable-native-access=ALL-UNNAMED` (or the module name) to silence the JDK's
native-access warning. Its output matches PyTorch as closely as the CPU path (121 dB on the bass segment).

## Scope

BS-Roformer: the MSST/audio-separator architecture without skip connections, hyper-connections or linear attention
(BS-Roformer-SW and similar); band layout and depth are read from the checkpoint.

HTDemucs: the released configuration (depth 4, 48 channels, CaC spectrogram branch, 5-layer cross
transformer at 512, as in `htdemucs` and `htdemucs_ft`); other configurations are rejected at load time.
