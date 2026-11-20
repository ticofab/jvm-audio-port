# Plan

How to replace splitter's Python with JVM libraries, from the analysis in `notes/`. Numbers marked *est.* are
estimates from FLOP counts and published throughput, not measurements; the first milestone exists to replace them.

## The short version

- **Port the two separation models.** That is where the Python actually is (PyTorch + audio-separator), and it is
  well-bounded: ~4–5k lines for Demucs, ~2.5–4k for BS-Roformer, sharing most of the infrastructure.
- **Don't port yt-dlp.** Its YouTube code is only ~12k lines, but YouTube changes something every few weeks (PO
  tokens, the SABR protocol, JS challenges that now need a real JS runtime). A port means owning that arms race alone.
  Use yt-dlp's standalone binary (no Python needed), wrapped as a `Flow.Publisher`.
- **Keep ffmpeg.** It isn't Python, and pure-JVM codecs (Concentus for Opus, jump3r for MP3) cover only part of
  what it does.
- **Pure Java first, faster backends behind an interface.** Correctness and portability come from a pure-Java
  (Vector API) backend; speed on Apple Silicon from calling Accelerate through the FFM API, both chosen at runtime.
- **Everything as `java.util.concurrent.Flow` components**, TCK-tested, so Pekko, Reactor and RxJava can all use
  them.

## What we'd build

A set of Java libraries (Java, not Scala, so every JVM language can use them without dragging in a Scala runtime):

| Module | Contents | Size *est.* |
|--------|----------|------------:|
| `tensor` | Minimal fp32 tensors and the ops both models need: matmul, conv1d/2d (+ transposed), GroupNorm/LayerNorm/RMSNorm, GELU (exact, erf-based), GLU, softmax attention, rotary and sinusoidal embeddings | ~1.5k |
| `tensor-accelerate` | Backend calling Apple Accelerate (`cblas_sgemm`, vDSP) through FFM; macOS only | ~0.5k |
| `weights` | Reader for PyTorch checkpoints (zip + pickle, never executing it) and safetensors | ~0.4k |
| `dsp` | STFT/iSTFT matching `torch.stft`/`istft` conventions exactly, crossover filter, PCM framing | ~0.5k |
| `demucs` | htdemucs (v4) inference: the bass sub-model `d12395a8` of htdemucs_ft | ~0.9k |
| `roformer` | BS-Roformer inference (BS-Roformer-SW; bass head only) | ~0.8k |
| `separation` | Chunking + overlap-add as `Flow.Processor`s, model-weight download/caching | ~0.5k |
| `youtube` | yt-dlp binary as a `Flow.Publisher<ByteBuffer>` | ~0.2k |
| tests | Per-layer and end-to-end comparison against PyTorch dumps, Flow TCK | ~1.5k |

Python appears only at development time, once, to dump reference tensors from PyTorch for the tests. Users never
need it.

## Milestones

### 0. Benchmark gate (week 1)

The biggest unknown decides the whole speed story: how fast can the JVM multiply matrices on this machine?
- Pure Java with the Vector API vs Accelerate through FFM, on the shapes that dominate: Demucs' transformer
  (2688×512 by 512×2048) and its 3×3 convs, Roformer's attention over 801 frames.
- Expectation *est.*: Vector API 100–400 GFLOP/s (128-bit NEON); Accelerate 1–2 TFLOP/s (uses Apple's matrix
  units); PyTorch MPS today ~1–2 TFLOP/s.
- Also try TornadoVM's Metal backend if it runs on our JDK: it's the only route to the GPU without writing
  Objective-C glue.

**Exit:** measured numbers in `notes/benchmarks.md`, and a backend decision.

### 1. Infrastructure (weeks 2–4)

`weights` (load both checkpoints in Java, including Demucs' fp16), `tensor` with the pure-Java backend, `dsp` with an
STFT/iSTFT that matches PyTorch to ~1e-6, and the reference-dump tooling: a dev-only Python script run once in
splitter's existing environment, saving every layer's input and output for one fixed chunk.

### 2. Demucs, bass sub-model only (weeks 5–8)

Chosen first: ~15× less compute than BS-Roformer (~14 TFLOP per song at 1 shift vs ~220), so it is usable even on the
pure-Java CPU backend (*est.* 1–2.5 min per song), and demucs.cpp exists as a cross-check.
- Validation: each layer against PyTorch given PyTorch's input, then a segment (target ≥ 70 dB SNR), then a song
  with fixed shift offsets (≥ 60 dB). For scale: Demucs' own random shifts make two PyTorch runs differ by 25–32 dB.
- Fix what audio-separator does wastefully: run only the bass sub-model (4× less work than today).

### 3. BS-Roformer, bass head only (weeks 9–12)

Reuses `tensor`, `weights`, `dsp`. New: band split, rotary embeddings, the gated attention, per-band mask heads.
- Bass head only: exact, and drops 70% of the weights (51 M of 175 M parameters).
- Needs the fast backend: *est.* 9–18 min per song on pure Java, near today's 2 min with Accelerate.

### 4. Streaming separation (weeks 13–14)

`Flow.Processor`s that take PCM chunks and emit bass chunks, with one model segment (~8–9 s of audio) as the unit
of backpressure. Decisions this needs:
- Demucs' whole-song mean/std normalization: replace with per-segment statistics (each segment is normalized again
  inside the model anyway) and measure the difference.
- The zero-phase crossover can't stream: replace it with a linear-phase FIR, a fixed ~10–50 ms delay with the same
  phase behaviour, and check it against today's output.
- Latency floor: one model window, ~10 s from the first downloaded byte to the first bass sample.

### 5. Splitter on the JVM libraries (week 15)

Replace the Python environment and `separate.py` with the libraries, wired as a Pekko graph: yt-dlp publisher →
ffmpeg decode → broadcast to both models → zip → crossover → ffmpeg encode. Success criterion: the MP3 matches
today's approved output (≥ 50 dB SNR, i.e. inaudible) and no `uv`/Python on the machine.

**Total *est.*:** ~15 weeks for one experienced developer, with milestone 0 able to move it either way.

## Biggest risks

1. **Speed.** If the pure-Java backend is far off and Accelerate disappoints, BS-Roformer becomes a 10+ minute step.
   Milestone 0 is there to find out first.
2. **Silent layout errors.** A wrong reshape, a swapped half of a GLU or a mismatched STFT convention still
   produces plausible-sounding audio. Only layer-by-layer comparison against PyTorch catches these, which is why
   the reference dumps come before any model code.
3. **Model licenses.** Demucs code and weights are MIT. BS-Roformer-SW's weights are a community release whose
   license still needs checking before anything is published.
4. **YouTube.** Out of our hands either way; the binary route keeps it someone else's problem.

## Decisions for you

1. **yt-dlp:** standalone binary (recommended), or NewPipeExtractor (pure JVM, but GPL-3.0, which would make
   anything linking it GPL, and its YouTube fixes lag yt-dlp's by days to weeks)?
2. **Scope of "native":** is a macOS-only fast path through Accelerate acceptable, alongside a portable pure-Java
   backend? (CUDA/Linux GPUs would be a later backend.)
3. **Library language and JDK:** Java on JDK 25 (LTS) is recommended. Note the Vector API is still an incubator
   module (12th round in JDK 27), so users need `--add-modules jdk.incubator.vector`.
4. **License** for the libraries (Apache-2.0 would match the Pekko ecosystem).

## Quick win, independent of all this

Today's splitter runs all four Demucs sub-models although bass needs one. Running only `d12395a8` should cut the
Demucs half of separation by about 4× (roughly 4 min → 2½ min per song) without touching quality. Worth doing in
splitter now, regardless of the port.
