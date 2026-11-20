# JVM landscape: replacing splitter's Python tools

Research notes, 2026-11-20. Scope: everything except the two models' architectures (covered elsewhere), meaning
YouTube download, NN inference runtimes on the JVM (Apple Silicon first), reading PyTorch checkpoints, audio DSP / codecs,
and Reactive Streams packaging.

Environment checked locally: Apple M1 Max, macOS 27.2.1, Homebrew OpenJDK 27.0.1.1, splitter on Pekko 2.0.0 /
Scala 3.10.1. Where I could not verify something first-hand it is marked **(unverified)**.

---

## TL;DR

| Area | Recommendation |
|---|---|
| YouTube download | **Keep yt-dlp, but as a standalone binary** (`yt-dlp_macos` or `brew install yt-dlp`) plus `deno`. Don't port it. Use NewPipeExtractor only if "no external process at all" becomes a hard requirement *and* GPL-3.0 is acceptable. |
| NN inference | **ONNX Runtime Java (CoreML EP + CPU)** is the lowest-effort way to get fast inference with no Python. The most native option is **pure Java plus FFM to Accelerate (CPU/AMX) or Metal/MPSGraph (GPU)**, at high effort. Keep DJL/libtorch as the reference-accuracy oracle, not the product. TornadoVM (Metal backend) is the interesting wildcard. |
| Checkpoints | Write a small **Java PyTorch-zip reader** on top of `net.razorvine:pickle` (Pyrolite) with a custom `persistentLoad`. Normalize everything to **safetensors** (trivial to read from Java with FFM `MemorySegment` mmap). |
| DSP | All small: STFT via JTransforms (or a hand-written Vector API radix-2/4 FFT, since sizes are fixed powers of two), `sosfiltfilt` port (~150 LOC), WAV I/O by hand or `javax.sound.sampled`. |
| Codecs | **Keep ffmpeg** for Opus/AAC decode and MP3 encode. Pure-JVM paths exist (Concentus for Opus, jump3r for LAME-MP3) but they're slower and the demuxing is extra work. ffmpeg isn't Python, so it doesn't conflict with the stated goal. |
| Reactive Streams | Implement components directly against `java.util.concurrent.Flow`, verify with `reactive-streams-tck-flow` 1.0.4, and plug into Pekko via `JavaFlowSupport`. Element = fixed-size audio chunk (e.g. one model segment); `request(n)` counts chunks, not samples. |

---

## 1. yt-dlp

### 1.1 Size (measured on the clone in `sources/yt-dlp`)

Version `2026.10.26` (clone HEAD `4adbf698`, 2026-11-19, shallow clone).

| Part | Lines (`.py` incl. comments/blank) |
|---|---:|
| `yt_dlp/` total | 226,070 |
| `yt_dlp/extractor/` (941 files, ~1,800 sites) | 196,381 |
| `yt_dlp/extractor/youtube/` (all) | 11,725 py (12,551 incl. vendored JS + READMEs) |
| of which `_video.py` (watch-page/player extraction) | 4,612 |
| `_base.py` (innertube clients, ytcfg, auth) | 1,388 |
| `_tab.py` (channels/playlists, not needed) | 2,556 |
| `jsc/` JS-challenge framework (director, providers for deno/bun/node/quickjs, EJS loader) | ~1,200 py |
| `jsc/_builtin/vendor/yt.solver.core.js` (EJS solver, vendored) | 368 lines / 12.6 KB |
| `pot/` PO-token provider framework | ~1,550 py |
| `yt_dlp/jsinterp.py` (legacy pure-Python JS interpreter) | 971 (no longer referenced by the YouTube extractor; grep finds no use) |
| Core you'd also need: `YoutubeDL.py` 4,552, `extractor/common.py` 4,185, `networking/` 2,606, `downloader/` 4,275, `utils/` 7,330 | |

A YouTube-only, audio-only port is realistically **5–8k lines of Java**: innertube client, format selection, ranged
HTTP download, JS challenge hook, PO-token hook. The lines aren't the problem, though. The problem is the **churn**
(see 1.4).

### 1.2 What downloading one audio stream involves (2026)

1. **Watch page / ytcfg.** Fetch `https://www.youtube.com/watch?v=…` to get `ytcfg` (API key, visitor data,
   client version) and the **player JS URL** (`/s/player/<id>/player_ias.vflset/…/base.js`). The player id changes
   every few days.
2. **Innertube `/youtubei/v1/player` POST** with a client context. yt-dlp ships 15 client profiles in
   `INNERTUBE_CLIENTS` (`web`, `web_safari`, `web_embedded`, `web_music`, `web_creator`, `android`, `android_vr`,
   `ios`, `ios_producer`, `visionos`, `mweb`, `tv`, `tv_downgraded`, `tv_samsung`, `tv_simply`). Current defaults
   (`_video.py:144`) are `('tv_samsung', 'visionos', 'web')`. If no JS runtime is available it uses `('visionos',)`
   only, a client marked `REQUIRE_JS_PLAYER: False`. Each client has different PO-token rules, format sets, and
   breakage. The `web` client needs a **GVS PO token** for HTTPS/DASH formats (`WEB_PO_TOKEN_POLICIES`). TV clients
   currently don't, but need the JS player.
3. **`streamingData.adaptiveFormats`** lists audio-only formats (Opus in WebM, itag 251 ~160 kbps; AAC in MP4,
   itag 140). Each URL is either plain or carries an obfuscated `s` (signature) in `signatureCipher`, and almost
   always an `n` parameter.
4. **JS challenges.** Two functions inside the player JS:
   - **sig**: transforms `s` into the real `sig` query parameter. Without it you get a 403.
   - **n**: transforms the `n` parameter. Without it YouTube throttles to ~tens of KB/s or returns a 403.
   Since late 2025 these are heavily obfuscated (global string arrays, control-flow tricks), and regex extraction plus a
   toy interpreter stopped being reliable. yt-dlp 2025.11.12 moved to **EJS**: the `yt-dlp/ejs` solver
   (`yt.solver.core.js`, built on `meriyah` 6.1.4 + `astring` 1.9.0, i.e. a real JS parser/AST rewriter) runs inside
   an **external JS runtime**. **Deno ≥ 2.3** is the default and the only one enabled by default; Node ≥ 22, Bun ≥ 1.2.11,
   and QuickJS/QuickJS-ng are opt-in via `--js-runtimes` (version floors from `yt_dlp/utils/_jsruntime.py`). Running
   YouTube without a JS runtime is deprecated, and you only get the reduced `visionos` format set.
   Sources: https://github.com/yt-dlp/yt-dlp/issues/15012 , https://github.com/yt-dlp/ejs ,
   https://github.com/yt-dlp/yt-dlp/wiki/EJS
5. **PO tokens** ("proof of origin"): BotGuard/DroidGuard attestation tokens bound to visitor data or video id.
   Generating them for web needs running Google's BotGuard VM in a browser-like JS environment. yt-dlp delegates this
   to plugins (`pot/` framework, e.g. the `bgutil` provider). NewPipeExtractor exposes a `PoTokenProvider` interface,
   which the NewPipe app implements with an Android WebView. NewPipeExtractor v0.27.0 (2026-11-07) **removed**
   Android/iOS PO-token paths because "poToken for ANDROID and IOS clients cannot be generated anymore".
6. **SABR.** YouTube is progressively forcing its proprietary "SABR" streaming protocol (server-driven adaptive
   bitrate, protobuf over POST) on clients, which removes plain URLs. Both yt-dlp (`_video.py:3549` "YouTube is forcing
   SABR streaming for this client") and NewPipe (PR #1508, merged 2026-06-09: "Workaround SABR enforcement by using
   another player client") deal with it by hopping to clients not yet affected.
7. **Download** with HTTP range requests in chunks (YouTube throttles single long GETs), then feed the bytes to
   ffmpeg (splitter does exactly this today: `yt-dlp -f bestaudio -o - | ffmpeg … -ac 2 -ar 44100`).

### 1.3 JVM prior art: NewPipeExtractor

- Repo: https://github.com/TeamNewPipe/NewPipeExtractor ; **GPL-3.0**; ~2.1k stars; last push 2026-11-19.
- Releases in 2026: v0.25.0 (Jan 11), 0.25.1, 0.25.2, 0.26.0 (Feb 22), 0.26.1–0.26.5 (Apr–Aug), **0.27.0 (2026-11-07)**,
  **0.27.1 (2026-11-10)**. v0.27.0 raised the baseline to **Java 17**. Distributed via JitPack.
- **92 commits** touched `services/youtube/` in the last 12 months (GitHub API, since 2025-11-20).
- **Deciphering:** it uses **Mozilla Rhino** (`org.mozilla:rhino` 1.8.1, pinned because 1.9.0 needs Android minSDK 26)
  in *interpreted mode* with `initSafeStandardObjects()` (see `utils/JavaScript.java`). The sig and n functions are
  **cut out of the player JS with regexes** (`YoutubeSignatureUtils`, `YoutubeThrottlingParameterUtils`, plus a
  hand-written `EcmaScriptTokenStream` to find matching braces) and only those snippets run in Rhino. There's also a
  `JavaScript.setExternalJavaScriptInterpreter(...)` hook to plug in another engine. The regexes are the fragile part:
  recent commits include "[YouTube] Fix n-sig parsing for player c9168c90" (2026-01-27) and "[YouTube] Update n-sig
  parsing" (2026-09-07). This is the approach yt-dlp gave up on in favour of a full JS parser.
- Clients used: WEB, WEB_REMIX, VISIONOS etc. (`ClientsConstants.java`, WEB version `2.20260805.01.00`).
- **How often it breaks:** the NewPipe app shipped several 2026 hotfixes for YouTube ("Content not available", SABR
  enforcement, e.g. NewPipe 0.28.6, https://newpipe.net/blog/). Downstream apps such as Spotube had to bump the
  extractor to fix playback (Feb 2026, https://opencollective.com/spotube/updates/v5-1-1-release-notes). Expect
  **breakage roughly every 1–3 months, with a fix lag of days to weeks**. yt-dlp typically fixes within hours to days and
  had ≥13 releases in the last 12 months. VLC dropped its YouTube Lua script in May 2026 as unmaintainable
  (https://mailman.videolan.org/pipermail/vlc-commits/2026-May/073117.html), a useful signal of the difficulty.
- Library ergonomics: it requires you to implement a `Downloader` (HTTP) class and is designed around the Android app.
  It works on a desktop JVM (several desktop and server projects use it), but there is no stable API promise.
- **License impact:** GPL-3.0 is viral for distribution. If splitter is distributed, linking NewPipeExtractor makes the
  whole binary GPL. That's fine for a personal tool, a blocker for anything closed.

Other JVM options: a few small Java/Kotlin innertube wrappers exist, but none is maintained at yt-dlp's level. Embedding a
JS engine yourself is possible (Rhino, GraalJS/Polyglot, or **running the EJS solver itself on GraalJS**, since it's
Unlicense, ~12.6 KB plus meriyah/astring). In principle that would let a JVM port use the *same* solver as yt-dlp.
Whether GraalJS runs meriyah fast enough on a ~2–3 MB player JS is **unverified**.

### 1.4 Standalone binary (no Python env)

- yt-dlp GitHub release `2026.10.26` ships **`yt-dlp_macos`** (PyInstaller-frozen universal binary, no Python
  needed, and it **bundles `yt-dlp-ejs`**) plus `yt-dlp_macos.zip`. It still needs **deno** on `PATH` (or bun/node via
  `--js-runtimes`).
- Homebrew: `yt-dlp` 2026.10.26 (Unlicense) depends on `python@3.15`, `deno`, `curl-impersonate`, `certifi`, `cffi`.
  So brew installs a Python, but it's brew-managed, not a venv the user builds. `deno` is 2.9.8 in brew today.
- Self-update: `yt-dlp_macos -U` updates in place. Keeping it current is the whole game, since YouTube changes weekly.

### 1.5 Recommendation (honest)

1. **Keep yt-dlp as an external binary** (`yt-dlp_macos` downloaded and auto-updated by splitter into a cache dir, plus
   `deno`), wrapped as a `Flow.Publisher<ByteBuffer>` around the process's stdout. This meets "no Python env to set up"
   with zero maintenance burden on you. It is the only option that tracks YouTube's weekly changes for free.
2. **NewPipeExtractor** if you want an in-JVM library and accept GPL-3.0 and occasional multi-week outages. Wrap it
   behind the same Publisher interface so you can switch.
3. **Don't port yt-dlp.** The code is small, but you'd be signing up to re-solve every YouTube change (player JS
   obfuscation, SABR, PO tokens, client churn) alone, forever.

---

## 2. Neural-net inference on the JVM without Python (Apple Silicon)

### 2.1 Building blocks and their 2026 status

**ONNX Runtime Java** (`com.microsoft.onnxruntime:onnxruntime`)
- Latest on Maven Central: **1.32.0** (2026-11-13). I downloaded it and inspected it: the jar bundles
  `native/osx-aarch64/libonnxruntime.dylib`, which **links `CoreML.framework` and `Metal.framework`**, exports
  `OrtSessionOptionsAppendExecutionProvider_CoreML`, and contains `CoreMLExecutionProvider` and
  `WebGpuExecutionProvider` strings. The Java API has `SessionOptions.addCoreML()`, `addCoreML(EnumSet<CoreMLFlags>)`, and
  `addCoreML(Map<String,String>)` (options such as `ModelFormat=MLProgram`, `MLComputeUnits=CPUAndGPU|ALL|…`).
  https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.SessionOptions.html ,
  https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
  - **(unverified)** whether the WebGPU EP is actually usable from the Java jar on macOS. I tried to check with
    `OrtEnvironment.getAvailableProviders()` but the probe was blocked in this session. Run that one-liner before relying
    on it.
  - Caveats: the CoreML EP **partitions** the graph and falls back to CPU for unsupported ops/shapes, and each
    partition boundary costs a copy. Transformer attention plus complex reshapes in Roformer/HTDemucs may fragment
    badly. Dynamic shapes hurt (use fixed segment length). STFT/iSTFT should stay *outside* the ONNX graph (do them in
    Java), which is also how most published Demucs/Roformer ONNX exports work. Export to ONNX needs Python *once*, at build
    time, or you can download pre-exported models. Some community ONNX exports of htdemucs/BS-Roformer exist (other
    agents should confirm fidelity).
  - Python-free at runtime: **yes**. Native code: yes (C++ in a jar), but no install step.

**DJL (Deep Java Library)**, https://github.com/deepjavalibrary/djl , Apache-2.0
- Latest **v0.40.0 (2026-11-16)**, monthly cadence. Engines: PyTorch (libtorch), OnnxRuntime (bundles ORT **1.21.1**,
  old), TensorFlow (2.16.1), XGBoost, LightGBM, HF tokenizers (Rust). (MXNet/PaddlePaddle/TFLite were dropped in earlier
  releases.)
- **PyTorch engine = libtorch via JNI, no Python.** By default it downloads the native libtorch into a cache on first
  use, or you can bundle `ai.djl.pytorch:pytorch-native-cpu:2.7.1:osx-aarch64`. Supported libtorch: 1.13.1, 2.5.1,
  **2.7.1** (recommended), so it trails PyTorch by over a year. x86_64 macOS dropped since 0.30.0.
  https://docs.djl.ai/master/engines/pytorch/pytorch-engine/index.html
- **MPS:** `PtDeviceType` maps `"mps"` to libtorch device type 13 (in source), so `Device.of("mps", 0)` works in
  principle. MPS op coverage is whatever libtorch 2.7.1 has (FFT on MPS has historically had gaps).
- **Model format:** DJL's PyTorch engine loads **TorchScript** (`torch.jit.trace/script`), not arbitrary pickled
  `state_dict`s. You'd need a one-time TorchScript export (Python, at build time). Alternatively, build the network with
  DJL's NDArray API and load weights yourself (see §3).
- Role: best **numerical oracle** (same kernels as Python torch), heaviest dependency (~200+ MB libtorch).

**Java Vector API.** Still incubating. JDK 27 (GA 2026-09-15) delivered **JEP 537 "Vector API (Twelfth Incubator)"**,
and **JDK-8397768 "Vector API (Thirteenth Incubator)" is already Submitted** for JDK 28. It's waiting on Valhalla value
classes, not on API problems. Needs `--add-modules jdk.incubator.vector`. https://openjdk.org/jeps/537 ,
https://openjdk.org/jeps/0
- On M1/M2/M3 the Vector API maps to **128-bit NEON** (4 floats per vector). Apple's **AMX / SME** matrix units (SME on
  M4+) are *not* reachable from the Vector API; only Accelerate/BNNS/CoreML use them (M4-SME use by HotSpot:
  **unverified**, I believe no).

**FFM API (Project Panama).** Final since JDK 22 (JEP 454). `jextract` generates bindings from C headers.
- **Accelerate** (C API: `cblas_sgemm`, vDSP FFT `vDSP_DFT_*`/`vDSP_fft_zrip`, vForce, BNNS Graph) is directly callable
  via FFM: `SymbolLookup.libraryLookup("/System/Library/Frameworks/Accelerate.framework/Accelerate", arena)`. This is
  the cheapest route to "AMX-speed GEMM" from Java. No Java binding project exists that I could find, but it's a few
  hundred lines of jextract output.
- **Metal / MPS / MPSGraph** are Objective-C/Swift APIs. From FFM you call `objc_msgSend` via `libobjc` (TornadoVM's
  Metal backend does exactly this, see "Resolve the Objective-C message-send handles once" in its 7.3 changelog), or you
  write a thin C/Swift shim with a C ABI. Doable but laborious.
- **MLX via FFM:** `Alipsa/jmlx` (MIT, 1 star, active Nov 2026) wraps Apple MLX through mlx-c with jextract bindings on
  JDK 25. It's a prototype and needs a native bootstrap build. It does show that MLX (Apple's own GPU array library, with
  FFT, conv, and attention ops) is reachable from Java. https://github.com/Alipsa/jmlx

**TornadoVM**, https://github.com/beehive-lab/TornadoVM , relicensed to **Apache-2.0** in 8.0
- Very active: **8.0.0 (2026-10-26)** replaced Graal with its own "TC" bytecode-to-kernel compiler. **8.1.0 (2026-11-06)**
  publishes OpenCL **and Metal** drivers on Maven Central ("run without an SDK").
- **Metal backend** since 4.0 (early 2026, PR #796), with `simdgroup_float8x8` hardware-matrix GEMMs (#850),
  half-precision SIMD-group matrix ops (#1223), shared command buffers ("6x LLM decode", #1216), and **MLX library tasks
  on the Metal backend** (#1188, #1237). Changelog: https://github.com/beehive-lab/TornadoVM/blob/develop/docs/source/CHANGELOG.rst
- You write kernels in Java (`@Parallel` loops or the `KernelContext` API). Prior art: GPULlama3.java (MIT, "jitllm"
  v3.0.0, 2026-11-12) runs Llama/Qwen/etc. on Metal through TornadoVM.
- It needs a TornadoVM-patched JDK/launcher setup (less so since 8.1 Maven artifacts). Exact JDK 27 support:
  **unverified**.

**Project Babylon / HAT**, https://openjdk.org/projects/babylon
- **JDK-8361105 "Code Reflection (Incubator)" is in state *Submitted*, not targeted** (JEP index, today). HAT lives in the
  `openjdk/babylon` repo (commits daily, last 2026-11-18). **Backends: `cuda`, `hip`, `opencl`, `spirv`. No Metal.**
  On macOS it runs only via Apple's deprecated OpenCL. Babylon also has ONNX work (Java-to-ONNX code models, an ORT FFM
  runtime, a CoreML provider option merged Oct 2025). It's research-grade, so don't build on it in 2026.

**Pure-Java inference prior art**
- **llama3.java** (mukel, MIT; last push 2026-04): single-file Llama inference with Vector API matvec for quantized
  weights. Self-reported CPU: Llama-3-8B Q4_0 **6.95 tok/s vs llama.cpp 7.53** (i9-13900H), **8.03 vs 9.26** (Ryzen
  3950X). That's about 80–93% of llama.cpp. https://github.com/mukel/llama3.java
- **Jlama** (tjake, Apache-2.0): pure-Java LLM engine with Vector API, a pure-Java safetensors loader, and an optional native
  kernels module. **Last release v0.8.4 (2025-01-01), last push 2025-10.** Effectively in maintenance limbo.
  https://github.com/tjake/Jlama
- **Important caveat:** LLM decode is a **memory-bound matrix-vector** workload, which is why Java gets close to C there.
  Source separation (Demucs convs plus transformer, Roformer attention over a whole segment, batch of frames) is
  **compute-bound GEMM/conv**. Here the gap to Accelerate (AMX) or the GPU is much bigger. Back-of-envelope: M1 Max P-core
  NEON peak is ~100 GFLOP/s fp32 per core (4×FMA×4 lanes×~3.2 GHz), ~0.8 TFLOP/s over 8 P-cores *peak*. A Java GEMM
  typically reaches 30–60% of that. Accelerate's AMX sgemm reaches roughly 1–2 TFLOP/s on one cluster, and the M1 Max GPU is
  ~10 TFLOP/s fp32. All **estimates, unverified**; benchmark with JMH before choosing.

### 2.2 Comparison

| Approach | Truly native JVM / no Python at runtime | Python needed at build time? | Expected speed on M1 Max (relative to PyTorch MPS today = 1×) | Effort | Notes |
|---|---|---|---|---|---|
| **Pure Java CPU (Vector API)** | Yes, 100% Java (incubator flag) | No (if weights are read from .ckpt/.th directly, §3) | ~0.05–0.2× (**guess**). Convs/attention compute-bound on 128-bit NEON. | High: write every op (conv1d/2d, GroupNorm, LayerNorm, attention, rotary, GELU/GLU, STFT) | Maximum control and portability, slowest. Great as a reference/fallback. |
| **Java + FFM → Accelerate (cblas/BNNS/vDSP)** | Yes (Java + OS framework, no extra natives) | No | ~0.3–0.8× (**guess**). GEMMs on AMX, glue in Java. | High (same ops as above, but heavy ops become sgemm calls via im2col) | Best "pure JVM ethos with real speed" on CPU. macOS-only unless you add an OpenBLAS fallback. |
| **Java + FFM → Metal / MPSGraph / MLX** | Yes, but needs an ObjC shim or `objc_msgSend` plumbing (or mlx-c dylib) | No | ~0.7–1.2× (**guess**) | Very high (Metal); medium-high via MLX (MLX has the ops) | MLX route = depend on a native dylib (like ORT). jmlx is a prototype. |
| **TornadoVM (Metal backend)** | Yes (Java kernels compiled to MSL) | No | ~0.3–1× depending on kernel quality (**guess**) | High (write kernels), plus TornadoVM runtime setup | Active project with Metal GEMM primitives and MLX tasks. Worth a spike. |
| **ONNX Runtime Java (CPU / CoreML EP)** | Python-free; native C++ shipped inside the jar | **Yes, once** (torch → ONNX export), unless you use pre-exported ONNX | CPU EP ~0.3–0.6×; CoreML EP potentially ~0.5–1.5× *if* the graph doesn't fragment (**guess**) | **Low–medium** (export plus STFT/chunking glue in Java) | Fastest path to working. Maven 1.32.0 bundles CoreML. Verify op coverage per model. |
| **DJL + libtorch (CPU/MPS)** | Python-free; libtorch JNI (~200 MB native) | **Yes, once** (TorchScript export) | ~0.8–1× (same kernels as today; libtorch 2.7.1) | Low–medium | Closest numerics to the current Python output. Heavy, lags PyTorch versions. Best as oracle. |
| Babylon / HAT | N/A in 2026 (no Metal, not in JDK) | — | — | — | Watch only. |

Suggested path: (1) get a working, accurate baseline with ONNX Runtime (or DJL as the numerical oracle); (2) in parallel,
build the pure-Java op set with Vector API on CPU, tested against the oracle; (3) swap hot ops (GEMM/conv) to
FFM→Accelerate; (4) evaluate GPU via TornadoVM-Metal or MLX if (3) is too slow.

---

## 3. Reading PyTorch checkpoints from Java

Format (PyTorch ≥ 1.6 `torch.save`): a **zip** (stored, not compressed) containing `<root>/data.pkl` (pickle protocol 2),
`<root>/data/<key>` raw little-endian storages, `<root>/byteorder`, `<root>/version`, and sometimes `.data/serialization_id`.
In the pickle, tensors are `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, requires_grad, backward_hooks)`,
where `storage` comes from a **persistent id** tuple `('storage', torch.FloatStorage|<dtype>, key, location, numel)`.
Legacy pre-1.6 files are a non-zip pickle stream with a different layout. Docs:
https://docs.pytorch.org/docs/stable/notes/serialization.html

Existing pieces:
- **`net.razorvine:pickle` 1.5** (Irmen de Jong's Pyrolite, MIT, https://github.com/irmen/pickle): a full unpickler
  (protocols 0–5) for Java. Unknown classes come back as `ClassDict` maps, you can `Unpickler.registerConstructor(module,
  name, IObjectConstructor)`, and you can override `persistentLoad(Object pid)`. This is exactly what's needed:
  register `torch._utils._rebuild_tensor_v2`, `torch._utils._rebuild_parameter`, `collections.OrderedDict`,
  `torch.*Storage` / `torch.storage.UntypedStorage`, and make `persistentLoad` return a lazy view of the zip entry.
  Estimated ~300–500 LOC. TorchSharp.PyBridge does the same in .NET, and the Rust crates `burn-store` (pytorch reader) and
  `pt-loader` are good design references.
- **Demucs `.th` files** are `torch.save`d *packages* (dict with `klass`, `args`, `kwargs`, `state`, …) that reference
  Demucs Python classes. With Pyrolite these just become `ClassDict`s, so you read `state` and the `kwargs` hyperparameters.
  Some Demucs checkpoints store **fp16-quantized** state (`quantizer` / `half`): check per file. Lightning `.ckpt` files
  (Roformer) keep weights under `state_dict` with a `model.` prefix. Both points are **(unverified per file)**; the model
  agents should confirm.
- **No maintained Java "torch.load" library** found. DJL doesn't read raw state_dicts (TorchScript only). Jlama reads only
  safetensors.
- **Safetensors** (https://github.com/huggingface/safetensors): 8-byte LE header length, JSON header `{name: {dtype, shape,
  data_offsets}}`, then raw bytes. A Java reader is ~100 LOC: `FileChannel.map` into a `MemorySegment` (FFM) for zero-copy
  mmap, and also no 2 GB limit, unlike `MappedByteBuffer`. Java implementations exist (Jlama's `SafeTensorSupport`, small
  repos like `tmanabe/safetensors-java`), but writing your own is simpler than depending on them.
- **Recommendation:** a one-time Java converter `.ckpt/.th → .safetensors` (plus a small JSON of hyperparameters), with
  safetensors as the runtime format. Pickle is code execution in Python, but a Java Pyrolite reader with only
  whitelisted constructors is safe (it never runs arbitrary code).

---

## 4. Audio DSP building blocks on the JVM

| Need | Options | Notes |
|---|---|---|
| FFT / STFT | **JTransforms 3.2** (`com.github.wendykierp:JTransforms`, BSD-2-Clause, pure Java, multithreaded; last push 2025-12, Maven release 3.2). Or a hand-written radix-2/4 real FFT with the Vector API. Or FFM→**vDSP** (Accelerate) on macOS. | Demucs n_fft 4096 / hop 1024; Roformer n_fft 2048 / hop 512 (other agents to confirm). Fixed power-of-two sizes make a custom FFT easy. Must match `torch.stft(center=True, pad_mode='reflect', window=hann(periodic), normalized=…)` and `torch.istft` overlap-add plus window-sum normalization exactly. Write golden tests against numbers dumped from Python once. JTransforms' "faster than FFTW" claim is the authors' own and **unverified**. |
| IIR / crossover | Port of `scipy.signal.butter(4, 250, 'low', fs=44100, output='sos')` plus `sosfiltfilt`. | Coefficients are fixed for 44.1 kHz/250 Hz and could be hardcoded, or computed via bilinear transform (~60 LOC). `sosfiltfilt` = odd-extension padding (`padlen = 3*(2*len(sos) + 1 - min((sos[:,2]==0).sum(), (sos[:,5]==0).sum()))`), steady-state initial conditions (`sosfilt_zi`), forward pass, reverse, backward pass, then trim. ~150 LOC in Transposed-DF-II biquads with `double` precision. Zero-phase means **non-causal over the whole signal**, so either run it on the full buffer (simple, fine at song length: ~4 min × 44.1 kHz × 2 ch ≈ 21M samples ≈ 85 MB as float) or replace it with a linear-phase FIR / Linkwitz-Riley crossover if true streaming matters. |
| WAV I/O | `javax.sound.sampled` (`AudioSystem.getAudioInputStream`/`write`, PCM 16/24-bit; PCM_FLOAT read support is decent, float write **unverified**). Or hand-roll RIFF (~150 LOC, recommended: handles 32-bit float, `WAVE_FORMAT_EXTENSIBLE`, >2 GB via RF64 if needed). | In a streaming design WAV mostly disappears: ffmpeg can emit raw `f32le` PCM to stdout and accept it on stdin. |
| Resampling | Not needed if ffmpeg outputs 44.1 kHz (it already does: `-ar 44100`). If ever needed: Concentus ships a Java port of the speexdsp resampler, or write a windowed-sinc polyphase resampler. | Models are trained at 44.1 kHz. |
| Opus / AAC decode | **Concentus** (pure Java/C#/Go port of libopus 1.1.2, bit-exact, BSD-style license; active, push 2026-11-17; Maven `io.github.jaredmdobson:concentus` 1.0.2) runs "about 40–50% as fast as libopus". You'd also need a **WebM/Matroska demuxer** (EBML: small, writable) or MP4 demux for itag 140 (`mp4parser`, Apache-2.0, stale since 2024) plus an AAC decoder (JAAD: old, incomplete). https://github.com/lostromb/concentus | Viable for Opus (itag 251) only. Decoding a 4-min track even at 0.5× libopus speed takes well under a second, so speed isn't an issue. Correctness and the long tail are. |
| MP3 encode | **jump3r** (pure-Java LAME 3.98.4 port, LGPL-2.1+, Codeberg fork `RaphiMC/jump3r`, release 1.0.6, commits Oct 2026). `java-lame` (also 3.98.4, stale). Wrappers that are *not* pure Java: `lame4j` (Rust/JNI), `lamejb` (JNA). | LAME 3.98.4 is older than ffmpeg's libmp3lame 3.100, and V2 VBR quality should be close. LGPL in a jar is OK if dynamically replaceable. Speed probably a few times slower than native: **unverified**. |
| MP3 decode | JLayer / mp3spi (umjammer forks, LGPL-2.1, active 2026). | Not needed by splitter (YouTube gives Opus/AAC). |

**Verdict on codecs: keep ffmpeg.** It isn't Python, it's already a hard dependency, it handles every YouTube container and
codec variant, and in a streaming design it's just two `Flow.Processor`s wrapping a process
(`bytes → f32 PCM` and `f32 PCM → MP3 bytes`). Revisit pure-JVM only if a single self-contained jar becomes a goal, in
which case Concentus + EBML demuxer + jump3r is the path.

---

## 5. Reactive Streams packaging

### 5.1 Interop surface

- Target **`java.util.concurrent.Flow`** (JDK 9+). It is semantically identical to `org.reactivestreams`
  (spec 1.0.4), and `org.reactivestreams:reactive-streams:1.0.4` ships `FlowAdapters` to convert both ways.
- **Pekko 2.0** (splitter's version) supports both. JDK Flow interop lives in
  `org.apache.pekko.stream.scaladsl.JavaFlowSupport` / `javadsl.JavaFlowSupport`: `Source.fromPublisher`,
  `Source.asSubscriber`, `Sink.asPublisher(fanout)`, `Sink.fromSubscriber`, `Flow.fromProcessor(() => proc)`,
  `Flow.toProcessor` (non-fanout publishers reject a 2nd subscriber with `IllegalStateException`).
  https://pekko.apache.org/docs/pekko/2.0/stream/reactive-streams-interop.html . Open PR apache/pekko#2460 proposes moving
  Flow support into the main `Source`/`Sink` (not merged).
- **Reactor** 3.9.0: `JdkFlowAdapter.publisherToFlowPublisher` / `flowPublisherToFlux`.
- **RxJava 4.x** (alpha-22, 2026-11-19) is **natively `java.util.concurrent.Flow`-based** on Java 26, with virtual-thread
  operators. RxJava 3 uses `org.reactivestreams` plus `FlowAdapters`.
- **Mutiny**, Helidon, JDK `HttpClient` (`BodyPublishers.fromPublisher`, `BodySubscribers.fromSubscriber`) all speak `Flow`
  directly. That's handy for the downloader: the JDK `HttpClient` can stream response bytes into a `Flow.Subscriber`.

### 5.2 Compliance

- **TCK:** `org.reactivestreams:reactive-streams-tck-flow:1.0.4` (TestNG-based) provides
  `FlowPublisherVerification<T>`, `FlowSubscriberWhiteboxVerification<T>`/`Blackbox`, and
  `IdentityFlowProcessorVerification<T>`. Run them in CI for every component (they test the hard rules: §1.3 serial
  signalling, §2.13 no nulls, §3.9 `request(n<=0)` gives `onError`, demand overflow to `Long.MAX_VALUE` = unbounded,
  cancellation idempotence, no stack-overflow on synchronous recursion via §3.3).
- **Don't hand-roll the state machines** if you can avoid it: implementing a compliant `Publisher` is subtle. Options:
  - JDK `SubmissionPublisher` (TCK-compliant, buffer-backed, `Executor`-driven). Good for sources (download, decoder output).
  - Build each processor internally on a small, well-tested demand-tracking core (an `AtomicLong` demand plus a
    single-consumer drain loop, the "trampolined drain" pattern from RxJava/Reactor), and share it across all components.
  - Or implement components in Pekko and export them via `Flow.toProcessor`. That's simplest for splitter, but it ties the
    library to Pekko, which defeats the "usable by Reactor/RxJava" goal.

### 5.3 Design notes for chunked / streaming inference

- **Element type:** an immutable `AudioChunk(long startFrame, int channels, int frames, float[] or MemorySegment pcm,
  int sampleRate)`, planar or interleaved (pick one; models want planar `[C, T]`). Backpressure granularity = **chunks**:
  `request(1)` means "one more chunk". Keep chunks around **model segment size** (HTDemucs ~7.8 s; Roformer chunk ~8 s
  at 44.1 kHz; confirm with the model notes) so each `onNext` is one inference call. Smaller transport chunks (e.g. 4096
  frames from the decoder) should be **re-chunked** by a dedicated `Rechunker` processor.
- **Overlap-add:** both separators run on overlapping windows (e.g. 25% overlap with a triangular/Hann crossfade in
  Demucs `apply_model`; Roformer `num_overlap`). Model this as `Rechunker(segment, hop)` → `Infer` → `OverlapAdd(hop,
  window)`. OverlapAdd holds `segment-hop` frames of state and emits finalized frames only. Latency = one segment. Demand
  maps 1:1 through Infer. OverlapAdd may need `request(2)` upstream to emit the first chunk.
- **Normalization** (Demucs normalizes the whole mix by global mean/std; audio-separator may normalize peak) makes the
  op **non-streaming** unless you compute stats on a first pass or accept a running estimate. This is a choice to make
  explicitly. The same applies to zero-phase `sosfiltfilt` (§4). For a 4-minute song, "materialize the whole track, then
  stream through the models" is perfectly acceptable. The Flow API then just provides a uniform pipeline, progress, and
  cancellation.
- **Bounded buffers and memory:** stem outputs are as large as the input (×4–6 stems for multi-stem models, though we only
  keep bass). Drop unneeded stems inside the processor so they never cross the boundary. Use off-heap `MemorySegment`s from
  a pooled `Arena` if GC pressure shows up. Document whether a chunk can be reused after `onNext` returns (simplest: no
  reuse, hand ownership over).
- **Threading:** inference is blocking and long (hundreds of ms to seconds per chunk). Run it on a dedicated executor
  (one thread per GPU/accelerator context; MLX/Metal contexts are thread-affine, see the jmlx notes). Never inside
  `onNext` on the caller's thread: enqueue and drain from the worker. Pekko users can also wrap with `.async` /
  `mapAsync(1)`.
- **Cancellation:** `cancel()` must stop the worker between chunks and release native resources (ORT `OrtSession`,
  Metal buffers). Errors from native code go through `onError`, never thrown from `onNext`.
- **Progress:** derive it from `startFrame / totalFrames` on emitted chunks. No side channel needed.
- **Two-model fan-out (splitter's crossover):** mix → broadcast → [Roformer → bass → lowpass] and [Demucs → bass → highpass]
  → zip by `startFrame` → sum. A `Flow.Publisher` with fanout, or do the broadcast/zip in Pekko (`Broadcast`/`Zip`) and keep
  library components single-subscriber. Simpler, and recommended.
- **External processes as Flow components** (yt-dlp, ffmpeg): a `ProcessPublisher` (stdout → `ByteBuffer` chunks, demand
  = reads) and `ProcessProcessor` (stdin writer + stdout publisher). OS pipe buffers provide the natural backpressure.
  splitter's existing `ProcessStreams.flow` is effectively this in Pekko.

---

## Uncertainties / things to verify next

1. ORT 1.32.0 Java on macOS: `OrtEnvironment.getAvailableProviders()` output (is WebGPU listed?), and CoreML EP partition
   count for each exported model (`session.logSeverityLevel=0`, or the ORT profiling JSON).
2. Pre-exported ONNX models for htdemucs_ft / BS-Roformer-SW: fidelity vs PyTorch (SDR on a test clip).
3. TornadoVM 8.1 on JDK 27 plus Metal on M1 Max: run its GEMM benchmark.
4. Real FLOPs per segment for both models (from the architecture notes), to turn the speed "guesses" in §2.2 into numbers.
5. Checkpoint specifics: legacy vs zip format, fp16-quantized Demucs states, Lightning key prefixes.
6. NewPipeExtractor on a desktop JVM for an audio-only download today (quick spike, ~50 LOC with OkHttp `Downloader`).

## Sources (accessed 2026-11-20)

- yt-dlp: local clone `sources/yt-dlp` (2026.10.26); https://github.com/yt-dlp/yt-dlp/releases (2026.10.26 assets incl. `yt-dlp_macos`);
  https://github.com/yt-dlp/yt-dlp/issues/15012 ; https://github.com/yt-dlp/ejs ; https://pypi.org/project/yt-dlp-ejs/ ;
  `brew info yt-dlp` (deps: python@3.15, deno, curl-impersonate).
- NewPipeExtractor: https://github.com/TeamNewPipe/NewPipeExtractor (releases v0.25.0–v0.27.1, `gradle/libs.versions.toml`,
  `utils/JavaScript.java`, PR #1508, v0.27.0 notes); https://newpipe.net/blog/ ;
  https://opencollective.com/spotube/updates/v5-1-1-release-notes ; https://mailman.videolan.org/pipermail/vlc-commits/2026-May/073117.html
- ONNX Runtime: Maven Central `com.microsoft.onnxruntime:onnxruntime:1.32.0` (jar inspected with `otool`/`nm`);
  https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html ;
  https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.SessionOptions.html
- DJL: https://github.com/deepjavalibrary/djl (v0.40.0, `PtDeviceType.java`, `gradle/libs.versions.toml`);
  https://docs.djl.ai/master/engines/pytorch/pytorch-engine/index.html
- JDK: https://openjdk.org/jeps/537 ; https://openjdk.org/jeps/0 (JEP index: 8397768 Vector API 13th incubator Submitted;
  8361105 Code Reflection Submitted); https://blogs.oracle.com/java/the-arrival-of-java-27 ;
  https://www.infoq.com/news/2026/09/java27-released/
- TornadoVM: https://github.com/beehive-lab/TornadoVM (v8.0.0, v8.1.0, CHANGELOG); https://infoq.com/news/2026/10/tornadovm-7-gpu
- Babylon/HAT: https://openjdk.org/projects/babylon ; https://github.com/openjdk/babylon/tree/code-reflection/hat/backends/ffi ;
  https://mail.openjdk.org/pipermail/babylon-dev/2026-February/005110.html ;
  https://mail.openjdk.org/pipermail/babylon-dev/2025-October/003841.html
- LLM-in-Java: https://github.com/mukel/llama3.java ; https://github.com/tjake/Jlama ; https://github.com/beehive-lab/GPULlama3.java ;
  https://github.com/Alipsa/jmlx
- Checkpoints: https://docs.pytorch.org/docs/stable/notes/serialization.html ; https://github.com/irmen/pickle ;
  https://docs.rs/burn-store/0.21.0/src/burn_store/pytorch/reader.rs.html ; https://github.com/huggingface/safetensors
- DSP/codecs: https://github.com/wendykierp/JTransforms ; https://github.com/lostromb/concentus ; https://codeberg.org/RaphiMC/jump3r ;
  https://github.com/umjammer/jlayer ; https://github.com/sannies/mp4parser
- Reactive Streams: https://pekko.apache.org/docs/pekko/2.0/stream/reactive-streams-interop.html ; https://github.com/apache/pekko/pull/2460 ;
  https://github.com/reactive-streams/reactive-streams-jvm (1.0.4, TCK-flow) ; https://github.com/ReactiveX/RxJava (4.x README)
