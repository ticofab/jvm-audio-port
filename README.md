# jvm-audio-port

Research and planning for replacing the Python tools that [splitter](../splitter) depends on with native JVM
libraries, exposed as Reactive Streams (`java.util.concurrent.Flow`) components so they plug into Pekko Streams,
Reactor, RxJava and friends. Goal: no Python environment to set up, ever.

## What splitter needs Python for

Splitter runs one Python virtualenv per run, with two top-level packages: `audio-separator` 0.48.0 and `yt-dlp`.
Everything below comes from those two (plus their dependencies).

| # | Need | Python today | Pulled in by | Role in splitter |
|---|------|--------------|--------------|------------------|
| 1 | **YouTube download** | `yt-dlp` | direct | Fetch title/duration, stream the best audio track to stdout |
| 2 | **Separation model A** | BS-Roformer-SW (6 stems), PyTorch | `audio-separator` | Bass below 250 Hz (keeps fills) |
| 3 | **Separation model B** | Demucs v4 `htdemucs_ft`, PyTorch | `audio-separator` (vendored Demucs code) | Bass above 250 Hz (clean mids) |
| 4 | **Tensor runtime + GPU** | PyTorch (MPS on Apple Silicon, CUDA, CPU) | `audio-separator` | Runs both models |
| 5 | **Separation orchestration** | `audio-separator` itself | direct | Model download/caching, chunking with overlap-add, normalization, stem writing |
| 6 | **Weights format** | `torch.load` (pickle in zip) | PyTorch | Loads `.ckpt` / `.th` checkpoints |
| 7 | **STFT / iSTFT** | `torch.stft` / `torch.istft` | PyTorch | Both models work on spectrograms |
| 8 | **Crossover filter** | `scipy.signal` `butter` + `sosfiltfilt` | splitter's `separate.py` | 4th-order zero-phase Butterworth at 250 Hz |
| 9 | **Audio file I/O** | `soundfile` (libsndfile), `librosa`, `pydub` | `audio-separator`, `separate.py` | Read the mix, write stems (WAV) |

Not Python, and out of scope for now: `ffmpeg` (decoding YouTube's Opus/AAC to PCM, encoding the MP3) and `uv` (only
exists to build the Python environment, so it disappears with Python).

## Sizes (lines of Python, at the versions cloned in `sources/`)

| Codebase | Lines | Of which relevant to splitter |
|----------|------:|-------------------------------|
| yt-dlp | 226,070 | YouTube extractor: 11,725 (plus core networking, format selection, JS challenge solving) |
| audio-separator 0.48.0 | 20,585 | Roformer code: 3,882; vendored Demucs: 5,740; plus the MDXC/Demucs separator classes |
| Demucs (upstream) | 7,256 | `htdemucs.py`, `transformer.py`, `spec.py`, `apply.py` |
| BS-RoFormer (lucidrains, original) | 2,220 | `bs_roformer.py`, `attend.py` |

Line counts overstate the port: most of it is training code, other architectures and other websites. The analysis
notes size the inference paths we actually run.

## Early findings

- **htdemucs_ft only needs one of its four networks for bass.** Its bag weights (`~/.cache/splitter/models/htdemucs_ft.yaml`)
  are an identity matrix over (drums, bass, other, vocals), so only sub-model `d12395a8` contributes to bass.
  audio-separator runs all four.
- **BS-Roformer-SW computes all six stems in one pass;** bass-only output doesn't save much compute.
- **Prior art exists for Demucs outside Python:** `demucs.cpp` (C++/Eigen) and `demucs.onnx` (ONNX with STFT outside
  the graph), both by sevagh.

## Layout

- `demucs-jvm/` — the library: pure-Java Demucs and BS-Roformer inference, the crossover, the `Accelerator` SPI, and
  `java.util.concurrent.Flow` components. See its README.
- `demucs-jvm-metal/` — the optional Apple GPU backend (Metal, through FFM).
- `notes/` — analysis per component: `htdemucs.md`, `bs-roformer.md`, `jvm-landscape.md` (JVM runtimes, weights,
  DSP, Reactive Streams, yt-dlp).
- `PLAN.md` — the porting plan, written from the notes.
- `sources/` — not kept: the code we studied, which the notes cite by path and line. Restore it with the commands
  below.

Build: `mvn install` in `demucs-jvm`, then in `demucs-jvm-metal`; splitter picks both up from the local Maven
repository. The libraries are not published.

## Status

Done: both separation models and the crossover run on the JVM, on the Apple GPU or the CPU, matching PyTorch/scipy
to float rounding; splitter uses them and no longer needs Python. yt-dlp was deliberately not ported (see
`notes/jvm-landscape.md` and splitter's `HISTORY.md`): splitter fetches its standalone build instead.

## Sources studied

The notes refer to these repositories at these commits:

| Folder | Repository | Commit |
|--------|------------|--------|
| `audio-separator` | [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | `4dd3477` (tag v0.48.0) |
| `demucs` | [adefossez/demucs](https://github.com/adefossez/demucs) | `829d218` |
| `bs-roformer` | [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) | `4b95192` |
| `msst` | [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | `1f700c3` |
| `demucs.cpp` | [sevagh/demucs.cpp](https://github.com/sevagh/demucs.cpp) | `4edae49` |
| `demucs.onnx` | [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) | `81fa192` |
| `yt-dlp` | [yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) | `4adbf69` |

To restore them into `sources/` (ignored by git):

```sh
mkdir -p sources && cd sources
while read dir repo commit; do
  git clone -q "https://github.com/$repo.git" "$dir" && git -C "$dir" checkout -q "$commit"
done <<'LIST'
audio-separator nomadkaraoke/python-audio-separator 4dd3477
demucs adefossez/demucs 829d218
bs-roformer lucidrains/BS-RoFormer 4b95192
msst ZFTurbo/Music-Source-Separation-Training 1f700c3
demucs.cpp sevagh/demucs.cpp 4edae49
demucs.onnx sevagh/demucs.onnx 81fa192
yt-dlp yt-dlp/yt-dlp 4adbf69
LIST
```
