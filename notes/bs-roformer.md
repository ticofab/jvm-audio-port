# BS-Roformer-SW: porting inference to the JVM

Research only. Scope: what it takes to run BS-Roformer-SW (6 stems, we use bass) natively on the JVM without
Python, matching what `audio-separator` 0.48.0 does today.

Path abbreviations used below:
- `AS` = `sources/audio-separator/audio_separator/separator`
- `MSST` = `sources/msst`
- `LR` = `sources/bs-roformer` (lucidrains upstream)
- `ROT` = `~/.cache/uv/archive-v0/9v4YiitWO-fN3ViZ/rotary_embedding_torch/rotary_embedding_torch.py` (v0.6.5, the
  installed dependency that audio-separator actually calls)

## TL;DR

- **Which implementation the checkpoint matches:** `AS/uvr_lib_v5/roformer/bs_roformer.py` (`BSRoformer`). It is the
  same as `MSST/models/bs_roformer/bs_roformer.py` with `skip_connection=False`, except that audio-separator does **not**
  apply `zero_dc`. Upstream lucidrains HEAD (v1.4.1) does **not** match: it adds hyper-connections and value-residual
  modules. The checkpoint loads with a strict `load_state_dict` (`AS/roformer/roformer_loader.py:119-125`).
- **Size:** 174.66 M fp32 parameters (699 MB). About 85% of that, 148.3 M, is in the 6 mask-estimator heads. A bass-only
  port needs **51.05 M params (204 MB)**: band split 1.07 M, 12 axial layers 25.26 M, bass head 24.72 M.
- **Compute:** one chunk (409,600 samples, 9.29 s) is **3.6 TFLOP** for bass only (3.8 TFLOP for all 6 stems). With
  overlap 2 that is about **0.8 TFLOP per second of audio**, so roughly **220–230 TFLOP for a 4:44 song (61 chunks)**.
  Your measured ~2 min on MPS works out to about 1.9 TFLOP/s sustained, which is plausible for an M1 Max GPU in fp32.
- **Ops are simple:** Linear, RMSNorm (L2-normalize), GELU (erf), Tanh, GLU, sigmoid gating, interleaved RoPE,
  non-causal softmax attention, STFT/iSTFT (Hann 2048/512, centered), and complex multiply. There are no convolutions,
  no recurrence and no data-dependent control flow.
- **Feasibility:**
  - Pure Java with the Vector API is feasible but slow: about **10–20 min per song** on the 8 P-cores (NEON is 128-bit).
  - Java calling Apple Accelerate `cblas_sgemm` (AMX) through the FFM API could get close to MPS speed.
  - ONNX export of the network core (STFT and iSTFT kept outside) is feasible, and ONNX Runtime Java is a ready-made
    route, but it is JNI-native rather than pure JVM.
- **Effort:**
  - Correct pure-Java implementation: about 3–4k LOC including tests, **3–5 weeks** for one experienced developer.
  - Fast GEMM/attention: another 2–4 weeks.
  - Riskiest parts: GEMM performance, and matching the exact STFT conventions and the (f,s,c) interleaving bit-for-bit.

---

## 1. Exact inference path (audio-separator 0.48.0, our invocation)

Our call site is `splitter/code/src/main/resources/separate.py:27-35`: `Separator(output_format="WAV",
output_single_stem="Bass")` with defaults, then `load_model("BS-Roformer-SW.ckpt")`. Only the content **below 250 Hz** of
the BS-Roformer bass stem is used. It goes through a 4th-order Butterworth applied with `sosfiltfilt`, crossed over with
htdemucs_ft (`separate.py:39-47`).

Precision is fp32 by default: `use_autocast=False`, `use_native_fp16=False`, `use_torch_compile=False`
(`AS/separator.py:136,147,148`; `AS/architectures/mdxc_separator.py:208-234`).

### 1.1 Loading and normalization
1. `prepare_mix` loads with `librosa.load(path, mono=False, sr=44100)` (`AS/common_separator.py:284`). This gives
   float32 audio; mono input is duplicated to stereo.
2. **Whole-song peak normalization of the input:** `spec_utils.normalize(mix, max_peak=0.9, min_peak=0.0)`
   (`AS/architectures/mdxc_separator.py:335`, `AS/uvr_lib_v5/spec_utils.py:100-123`, default
   `normalization_threshold=0.9` at `AS/separator.py:130`). If peak > 0.9, the whole mix is scaled to a peak of 0.9.
   - Why this barely matters for streaming: the first op of the network is a per-band, per-frame L2 normalization
     (BandSplit RMSNorm, see 1.4), so the network is scale-invariant apart from the `eps=1e-12` floor. The mask
     multiplies the input STFT, so the output scales linearly with the input. A fixed gain, or no gain, gives the same
     result to float precision.
3. **Output peak normalization:** each written stem is again peak-limited to 0.9 over the whole song
   (`mdxc_separator.py:361-385`). This is also whole-song, but it only kicks in if the bass stem peaks above 0.9.
4. Output is WAV with the input's subtype/bit depth, default 16-bit (`AS/common_separator.py:358,441-448`).

### 1.2 Chunking and overlap-add (`mdxc_separator.py:489-567`)
- **The chunk size is NOT the YAML `audio.chunk_size: 588800`.** audio-separator computes
  `chunk_size = stft_hop_length * (inference.dim_t - 1) = 512 * 800 = 409,600` samples (9.288 s)
  (`mdxc_separator.py:503-515`).
  - Exception: for inputs shorter than 10 s it switches to `segment_size=256`, i.e. 130,560 samples
    (`:416-422`, `:105`).
  - By contrast, MSST's own `demix` uses 588,800 with a linear-fade window and reflect border padding
    (`MSST/utils/model_utils.py:175-272`). Our outputs follow the audio-separator scheme, not MSST's.
- `step = chunk_size // num_overlap = 409600 // 2 = 204,800` (`:521`).
- **Schedule** (`_roformer_chunk_starts`, `:452-469`): starts at 0, step, 2·step, …. The first start with
  `offset + chunk >= L` is replaced by `max(L - chunk, 0)`, so the last chunk is right-aligned to the end of the song.
  Every chunk is full-length when `L >= chunk`, and there is no padding of the song.
  - A 4:44 song (12,524,400 samples) gives **61 chunks**.
- **Window:** `scipy.signal.windows.hamming(409600)`, which is the symmetric variant:
  `0.54 - 0.46 cos(2πn/(M-1))` (`:545`).
- **Accumulation:** `result[..., s:s+len] += out * w` and `counter[..., s:s+len] += w` (`:440-450, :551-563`). At the end,
  `counter.clamp_(min=1e-10)` and `result /= counter` (`:565-566`). This is a weighted average, so edge samples covered
  only by a near-zero window tail are still exact (just noisier).
- The model output length is `512*(T-1)` because there is no `length=` argument on `istft`. For full chunks that equals
  the chunk length. For songs shorter than a chunk it can be up to 511 samples shorter, which `safe_len` handles.
- All 6 stems are always computed: `num_stems = len(instruments)` (`:497`), mapped by order
  `['bass','drums','other','vocals','guitar','piano']` (`:669`). So **bass = mask_estimators.0**.

### 1.3 STFT (`AS/uvr_lib_v5/roformer/bs_roformer.py:474-488`)
- `torch.stft(n_fft=2048, hop_length=512, win_length=2048, normalized=False, window=torch.hann_window(2048),
  return_complex=True)`.
  - torch defaults: `center=True`, `pad_mode='reflect'` (pads 1024 on each side), `onesided=True`, and a **periodic**
    Hann window.
  - For 409,600 samples this gives **T = 801 frames** and **1025 bins**.
- `view_as_real` gives shape `(b, s=2, f=1025, t, c=2)`. Then `rearrange 'b s f t c -> b (f s) t c'` and
  `'b f t c -> b t (f c)'`.
- **Per-frame feature vector, length 4100, ordering:** `[f0·L·re, f0·L·im, f0·R·re, f0·R·im, f1·L·re, …]`, i.e. index
  `((f*2 + s)*2 + c)`. Getting this interleave right is essential.

### 1.4 Band split (`bs_roformer.py:182-202`, `:412-414`)
- `freqs_per_bands` has 62 bands:
  - 24×2, 12×4, 8×12, 8×24, 8×48, then 128 and 129 bins (sum 1025).
  - Each band's input dimension is `2*f*2` (complex × stereo): 8, 16, 48, 96, 192, 512 or 516.
  - Band boundaries are contiguous slices of the 4100-vector above.
- Per band: `RMSNorm(dim_in)` then `Linear(dim_in → 256)` with bias. The results are stacked to **x: (b, T=801, F=62, 256)**.

### 1.5 Axial transformer stack (`bs_roformer.py:498-520`)
There are 12 layers (`linear_transformer_depth=0`, so 2 sub-blocks each).
1. **Time transformer:** `x → (b·62, 801, 256)`. 496 sequences of length **801**, with RoPE over frame index 0..800.
2. **Freq transformer:** `x → (b·801, 62, 256)`. 801 sequences of length **62**, with RoPE over band index 0..61.

Each Transformer has depth 1 and `norm_output=False` (`:383`), and runs `x = attn(x) + x; x = ff(x) + x` (`:170-176`).
- **Attention** (`:87-120`):
  - `RMSNorm(256)`, then `to_qkv` Linear 256→1536 with no bias. Split as `(qkv h d)` with qkv=3, h=8, d=64 (q is the
    first 512 columns, then k, then v; head-major inside each).
  - RoPE on q and k. Then SDPA with scale `64^-0.5`, no mask (non-causal), dropout 0 in eval (`attend.py:94-115,128`).
  - Per-head gate: `sigmoid(Linear(256→8, bias)(x_normed))`, broadcast over d (`:116-117`).
  - Merge heads `(h d)` and apply `to_out` Linear 512→256 with no bias.
- **FeedForward** (`:77-84`): `RMSNorm(256)` → Linear 256→1024 (bias) → **exact GELU (erf)** → Linear 1024→256 (bias).
- **RMSNorm** (`:59-71`): `x / max(||x||₂, 1e-12) * sqrt(dim) * gamma`. This is F.normalize based, with no mean
  subtraction and no epsilon inside a sqrt.
- After the stack: `final_norm` RMSNorm(256) (`:522`).
- **"flash_attn: true" only selects the fused SDPA kernel.** The math is plain `softmax(QKᵀ/8)·V`, so the port does not
  need flash attention for correctness. A tiled/online-softmax implementation is still useful to avoid materialising
  62·8·801² scores (1.27 GB).

### 1.6 Rotary embedding (`AS/uvr_lib_v5/roformer/rotary.py:38-41,78`; `ROT:29-61,101,152`)
- `RotaryEmbedding(dim=64)` with `theta=10000`. `freqs[i] = 10000^(-2i/64)` for i = 0..31. These are stored in the
  checkpoint and checked: `data/0` holds 1.0, 0.7499, 0.5623, …
- Angles are `pos * freqs[i]` with `pos = 0..n-1`, and `repeat_interleave(…, 2)` makes them **interleaved pairs**
  (dims 2i and 2i+1 share angle i).
- `rotate_half` maps `(x0,x1) → (-x1, x0)` per pair. Output: `x*cos + rotate_half(x)*sin` on all 64 dims.
- The time and freq transformers have separate `RotaryEmbedding` instances with identical freqs. The 24 checkpoint
  tensors share 2 storages.

### 1.7 Mask estimation and application (`bs_roformer.py:205-248`, `:524-552`)
- Per stem and per band: `Linear(256→1024, bias) → Tanh → Linear(1024→2·dim_in, bias) → GLU(dim=-1)`. GLU is
  `a * sigmoid(b)` with a and b the first and second halves. Output is `dim_in` values in the same `(f s c)` layout as
  the input. The bands are concatenated to 4100 values per frame.
- The mask is **complex, per bin and per channel**: `stft * mask` is a complex multiply (`:543-546`). There is no
  summation over overlapping bands, because the bands are disjoint.
- iSTFT: `rearrange 'b n (f s) t -> (b n s) f t'`, then `torch.istft` with the same n_fft, hop, window and
  `center=True`, without a `length` argument (`:550-552`). Output length is 512·800 = 409,600.
  - Standard WOLA: divide by the sum of squared windows, then trim 1024 on each side.
- **No `zero_dc` in audio-separator.** MSST zeroes bin 0 before iSTFT (`MSST/models/bs_roformer/bs_roformer.py:647-649`).
  `zero_dc` is accepted but unused in `AS/.../bs_roformer.py:342,369`. Match audio-separator if the goal is
  bit-compatibility with current outputs.

### 1.8 Streaming implications
- The model is **non-causal over a 9.29 s window**: time attention spans all 801 frames. So the minimum latency is about
  one chunk (9.3 s), and with overlap 2 a new 4.64 s of output is produced per chunk.
  - Smaller chunks are possible, e.g. `dim_t=256` gives 2.97 s, which audio-separator itself uses for short files. This
    probably costs quality: the model was trained at 588,800 samples (~13.4 s).
- Whole-song dependencies are only the two peak normalizations and the right-aligned last chunk. All are replaceable:
  scale invariance removes the input one, and a streaming tail can just be zero-padded.
- The splitter's `sosfiltfilt` crossover is zero-phase and whole-file. That is a separate streaming concern.

## 2. Architecture inventory

### 2.1 Checkpoint key structure (verified from `data.pkl`, 1939 tensors)
```
layers.{0..11}.{0=time,1=freq}.layers.0.0.rotary_embed.freqs   (32,)
                              .layers.0.0.norm.gamma           (256,)
                              .layers.0.0.to_qkv.weight        (1536,256)
                              .layers.0.0.to_gates.weight/bias (8,256)/(8,)
                              .layers.0.0.to_out.0.weight      (256,512)
                              .layers.0.1.net.0.gamma          (256,)      RMSNorm
                              .layers.0.1.net.1.weight/bias    (1024,256)/(1024,)
                              .layers.0.1.net.4.weight/bias    (256,1024)/(256,)
final_norm.gamma                                               (256,)
band_split.to_features.{0..61}.0.gamma                         (dim_in,)
band_split.to_features.{0..61}.1.weight/bias                   (256,dim_in)/(256,)
mask_estimators.{0..5}.to_freqs.{0..61}.0.0.weight/bias        (1024,256)/(1024,)
mask_estimators.{0..5}.to_freqs.{0..61}.0.2.weight/bias        (2*dim_in,1024)/(2*dim_in,)
```
- There is no transformer-level output norm, no skip-connection parameters, no hyper-connection or value-residual
  parameters, and no linear-attention blocks.
- The archive prefix is `BS-Rofo-SW-Fixed2/`. This is the variant with the leftover zero "shared bias" removed so it
  loads with the standard MSST/UVR code (see `sources/audio-separator/docs/deton24-audio-separation-info-2026-03-15.md:1623`).

### 2.2 Parameter counts (exact, summed from the checkpoint)
| Module | Params |
|---|---|
| band_split (62 × RMSNorm + Linear) | 1,069,572 |
| 12 × (time + freq) transformers | 25,258,944 (1,052,456 per transformer) |
| final_norm | 256 |
| mask_estimators, each | 24,721,416 |
| mask_estimators, all 6 | 148,328,496 |
| **Total** | **174,657,268** (fp32, 699 MB) |
| **Needed for bass only** | **51,050,188** (204 MB) |
| Bass head restricted to bands 0–23 (bins 0–47, 0–1 kHz) | +6.7 M instead of 24.7 M |

The last row is an optional optimization, given that we only keep content below 250 Hz.

### 2.3 Op list
- Linear (GEMM + bias): qkv, gates, out, ff1, ff2, band-split per band, mask MLP per band.
- RMSNorm (L2 normalize · √d · γ), GELU (erf form), Tanh, Sigmoid (gates, GLU), GLU split.
- RoPE (interleaved), batched QKᵀ, softmax over last dim, ·V.
- Residual adds, plus reshape/transposes: `b t f d ↔ b f t d`, head split/merge, and the (f s c) interleave.
- STFT/iSTFT (real FFT of 2048, reflect pad, Hann periodic, WOLA), and complex multiply.
- Chunk-level overlap-add with a Hamming window and weight normalization.

Java has no `Math.erf`, so GELU needs an erf implementation; a port of a libm erf or a high-precision rational
approximation both work.

## 3. Compute cost

Per chunk, with T=801, F=62, d=256, 8 heads × 64:

| Part | GFLOP per chunk |
|---|---|
| Linear layers per transformer (49,662 tokens × ~1.05 M MAC) | 104.4 |
| Time attention (seq **801**, 496 sequences × 8 heads) | 81.5 |
| Freq attention (seq **62**, 801 sequences × 8 heads) | 6.3 |
| **Per axial layer** | **296.5** |
| **12 layers** | **3,558** |
| Band split | 1.7 |
| Mask estimator, per stem | 39.5 |
| Mask estimators, 6 stems | 237 |
| **Total, bass only** | **3,599** |
| **Total, 6 stems** | **3,796** |

STFT/iSTFT is about 3,200 FFTs of size 2048 per chunk, which is negligible.

- **Per second of audio (overlap 2):** about 0.82 TFLOP.
- **Per 4:44 song:** 61 chunks, about 220 TFLOP (bass only) or 232 TFLOP (6 stems).
- **The measured MPS time** (~120 s) works out to about **1.9 TFLOP/s** sustained. That is about 20% of the M1 Max GPU's
  ~10 TFLOPS fp32 peak, which is plausible for eager PyTorch.
- **Where the compute goes:** about 70% is plain GEMMs with large, regular shapes (M≈49,662, K∈{256, 512, 1024},
  N∈{1536, 1024, 256}), about 27% is time attention, and the rest is small. Dropping the 5 unused stems saves only about
  5%.

Estimated wall-clock per song (bass only, 220 TFLOP) at different sustained rates:

| Sustained rate | Minutes per song |
|---|---|
| 100 GFLOP/s | 37 |
| 200 GFLOP/s | 18 |
| 400 GFLOP/s | 9 |
| 1 TFLOP/s | 3.7 |
| 2 TFLOP/s | 1.8 |

What each option can realistically reach on this machine (Apple M1 Max, 8P+2E, 32 GB):
- **Pure Java with the Vector API** (`FloatVector.SPECIES_128`, 4 lanes on NEON):
  - Theoretical P-core peak is about 8 cores × 3.2 GHz × 32 FLOP/cycle ≈ 800 GFLOP/s.
  - A well-blocked, multithreaded Java GEMM plausibly reaches 200–400 GFLOP/s. That is an estimate, not measured here.
  - Result: **about 9–18 min per song**, or about half that with `num_overlap=1` (at some quality cost at chunk seams).
    Feasible for offline batch use, not interactive.
- **Java + FFM (JDK 22+) calling Accelerate `cblas_sgemm`**, which runs on the AMX units:
  - fp32 SGEMM throughput on M1 Max is commonly reported in the ~1–2 TFLOP/s range. Unverified here; benchmark before
    relying on it.
  - That would be close to MPS parity, with no Python and no extra dependency on macOS. It is not portable beyond macOS.
  - Attention can also be written as batched SGEMMs (801×64·64×801 per head).
- **GPU from the JVM:**
  - Metal through FFM is a large amount of work.
  - ONNX Runtime Java with the CoreML EP is a far smaller effort (section 5), but coverage and speed of the CoreML EP for
    this graph are unverified.

Memory per chunk (fp32), all fine:
- Activations x: 49,662×256, about 51 MB.
- qkv: about 152 MB.
- FF hidden: about 203 MB.
- Attention scores: process per (sequence, head) or tiled; materialising them all would be 1.27 GB.

## 4. Weights loading

The format is a PyTorch zip archive (the `torch.save` new format).
- 1923 entries, all **Stored (no compression)**, with data aligned to 64 bytes (`.storage_alignment`). Byte order is
  `little`, `.format_version` = 1.
- `BS-Rofo-SW-Fixed2/data.pkl` (402 KB) is pickle protocol 2. Its top level is the raw `OrderedDict` state dict, not
  `{'state_dict':…}` or `{'model':…}`.
- It uses only 3 globals:
  - `collections.OrderedDict`
  - `torch._utils._rebuild_tensor_v2`
  - `torch.FloatStorage`
- It uses only these opcodes: PROTO, GLOBAL, MARK, EMPTY_DICT/TUPLE, TUPLE/1/2, BINUNICODE, BININT/1/2, BINPERSID,
  REDUCE, NEWFALSE, BINPUT/LONG_BINPUT, BINGET/LONG_BINGET, SETITEM(S), BUILD, STOP.
- Each tensor is `_rebuild_tensor_v2((FloatStorage, key, 'cpu', numel), offset=0, shape, stride, requires_grad=False,
  OrderedDict())`. Every tensor is fp32, contiguous, and has offset 0.
- 1939 tensors map to 1917 storage files `data/<key>` (the rotary freqs are shared).

What a Java reader needs:
- **Direct route:** `java.util.zip.ZipFile`, plus a ~200–300 LOC minimal pickle interpreter for the opcodes above. Tensor
  data can be memory-mapped straight from the zip, since entries are stored and aligned: compute each entry's data offset
  from the local header. Existing JVM pickle libraries (e.g. Pyrolite) can parse the opcodes but need custom handling for
  BINPERSID and `_rebuild_tensor_v2`.
- **One-time conversion (recommended):**
  - Convert to **safetensors** with a throwaway script in the existing audio-separator venv, run at dev time only:
    `torch.load(weights_only=True)` → `safetensors.torch.save_file`.
  - Safetensors is an 8-byte little-endian header length, a JSON header (name → dtype, shape, offsets), and raw
    little-endian data. A reader is about 100 LOC with Jackson or a small JSON parser, plus `FileChannel.map`.
  - Optionally drop `mask_estimators.1-5` during conversion (204 MB file instead of 699 MB), and/or pre-transpose weights
    into the GEMM-friendly layout.

## 5. ONNX export feasibility

- **No ONNX export code or notes for roformers exist in the cloned repos.**
  - The only export code is for an unrelated Mamba model (`MSST/models/ex_bi_mamba2.py:275-301`).
  - The MSST README points to the external `ZFTurbo/MSS_ONNX_TensorRT` repo (`MSST/README.md:83`). It was not cloned and
    is not verified here.
  - The audio-separator docs mention an OpenVINO conversion branch that covers Mel-Roformers
    (`docs/deton24-audio-separation-info-2026-03-15.md:1813`).
- **Known blockers and workarounds:**
  - **STFT/iSTFT and complex tensors:** `torch.stft(return_complex=True)`, `view_as_real/complex`, complex multiply and
    `torch.istft` do not export cleanly. ONNX has an `STFT` op from opset 17 but no inverse STFT and no complex dtype.
    The standard fix is to export only the **real-valued core**, `(T, 4100) stft features → (T, 4100) mask` (band split →
    12 layers → final norm → bass mask head), and do STFT, complex multiply and iSTFT on the JVM. The core contains no
    complex ops.
  - **RoPE:** exports as elementwise cos/sin/mul/add; `repeat_interleave` and `rotate_half` become reshape, stack and
    neg.
    - audio-separator's `rotary.py` has a mutable cache and a `torch.compiler.is_compiling()` branch. Tracing at a fixed
      T=801 / F=62 bakes the angle tables as constants, which is fine for fixed-size chunks. Alternatively, export with
      precomputed cos/sin inputs.
  - **SDPA:** decomposes to MatMul/Softmax (opset ≥14). It is non-causal with no mask.
  - **einops `rearrange`/`pack`:** traces to Reshape/Transpose at fixed shapes. `beartype` decorators are harmless.
  - **Size:** the full model is under the 2 GB protobuf limit (699 MB). The bass-only core is about 204 MB.
- **JVM runtime for ONNX:**
  - **ONNX Runtime Java** (`com.microsoft.onnxruntime`) is JNI to native code. That means no Python, but not pure JVM.
    It offers CPU (MLAS, fast) and a CoreML EP on macOS (GPU/ANE; op coverage for this graph unverified).
  - Pure-JVM ONNX interpreters exist but would be slower than a hand-written port.
  - This route is the **quickest way to a working, reasonably fast JVM pipeline** (about 1–2 weeks). It is a good
    reference and a fallback even if a native port is pursued.

## 6. Effort estimate, risks, validation

### 6.1 Components and LOC (Java or Scala, pure-JVM CPU path)
| Component | LOC (approx.) |
|---|---|
| Weights reader (safetensors, plus optional .ckpt pickle reader +250) | 150–400 |
| Tensor buffers and layout helpers (float[] / MemorySegment, strides, transposes) | 300 |
| GEMM kernel (Vector API, cache-blocked, multithreaded via ForkJoin) | 400–800 |
| Attention (RoPE, per-head QKᵀ, online softmax, ·V, gating) | 250 |
| Layers: RMSNorm, GELU (erf), Tanh, GLU, FF, band split, mask head | 300 |
| STFT/iSTFT (JTransforms or a hand-written radix-2 real FFT; reflect pad; WOLA) | 200–300 |
| Chunker, Hamming overlap-add, normalization, WAV I/O | 250 |
| Tests and golden-file comparison harness | 600–900 |
| **Total** | **≈ 2.5–4k LOC** |

### 6.2 Timeline (one experienced developer)
- **Week 1:** weight conversion, weights reader, STFT/iSTFT matched to torch, band split. Golden tests at each stage.
- **Weeks 2–3:** transformer stack, mask head, iSTFT, chunking. End-to-end parity with PyTorch CPU fp32 on a full song.
- **Weeks 3–5:** performance work: blocked and multithreaded GEMM, tiled attention, buffer reuse, and possibly the
  Accelerate-via-FFM backend.
- **Add 2–4 weeks** for a GPU backend (Metal via FFM) or for aggressive CPU tuning / fp16.
- **Alternative:** ONNX export of the core plus ONNX Runtime Java with JVM-side STFT is about 1–2 weeks.

### 6.3 Riskiest parts
1. **Performance:** about 220 TFLOP per song is a lot for a JVM CPU path. GEMM efficiency decides whether the run takes
   10 min or 40 min. Vector API maturity on aarch64 (it is still an incubator module), JIT warm-up, and the lack of
   fp16/bf16 compute are the constraints.
2. **Layout and convention bugs** that give plausible-sounding but wrong output:
   - the `(f s c)` interleave and band slicing;
   - the qkv split order `(qkv h d)`;
   - interleaved rather than half-split RoPE;
   - RMSNorm being an L2-normalize with `max(norm, 1e-12)`;
   - exact GELU;
   - GLU half order (`a * sigmoid(b)`, where a is the first half).
3. **STFT parity:** periodic Hann, `center=True` with reflect padding, onesided output, unnormalized, and iSTFT WOLA
   normalisation and trimming. Reflect padding on chunks shorter than 1025 samples is an edge case.
4. **Chunk schedule details:** right-aligned last chunk, symmetric Hamming, weight-normalized average, short-file
   (< 10 s) segment-size switch.

### 6.4 Validation plan
- **Generate golden tensors** once in the existing audio-separator venv (dev-time Python only). Run on **CPU fp32**, not
  MPS, to get a deterministic reference.
  - Use forward hooks on: `stft_repr`, `band_split` output, the output of each of the 24 transformers, `final_norm`, the
    bass mask, and the reconstructed chunk audio.
  - Save to safetensors/npy for 2–3 short excerpts (e.g. a 9.29 s chunk, plus a < 10 s file) and one full song.
- **Unit parity, stage by stage:**
  - max abs / relative error at each stage; expect ~1e-5 relative for fp32 with different summation order;
  - RoPE and GELU tested in isolation;
  - STFT→iSTFT round-trip identity.
- **End-to-end:**
  - SNR between JVM and PyTorch bass stems; target above 60–80 dB.
  - Compare against the existing audio-separator `(Bass)` WAV output: expect small MPS-vs-CPU differences, and 16-bit
    quantisation.
  - Optionally check SDR on a MUSDB-style track to confirm no regression.
- **Performance benchmark:** time one chunk (3.6 TFLOP) and report GFLOP/s to choose between the Vector API, Accelerate
  FFM and ONNX Runtime.

## 7. Optional reductions that keep compatibility
- Compute only `mask_estimators.0` (bass). This is exact and saves about 5% of compute and 70% of weights.
- Compute only the bass head bands covering below ~1 kHz, then iSTFT with the other bins zeroed. This is exact for the
  part we keep after the 250 Hz crossover, provided the filtfilt stopband leakage above 1 kHz is negligible (8th-order
  effective, so it is). It saves memory, not meaningful compute.
- `num_overlap=1` halves compute but changes the output relative to current behaviour.
- Smaller `dim_t` reduces latency and makes time attention cheaper. That cost is quadratic, but attention is only 27% of
  the compute, so the overall effect is close to linear. It changes quality relative to current output.
