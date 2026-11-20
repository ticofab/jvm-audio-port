# htdemucs_ft (bass sub-model `d12395a8`): JVM port assessment

Scope: inference only, for the bass output of `htdemucs_ft`, as `audio-separator` 0.48.0 runs it in splitter.
This note does not contain port code.

Path abbreviations used below:

- `AS/` = `sources/audio-separator/audio_separator/separator/`
- `DM/` = `AS/uvr_lib_v5/demucs/` (the vendored Demucs code that actually runs)
- `UP/` = `sources/demucs/demucs/` (upstream adefossez/demucs)
- `CPP/` = `sources/demucs.cpp/`, `ONNX/` = `sources/demucs.onnx/`

## 0. Summary

| Item | Value |
|---|---|
| Network | `HTDemucs`, 4 sources, depth 4, CaC spectrogram branch + waveform branch, 5-layer cross-domain transformer at d=512 |
| Parameters | **41,984,456** (533 tensors), stored as **float16** in the `.th` (84 MB) |
| Segment | 39/5 s = **343,980 samples**; stride 257,985 (overlap 0.25) |
| Compute | **~167.5 GMAC ≈ 335 GFLOP per segment**, i.e. **~57 GFLOP per audio second per shift** |
| audio-separator settings | `shifts=2`, `overlap=0.25`, segment "Default" (=7.8 s), all **4** sub-models run |
| Bass-only port saves | 4x (one sub-model instead of four); another 2x if `shifts=1` |
| 4-min song, bass model only | ~14 TFLOP (shifts=1) / ~28 TFLOP (shifts=2) |
| CPU feasibility | Yes. PyTorch CPU on this M1 Max already does ~150 GFLOP/s effective. A tuned JVM port should take about 1–2.5 min per 4-min song at shifts=1, single model, with no GPU |
| Effort | **~4–5k LOC Java, ~7–10 weeks** for one experienced developer, including validation and performance work |
| Riskiest parts | GEMM/conv performance in pure Java; exact `torch.stft`/`istft` semantics; reshape and layout bugs around CaC and the transformer token order; the many normalization variants |

The claim is correct: in `htdemucs_ft`, bass comes only from sub-model `d12395a8`. Section 1.6 has the details.

## 1. Inference path for one song

### 1.1 Loading (audio-separator)

- Defaults: `demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True}`
  (`AS/separator.py:141`). splitter's `separate.py` doesn't override them. It sets only `output_single_stem="Bass"`.
  The `DemucsSeparator` constructor uses the same fallbacks: shifts 2 (`AS/architectures/demucs_separator.py:68`) and
  overlap 0.25 (`:74`).
- Mix loading: `librosa.load(path, mono=False, sr=44100)` (`AS/common_separator.py:284`). This resamples if needed
  and gives float32 in [-1, 1]. Mono input is duplicated to stereo (`:309-311`). Non-finite samples are rejected.
- Model: `get_demucs_model(name="htdemucs_ft", repo=model_dir)` (`demucs_separator.py:125`) reads
  `htdemucs_ft.yaml`, loads the 4 `.th` files and builds a `BagOfModels(models, weights, segment=None)`
  (`DM/repo.py:120-130`). `demucs_segments("Default", ...)` changes nothing (`DM/apply.py:263-273`).

### 1.2 Pre-processing: whole-song normalization

`demucs_separator.py:213-220`:

```
ref = mix.mean(0)                       # mono average, over the WHOLE song
ref_mean = ref.mean(); ref_std = ref.std()    # torch.std is unbiased (N-1)
mix = (mix - ref_mean) / max(ref_std, eps_f32)
```

After `apply_model`: `sources = sources * ref_std + ref_mean` (`:237`). Upstream Demucs does the same thing with
`std + 1e-8` instead of a clamp (`UP/api.py:269-287`).

This is a whole-song dependency, but a weak one. The network normalizes each segment again internally (section
1.5), so the global scale cancels out except through the `1e-5` epsilons. The global mean of music is roughly 0.
For streaming, a running or fixed estimate should be indistinguishable. Verify this empirically.

### 1.3 `apply_model`: the bag, then shifts, then split, then the segment (`DM/apply.py:124-260`)

The recursion is bag, then shifts, then split, then the single segment:

1. **Bag** (`:169-195`): it loops over **all** sub-models. Zero weights are not skipped. For each model it runs
   `apply_model(sub_model, mix, shifts=2, split=True, ...)`, multiplies source `k` by `weight[k]` and accumulates. It
   then divides each source by the sum of its weights. With the identity weight matrix, the output for source `k` is
   simply sub-model `k`'s source `k`.
2. **Shifts** (`:202-214`): `max_shift = int(0.5*44100) = 22050`. The mix is padded by 22050 samples on both sides.
   `TensorChunk.padded` takes real samples where they exist and zero-pads beyond the song edges (`:97-113`). For each of
   the `shifts` passes it draws `offset = random.randint(0, 22050)`, which is inclusive, and uses Python's unseeded
   global RNG. So **outputs are non-deterministic run to run**. It runs split inference on
   `padded[offset : offset + length + max_shift - offset]`, keeps `out[..., max_shift - offset:]`, and averages over the
   shifts. splitter's HISTORY.md measures run-to-run variance at about 25–32 dB SNR because of this.
3. **Split** (`:215-250`): `segment = int(44100 * Fraction(39,5)) = 343980`, `stride = int(0.75*343980) = 257985`.
   Chunks start at `range(0, length, stride)`. The transition weight is a triangle,
   `[1..171990, 171990..1] / 171990`, raised to `transition_power=1`. Accumulation is
   `out += w[:n]*chunk_out; sum_w += w[:n]`, then `out /= sum_w`.
4. **Segment** (`:251-260`): `valid_length = model.valid_length(n) = 343980` (`DM/htdemucs.py:470-482`). A short
   last chunk is padded to 343980 by `TensorChunk.padded`. This is centred and **pulls real neighbouring audio** from
   the underlying tensor where available, not zeros. The output goes through `center_trim` back to `n`.
   `HTDemucs.forward` would zero-pad on the right too if needed (`htdemucs.py:487-494`).

Then `sources[[0,1]] = sources[[1,0]]` (`demucs_separator.py:239`) swaps drums and bass, so that
`DEMUCS_4_SOURCE_MAPPER[Bass] = 0` (`:16`) picks the bass. Only the bass stem is written.

### 1.4 Output writing

`write_audio`, then `spec_utils.normalize(wave, max_peak=0.9, min_peak=0.0)` (`AS/common_separator.py:344` and
`:431`, `AS/uvr_lib_v5/spec_utils.py:100-123`, thresholds from `AS/separator.py:130-131`). If the stem's peak is above
0.9, the **whole stem** is scaled down to peak 0.9. That is a whole-song dependency, though usually inactive for a
bass stem. `min_peak=0` never amplifies. The stem is written with the input's subtype (PCM_16 etc.). splitter then
applies its 250 Hz crossover to it (`splitter/code/src/main/resources/separate.py`).

### 1.5 One segment through `HTDemucs.forward` (`DM/htdemucs.py:484-614`)

Shapes are for B=1 and one 343,980-sample segment.

**STFT** (`_spec`, `:384-404`, and `DM/spec.py:13-21`):

- nfft 4096, hop 1024. `le = ceil(343980/1024) = 336`.
- First, reflect-pad by 1536 on the left and `1536 + 336*1024 - 343980` on the right (`pad1d`, `DM/hdemucs.py:22-39`).
- `torch.stft(n_fft=4096, hop=1024, window=hann_window(4096)` (periodic), `win_length=4096, normalized=True,
  center=True, pad_mode="reflect")`. `center=True` adds a second reflect pad of 2048. `normalized=True` multiplies
  by 1/sqrt(4096).
- Drop the Nyquist bin, giving 2048 bins. Then keep frames `[2, 2+336)`. Result z is (1, 2, 2048, 336) complex.

**CaC** (`_magnitude`, `:416-425`): `view_as_real` and permute to (1, 4, 2048, 336). Channel order is
`[L.re, L.im, R.re, R.im]`.

**Per-segment normalization** (`:502-511`):

- Spectrogram: `x = (x - mean)/(1e-5 + std)` over (C, F, T).
- Waveform: `xt = (mix - meant)/(1e-5 + stdt)` over (C, T).
- Both `std` are unbiased, and the epsilon is added to the std, not the variance.

**Encoders** (`HEncLayer`, `DM/hdemucs.py:68-154`). There are 4 freq layers and 4 time layers. With
`norm_starts=4`, no encoder or decoder GroupNorm is active: `norm1`/`norm2` are Identity.

| idx | freq branch out (C, F, T) | time branch out (C, T) | layer ops |
|---|---|---|---|
| 0 | (48, 512, 336) | (48, 85995) | conv k8 s4 p2 (Conv2d `[8,1]` on freq), then +freq-emb (idx 0 only), GELU, DConv, 1x1 rewrite to 2C, GLU |
| 1 | (96, 128, 336) | (96, 21499) | same |
| 2 | (192, 32, 336) | (192, 5375) | same |
| 3 | (384, 8, 336) | (384, 1344) | same |

- The time encoder right-pads to a multiple of 4 before each conv (`hdemucs.py:126-129`). The padded lengths are
  343980, 85996, 21500 and 5376.
- The time branch is never "empty" here, because freqs stay above the kernel size. The two branches merge only in the
  transformer.
- Frequency embedding after encoder 0 (`htdemucs.py:534-539`): `x += 0.2 * (10 * emb.weight[f])`, with
  `freq_emb.embedding.weight` (512, 48). Note the 10x `ScaledEmbedding.scale` (`hdemucs.py:42-63`).
- The DConv on the freq branch runs on `(B*F, C, T)`. That is, it works along **time** per frequency row, after
  `permute(0,2,1,3)` (`hdemucs.py:142-148`).

**DConv** (`DM/demucs.py:99-170`), with depth 2 and compress 8. Each block is:
`x + LayerScale(GLU(GN1(Conv1x1(C/8→2C)(GELU(GN1(Conv1d(C→C/8, k3, dilation 1 then 2, pad=dil)(x)))))))`.

- `GN1` is `GroupNorm(1, ·)`. It normalizes over (C, T) of the whole segment row.
- dconv_mode=3 (from the checkpoint), so the decoders have DConv too.

**Bottleneck and transformer** (`htdemucs.py:542-556`, `DM/transformer.py:415-559`):

- `channel_upsampler` (1x1 conv 384→512) on both branches.
- Freq branch: tokens are `(t1 fr)`, which is **time-major**: 336*8 = **2688 tokens**. Add `LayerNorm`, then the 2D
  sin embedding (`create_2d_sin_embedding`, `transformer.py:29-48`).
- Time branch: **1344 tokens**. Add `LayerNorm`, then the 1D sin embedding (`create_sin_embedding`, `:19-26`). It uses
  a different formula, `max_period ** (i/(half-1))`, with cos first.
- 5 layers per branch. Layers 0, 2 and 4 are self-attention (`MyTransformerEncoderLayer`, `:196-282`). Layers 1 and 3
  are cross-attention (`CrossTransformerEncoderLayer`, `:285-397`). Freq queries the old time state and time queries
  the *old* freq state (`old_x`, `:546-549`).
- Each layer is pre-norm: `x += γ1·MHA(LN(x))`, then `x += γ2·FFN(LN(x))`, then `x = GroupNorm(1, 512)(x)` (`norm_out`,
  over all tokens and channels).
- MHA has 8 heads, d_head 64, packed `in_proj` (1536×512) plus `out_proj`.
- FFN is 512→2048→512 with exact GELU. LayerScale γ is per channel.
- Cross-attention: q = `norm1(q)`, k = v = `norm2(k)`, FFN on `norm3`.
- Dropout is inactive at eval. Sparse attention is off (`t_sparse_*=False`).
- `channel_downsampler` (1x1, 512→384) follows the transformer.

**Decoders** (`HDecLayer`, `hdemucs.py:253-331`): `x + skip`, then rewrite conv (**3×3 Conv2d** on (F, T) for freq,
k3 Conv1d for time; C→2C), GLU, DConv, ConvTranspose k8 s4, then crop `pad=2` on each side, then GELU (except in the
last layer).

- Freq output: (16, 2048, 336), which is 4 sources × 2 ch × (re, im).
- Time output: (8, 343980), which is 4 sources × 2 ch. The crop uses the saved encoder lengths.

**Output** (`:581-614`):

- Freq: `x.view(1,4,4,2048,336) * std + mean`. The CaC mask is a reshape to complex (`_mask`, `:427-435`). Wiener
  filtering is off (`wiener_iters=0`, `cac=True`).
- iSTFT (`_ispec`, `:406-414`): zero-pad the Nyquist bin and 2 frames on each side, then
  `torch.istft(normalized=True, center=True, length=343980 + 2*1536 rounded up)`, then crop.
- Time: `xt * stdt + meant`.
- Result: **freq output + time output**, shape (1, 4, 2, 343980).

### 1.6 Sources order and the bass-only claim

- The checkpoint kwargs (read from `archive/data.pkl`, see section 4) have `sources = ['drums','bass','other','vocals']`.
  `demucs_separator.py:13` agrees.
- `htdemucs_ft.yaml` lists `models: [f7e0c4bc, d12395a8, 92cfc3b6, 04573f0d]` with identity weights. Upstream
  `UP/remote/htdemucs_ft.yaml` is identical.
- Row 1 (`d12395a8`) has weight 1 only for source 1 (bass), and `totals[1] = 1` (`apply.py:187-194`). So
  **bass = d12395a8's source-1 output, unchanged**. demucs.cpp does the same (`CPP/cli-apps/demucs_ft.cpp:224-239`, and
  `HT_HUB_PATH_FT_BASS = "d12395a8-e57c48e6.th"` in `CPP/scripts/convert-pth-to-ggml.py`).
- audio-separator still runs all 4 sub-models, each with 2 shifts (`apply.py:180-191`). A bass-only port does 1/4 of
  the work.
- Within `d12395a8`, all 4 output sources are still computed. Only the last ConvTranspose layers (freq 48→16, time
  48→8) could be pruned to the bass channels, which is a negligible saving.

### 1.7 Vendored vs upstream

`DM/htdemucs.py`, `transformer.py`, `hdemucs.py` and `spec.py` differ from `UP/` only in formatting, imports, MPS/CPU
fallbacks and the wiener import path. `DM/apply.py` has the same algorithm. Upstream adds a `segment` override, a
callback and a lock. So either codebase can serve as the reference.

### 1.8 Whole-song and long-range dependencies (for streaming)

1. Global mix mean and std (section 1.2). This is weak in practice, because of the per-segment re-normalization.
2. Output peak normalization to 0.9 (section 1.4). This needs the whole stem. Drop it, or use a limiter, when
   streaming.
3. Every segment is fully non-causal over its 7.8 s: per-segment input normalization, `GroupNorm(1)` over
   (C, T) in each DConv and each transformer `norm_out`, and full self- and cross-attention.
   - So streaming has to stay chunked: latency is at least one segment (7.8 s), plus the shift look-ahead (0.5 s),
     plus compute time.
   - With stride 5.85 s, a sample's overlap-add is final once the next segment has run.
4. Shifts need ±0.5 s of padding around the stream. That's fine.
5. Nothing else needs the whole song. There is no Wiener filter and no global statistics inside the model.

## 2. Architecture inventory (bass sub-model `d12395a8`)

### 2.1 Hyper-parameters (from the checkpoint's pickled `kwargs`)

| Parameter | Value |
|---|---|
| `segment`, `samplerate` | `Fraction(39,5)`, 44100 |
| `audio_channels` | 2 |
| `channels` | 48 |
| `growth` | 2 |
| `depth` | 4 |
| `nfft` | 4096 |
| `cac` | True |
| `wiener_iters` | 0 |
| `kernel_size` | 8 |
| `stride` | 4 |
| `time_stride` | 2 |
| `context` | 1 |
| `context_enc` | 0 |
| `norm_starts` | 4 |
| `norm_groups` | 4 |
| `dconv_mode` | 3 |
| `dconv_depth` | 2 |
| `dconv_comp` | 8 |
| `dconv_init` | 1e-3 |
| `multi_freqs` | [] |
| `freq_emb` | 0.2 |
| `emb_scale` | 10 |
| `emb_smooth` | True |
| `rewrite` | True |
| `bottom_channels` | 512 |
| `t_layers` | 5 |
| `t_heads` | 8 |
| `t_hidden_scale` | 4.0 |
| `t_emb` | "sin" |
| `t_norm_in` | True |
| `t_norm_first` | True |
| `t_norm_out` | True |
| `t_group_norm` | False |
| `t_layer_scale` | True |
| `t_gelu` | True |
| `t_cross_first` | False |
| `t_sparse_self_attn`, `t_sparse_cross_attn` | False |
| `t_max_period` | 10000 |
| `t_weight_pos_embed` | 1.0 |
| `t_sin_random_shift` | 0 |
| `use_train_segment` | default True |

Note that these differ from the `HTDemucs.__init__` defaults (`dconv_mode=1`, `bottom_channels=0`). You have to read
them from the checkpoint.

### 2.2 Parameter count (from the checkpoint tensors)

| Group | Params |
|---|---:|
| `encoder` (freq, 4 layers) | 1,424,940 |
| `tencoder` (time, 4 layers) | 1,424,172 |
| `decoder` (freq, 4 layers) | 4,562,620 |
| `tdecoder` (time, 4 layers) | 2,209,460 |
| `freq_emb` | 24,576 |
| `channel_{up,down}sampler{,_t}` | 788,224 |
| `crosstransformer` (norm_in ×2 + 10 layers) | 31,550,464 |
| **Total** | **41,984,456** |

That is 533 tensors, all `HalfStorage`, contiguous, with storage offset 0.

A few representative shapes:

- `encoder.0.conv.weight` (48,4,8,1)
- `decoder.0.rewrite.weight` (768,384,3,3)
- `tencoder.0.conv.weight` (48,2,8)
- `tdecoder.3.conv_tr.weight` (48,8,8)
- `decoder.3.conv_tr.weight` (48,16,8,1)
- `crosstransformer.layers.0.self_attn.in_proj_weight` (1536,512)
- `linear1` (2048,512)

The full listing was generated with a safe stub unpickler. The scratch outputs are not kept in the repo.

At fp32 the weights take 168 MB in memory (84 MB if kept as fp16 and widened on the fly).

### 2.3 Tensor ops needed

| Op | Where | Notes |
|---|---|---|
| Reflect pad, zero pad | `_spec`, `pad1d`, enc/dec | two reflect pads before the STFT |
| STFT / iSTFT (4096, hop 1024, periodic Hann, `normalized=True`, `center=True`) | spec/ispec | the iSTFT divides by the window-square envelope (torch's overlap-add normalization) |
| Conv2d with kernel `[8,1]` stride `[4,1]` (really a 1D conv along freq) | freq encoders | |
| Conv2d 1×1 | freq encoder rewrite | |
| Conv2d 3×3, pad 1 | freq decoder rewrite | **the most expensive conv** |
| ConvTranspose2d `[8,1]` stride `[4,1]`, ConvTranspose1d k8 s4 | decoders | plus cropping |
| Conv1d k8 s4, k3, k3 with dilation 2, 1×1 | time branch, DConv, up/down samplers | |
| GroupNorm(1, C) over (C, T) | DConv (×2 per block), transformer `norm_out` | eps 1e-5, affine |
| LayerNorm(512) | transformer `norm_in`, norm1/2/3 | eps 1e-5 |
| GELU (exact, erf) | encoders, DConv, FFN, decoders | **Java has no `Math.erf`**, so you have to write one |
| GLU (split channels in half: a·σ(b)) | rewrite, DConv | |
| LayerScale (per-channel multiply) | DConv, transformer γ1/γ2 | |
| Embedding lookup + scale | freq emb | |
| Linear / GEMM | MHA in/out proj, FFN | |
| Multi-head scaled-dot-product attention + softmax | self (2688², 1344²) and cross (2688×1344) | |
| Sin positional embeddings, 1D and 2D | transformer input | two different formulas |
| Mean/std reductions (unbiased), elementwise, permutes and reshapes | normalization, CaC, token layout | |

These are not used, so you can skip them: LSTM, LocalState attention, Wiener filter, sparse attention masks, CAPE,
dropout and MultiWrap.

## 3. Compute cost

### 3.1 MACs per segment

These are analytic MAC counts, computed from the shapes in section 1.5 (scratch script, not kept).

| Block | GMAC | Share |
|---|---:|---:|
| Transformer self-attention layers (3 per branch) | 65.8 | 39% |
| Transformer cross-attention layers (2 per branch) | 40.2 | 24% |
| Freq decoder 3×3 rewrite convs | 28.5 | 17% |
| Freq decoder ConvTranspose | 5.8 | 3.5% |
| Freq encoder convs | 5.0 | 3% |
| Time decoder k3 rewrite | 4.8 | 3% |
| All remaining convs + DConv + up/down | ~17 | ~10% |
| **Total** | **~167.5 GMAC ≈ 335 GFLOP** | |

- Of the transformer total, about 42.5 GMAC is the attention matmuls (QKᵀ, AV). The rest is projections and FFN.
- STFT/iSTFT cost is negligible: about 0.2 GFLOP for the STFT, plus iSTFT of 8 channels.
- About 95% of the FLOPs can be expressed as GEMM: linear layers and im2col convs.

### 3.2 Per audio second and per song

- Segments per audio second = 44100/257985 = 0.171. That gives **~57 GFLOP per audio second per shift**, for one
  sub-model.
- A 4-min song (240 s) takes about 42 segments per shift: **~14 TFLOP at shifts=1, ~28 TFLOP at shifts=2**.
- audio-separator's actual workload (4 models × 2 shifts) is about 112 TFLOP.

Sanity check against splitter's HISTORY.md: the Demucs CLI (shifts=1, 4 models) on CPU took 94 s for a 60 s clip.
That's 11 segments × 4 models × 335 GFLOP ≈ 14.7 TFLOP in 94 s, about **155 GFLOP/s effective for PyTorch CPU on this
M1 Max**. That's consistent with the estimate.

### 3.3 JVM CPU feasibility

- **Hardware:** this machine is an Apple M1 Max with 8 performance cores, 2 efficiency cores and 32 GB. It runs
  OpenJDK 27.
  - NEON is 128 bits wide (4 fp32 lanes), so the FMA peak is about 100 GFLOP/s per performance core, or about
    800 GFLOP/s across them.
  - There is no SVE, and the JVM can't reach Apple's AMX directly.
- **Pure Java with the Vector API** (`jdk.incubator.vector`, which needs `--add-modules`; check its status on JDK 27):
  - A well-blocked, multithreaded SGEMM typically reaches 30–50% of peak on large matrices.
  - Here most GEMMs are large (2688×512×1536, 2688×512×2048, im2col convs with K = 384·9), so 150–350 GFLOP/s on
    GEMMs is plausible, and maybe 100–250 GFLOP/s end to end.
  - That works out to **about 1–2.5 min per 4-min song for bass at shifts=1, or 2–5 min at shifts=2**. That's roughly
    PyTorch-CPU speed for one model, and much faster than today's CPU path, which runs 4 models.
- **FFM (Panama, final since JDK 22) calling Accelerate `cblas_sgemm`:**
  - This uses AMX, so the GEMM-able 95% could run several times faster, and vDSP covers the FFT.
  - There's still no Python and no extra runtime, but it is macOS-specific. Keep a pure-Java fallback.
- **GPU:** not needed for an offline 1–3 min per song. The JVM GPU options (Metal through FFM, or CUDA bindings) would
  be a large separate project. If GPU speed matters, ONNX Runtime Java with the CoreML or CUDA execution provider is
  the realistic route (section 5.2). It is not "native JVM code", though.
- **Memory per segment** (fp32):
  - Activations peak at about 66 MB each, for example the (96, 512, 336) rewrite output.
  - A naive im2col of the decoder-3 3×3 conv (48·9 × 512·336) would take about 300 MB, so tile it.
  - Attention scores are 2688² ≈ 29 MB per head; process head by head, or tile.
  - About 0.5–1 GB of working set per concurrent segment is realistic.

## 4. Weights loading

### 4.1 The `.th` format, observed on `d12395a8-e57c48e6.th`

- A standard `torch.save` zip file. There are 535 entries, **all stored (uncompressed)**:
  - `archive/data.pkl` (74.5 KB)
  - `archive/version` (contains "3")
  - `archive/data/0` … `archive/data/532`, the raw little-endian storages
- `data.pkl` is pickle protocol 2. The top-level dict has
  `{klass, args, kwargs, state, training_args, metrics}`.
  - `klass` is `GLOBAL 'demucs.htdemucs HTDemucs'`. It is only a reference and is never called.
  - `kwargs` holds the hyper-parameters (section 2.1). `segment` is `fractions.Fraction('39/5')`.
  - `state` is an `OrderedDict` of name to `_rebuild_tensor_v2(persistent_id, offset, shape, stride, False,
    OrderedDict())`, where `persistent_id = ('storage', torch.HalfStorage, '<key>', 'cpu', numel)`.
  - `training_args` and `metrics` contain numpy scalars (`numpy.core.multiarray.scalar`, `_codecs.encode`). They're
    irrelevant and can be ignored.
- Opcodes in the pickle: about 25 distinct ones, namely `PROTO, EMPTY_DICT/LIST/TUPLE, BINUNICODE, GLOBAL, MARK,
  TUPLE/1/2/3, BININT/1/2, BINFLOAT, NEWTRUE/NEWFALSE, NONE, BINPUT/LONG_BINPUT, BINGET/LONG_BINGET, BINPERSID,
  REDUCE, SETITEMS, APPEND(S), BUILD, STOP`.
- audio-separator's loader already restricts globals (`DM/states.py:35-74`). It then converts fp16 to fp32 via
  `load_state_dict` (`:77-104`). The checkpoint was saved with `half=True` (`get_state`, `:116-126`).

### 4.2 A minimal Java reader

This is about 300–400 LOC:

- `java.util.zip.ZipFile`.
- A tiny pickle VM with a stack, a memo and marks. It handles the opcodes above. It maps
  `GLOBAL torch._utils _rebuild_tensor_v2` to a tensor descriptor and turns `BINPERSID` into a storage reference.
  Every other global becomes an inert placeholder, so nothing ever executes.
- Read `archive/data/<key>` into a `ByteBuffer` (little-endian) and widen with `Float.float16ToFloat` (JDK 20+).
- Check that all strides are contiguous and offsets are 0. They are in this file, so you don't need a general
  strided-view loader.

### 4.3 Simpler alternatives

- Ship a static name → (key, shape) table. It's 533 lines and the layout is fixed per model. Then read only the
  storages.
- Or convert once to safetensors or a raw `.bin` with a manifest, done offline with Python. That is what demucs.cpp
  does: a ggml-like format of `[n_dims, name_len, dims…, name, fp16 data]` (`CPP/scripts/convert-pth-to-ggml.py`).
- Reading the `.th` directly in Java is cheap and avoids a conversion step, so I'd do that.

## 5. Prior art

### 5.1 demucs.cpp (C++17, Eigen, by sevagh)

- **Size:** `src/` is 8,640 LOC, including v3 support. The v4-only part is roughly 4.5–5k LOC:
  - `model.hpp` v4 part (~670): the weight structs
  - `model_load.cpp` v4 part (~1,300): mostly per-tensor boilerplate
  - `conv.hpp` (528): templated im2col + GEMM convs and transposed convs
  - `encdec.cpp` (~360)
  - `layers.cpp` v4 part (~530): GroupNorm, fused GN+GELU, GLU, LayerNorm, DConv, attention layer
  - `crosstransformer.cpp` (339)
  - `model_inference.cpp` (~475)
  - `model_apply.cpp` (~290)
  - `dsp.cpp/.hpp` (~290): STFT/iSTFT on Eigen FFT

  Tests (`test/test_layers.cpp`, 2,234 LOC) compare against PyTorch dumps from `scripts/demucs_pytorch_layer_test.py`
  (min/max/mean summaries per layer, `NEAR_TOLERANCE 1e-4`).
- **Weights:** a one-time Python conversion to a ggml-style fp16 file (above). The loader widens fp16 to float.
- **STFT** (`src/dsp.cpp`): a hand-written reflect pad, Hann window, Eigen FFT, `1/sqrt(nfft)` scaling. The iSTFT
  divides by the summed squared window plus `1e-8`, which mimics torch/librosa.
- **Transformer** (`src/crosstransformer.cpp`, `layers.cpp:377+`):
  - Explicit 2D and 1D sin embeddings, ported line by line.
  - A per-head loop with `Q·Kᵀ/√64`, a max-subtracted softmax, then `·V`.
  - It reuses one `common_encoder_layer` for both the self and the cross layers.
- **Accuracy** (`CPP/.github/SDR_scores.md`): with the shift offset fixed to 1337 in both, htdemucs_ft SDR matches
  PyTorch within about 0.1 dB per source. For bass it was 4.590 vs 4.468 dB. They report no waveform SNR.
- **Speed** (`CPP/.github/PERFORMANCE.md`): it was deliberately written for low memory (WASM, phones), with many small
  loops and GEMVs.
  - A single htdemucs model on a 4-min song took 10m23s on a 16-core 5950X with OpenBLAS.
  - The 4-thread variant took 4m09s.
  - That's slower than PyTorch CPU. The author says large-matrix formulations would be faster.
  - Lesson: structure the port around big GEMMs, not loops.
- **Deviations from PyTorch** worth not copying (`CPP/src/model_apply.cpp`):
  - Only `shifts=1`, with `rand() % max_shift` (PyTorch's range is inclusive).
  - The last short chunk is zero-padded symmetrically, while PyTorch pads with real neighbouring audio
    (`TensorChunk.padded`).
  - These are small edge differences, but they break exact comparison.

### 5.2 demucs.onnx (by sevagh)

- `torch.onnx.export` fails on `stft`/`istft` inside the model. The fix is to move `_spec`, `_magnitude`, `_ispec` and
  `_mask` out as standalone functions. The core then becomes `forward(mix, x_cac) -> (x_spec_out, xt_out)`
  (`ONNX/demucs-for-onnx/demucs/htdemucs.py`, `ONNX/scripts/convert-pth-to-onnx.py`, opset 17, fixed input
  `(1,2,343980)`).
- `transformer.py` is unchanged. The cross-domain transformer exports fine as ordinary MatMul/Softmax.
- The C++ side reuses demucs.cpp's STFT and apply logic. It is about 1k LOC in `src/` plus the CLI, with a reduced
  ONNX Runtime build.
- The README claims "significantly faster" but publishes no numbers and no accuracy table.
- Lesson for us: **ONNX Runtime Java with STFT in Java** is a 1–2 week path to a working, fast, Python-free runtime.
  It is not pure JVM, because it bundles a native `libonnxruntime`. It is also an excellent **oracle** for validating
  a pure-JVM port without Python at test time.

## 6. Effort, risks and validation

### 6.1 Effort for a pure JVM port (one experienced developer)

| Work item | LOC (Java) | Weeks |
|---|---:|---:|
| `.th` reader (zip + pickle subset + fp16) and weight binding | 400 | 0.5 |
| Tensor core and ops (conv via im2col, conv-transpose, norms, GELU/erf, GLU, softmax), Vector-API SGEMM, threading | 1,500 | 2–2.5 |
| STFT/iSTFT with exact torch semantics (own radix-2 FFT, or JTransforms) | 300 | 1 |
| HTDemucs graph: encoders and decoders, DConv, freq-emb, transformer, positional embeddings, CaC, per-segment normalization | 900 | 1.5 |
| Apply layer: global normalization, shifts, split and overlap-add, `TensorChunk.padded` semantics; a `Flow` wrapper | 350 | 0.5 |
| Validation harness: Python dump script (one-off) and Java comparison tests | 800 | 1–1.5 |
| Performance tuning: blocking, tiling of im2col and attention, buffer reuse, parallelism | — | 1–2 |
| **Total** | **~4,300 (3.5–5k)** | **~7–10** |

This is in line with demucs.cpp's v4 portion (about 4.5–5k LOC C++, though a lot of that is boilerplate). Optional
extra: an FFM backend to Accelerate BLAS/vDSP, about 150 LOC and 0.5 weeks.

### 6.2 Riskiest parts

1. **Performance.** The port is only useful if the GEMM-shaped work (95% of FLOPs) runs at a decent fraction of peak.
   Risks: the incubator status of the Vector API, C2 intrinsification (keep the kernels in Java, not Scala), and
   memory bandwidth on the large im2col.
   - Mitigation: benchmark a single 2688×512×2048 GEMM and the decoder 3×3 conv in week 1, before building the rest.
2. **STFT/iSTFT equivalence.** Two reflect pads, periodic Hann, `normalized=True`, dropping the Nyquist bin, the
   frame slice `[2:2+le]`, the iSTFT window-envelope division, and the `length`/crop arithmetic.
   - Each one is easy to get subtly wrong, and errors show up as edge clicks or a level offset.
3. **Layout and reshape correctness.**
   - CaC channel interleave `[L.re, L.im, R.re, R.im]`
   - Freq-DConv `permute(0,2,1,3)`
   - Transformer token order `(t1 fr)` (time-major)
   - Output `view(B,S,-1,F,T)` and the complex re-pack
   - The time-branch pad-to-stride and crop-to-saved-length
4. **Normalization zoo.**
   - Per-segment unbiased std with `+1e-5` on the std
   - GroupNorm(1) over (C, T) in DConv and `norm_out`, as opposed to LayerNorm over C elsewhere
   - Global mix normalization with the eps clamp
5. **Exact GELU.** Java lacks `erf`. Use an fdlibm-quality port (`s_erf.c`, about 1 ulp).
   Cruder approximations such as Abramowitz–Stegun 7.1.26 (about 1.5e-7 absolute error) are probably fine
   numerically, but they make layer-level comparisons noisier.
6. **Apply-layer details.**
   - Inclusive `randint`
   - Last-chunk padding from real audio
   - Triangle weights normalized by 171990
   - `out[..., max_shift - offset:]`
   - Divide by `sum_weight`

### 6.3 Validation plan

Bit-for-bit equality is **not achievable** and not needed. GEMM summation order, the FFT algorithm and exp/erf
implementations differ, and even PyTorch CPU vs MPS differ. The natural acceptance bar is set by Demucs' own random
shifts: 25–32 dB run-to-run SNR (splitter HISTORY.md). A port that stays far above that, at the same shift offsets,
is indistinguishable.

1. **Reference dumps, one-off and offline.** splitter's existing Python venv already has torch and audio-separator.
   - Load `d12395a8` and register forward hooks on every module. Dump the inputs and outputs of: STFT, CaC,
     normalized x/xt, each `encoder[i]`/`tencoder[i]` (and DConv internals), freq-emb, up-sampler,
     transformer `norm_in` + positional embedding, each transformer layer for both branches, the down-sampler, each
     decoder, `_ispec`, and the final sum.
   - Use one real 343,980-sample segment of a song, plus a random-noise segment.
   - Write them as `.npy`. The header is trivial to parse in Java.
   - For the apply level, monkey-patch `random.randint` to return fixed offsets.
2. **Op-level unit tests.** Compare against torch on random inputs: conv2d `[8,1]`, 3×3, conv-transpose, GroupNorm,
   LayerNorm, GELU, GLU, MHA, stft/istft. Target max relative error ≤ 1e-6, or ≤ 1e-5 for long reductions.
3. **Layer-level tests**, fed with PyTorch's *input* to that layer so errors don't compound. Target relative L2 error
   below 1e-5 in the convolutional layers and below 1e-4 in the transformer layers.
4. **Chained, single segment.** The full `forward` on a real segment. Target SNR vs PyTorch CPU fp32 of 70 dB or more.
   Also compare each layer's chained output, to see where drift accumulates.
5. **End to end** with fixed shift offsets and `shifts=1`/`2`. Target 60 dB SNR or more on the bass stem, for a few
   songs including a mono one and one with a short tail.
6. **Pipeline check.** Run the port in place of audio-separator's htdemucs_ft in splitter and compare after the 250 Hz
   crossover. Expect a difference within run-to-run variance, plus the deliberate change of running only one sub-model.
   That change is exact, because the other sub-models contribute 0 to bass.
7. Optional: use an ONNX Runtime export of the same core as a Python-free CI oracle (section 5.2).

## 7. Recommendations

- Port only `d12395a8` and keep `shifts` configurable. Default to 1 for speed, and 2 to match today's output
  statistics.
- Structure the code around large GEMMs (im2col convs, full-sequence projections) and tile the attention by head.
  This is the opposite of demucs.cpp's loop-heavy style, which explains its slowness.
- Build the GEMM benchmark first. If pure Java falls far short of about 150 GFLOP/s here, decide early between an FFM
  backend to Accelerate and ONNX Runtime Java.
- For streaming, process 7.8 s windows with stride 5.85 s and the same triangular overlap-add. Replace the global
  normalization with a fixed or running estimate, and drop the global peak normalization.
