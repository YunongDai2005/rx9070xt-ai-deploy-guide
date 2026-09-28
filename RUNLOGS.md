# Run Logs · RX 9070 XT

Real run logs with annotations. [← Back to main doc](README.md) · [中文](RUNLOGS.zh.md)

The purpose of this file: **let you know what a healthy log looks like before you run it
yourself.** While following the main document, diff your logs against these and any deviation
becomes obvious immediately.

> **On "results"**: this file records **measurable** metrics only — wall time, VRAM, layer
> counts, throughput, errors. It contains **no assessment of the visual quality of generated
> output**, because no output image or video was viewed while writing it. Judge quality
> yourself; this document exists to tell you how long a given setting should take and how
> much VRAM it should use.

---

## 1. Ollama + Vulkan loading Qwen3.5 9B

### 1.1 Device detection

```
"inference compute" id=0 library=Vulkan name=Vulkan0
    description="AMD Radeon RX 9070 XT" type=discrete
    total="15.9 GiB" available="15.1 GiB"
"gpu memory" id=0 library=Vulkan available="14.6 GiB" free="15.1 GiB"
    minimum="457.0 MiB" overhead="0 B"
common_param: - Vulkan0 : AMD Radeon RX 9070 XT (16304 MiB, 15437 MiB free)
```

✅ **What you should see**: `library=Vulkan`, `type=discrete`, `15437 MiB free`.
❌ **If you see `library=ROCm`**: your Ollama build now supports ROCm on RDNA4, so this
document's Vulkan conclusions may no longer apply — re-measure throughput yourself.

### 1.2 Layer offload — the three lines that matter most

```
load_tensors: offloaded 34/34 layers to GPU
load_tensors: offloading 32 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors:      Vulkan0 model buffer size =  4717.38 MiB
load_tensors:  Vulkan_Host model buffer size =   545.63 MiB
```

✅ **`34/34` is the success marker.** The denominator is the model's total layer count; the
two numbers must match.
❌ **If you see something like `28/34`**: layers are stuck on CPU and throughput will collapse
to single-digit tok/s. Usually another process is holding VRAM — run the teardown script from
main doc §3.3 first.

VRAM allocation breakdown:

```
common_memory_breakdown_print:
  Vulkan0 (RX 9070 XT) | 16304 = 15386 + (5231 = 4717 + 306 + 208) + -4314
                         total   usable  sum   weights KV  other
```

### 1.3 Context: headroom that's easy to miss

```
print_info: n_ctx_train           = 262144
print_info: n_ctx_orig_yarn       = 262144
OLLAMA_CONTEXT_LENGTH:8192
```

> **Qwen3.5 9B natively supports 256K context; this deployment runs 8192.**
> That is not a model limit, it is a 16 GB trade-off — weights already take 4.7 GB, and KV
> cache grows linearly with context.
> **If you want long context**: either drop the quantization, or switch to llama.cpp with a
> quantized KV cache as in main doc §3.2.

Other defaults worth noting:

```
OLLAMA_FLASH_ATTENTION:false    <- not enabled on the Vulkan backend
OLLAMA_NUM_PARALLEL:1           <- single concurrency, sensible on a 16 GB card
OLLAMA_KEEP_ALIVE:5m0s          <- auto-unload after 5 idle minutes, frees VRAM for games
tokenizer.ggml.tokens arr[str,248320]   <- 248320-token vocabulary
```

### 1.4 Measured throughput

| Metric | Value |
|---|---|
| Generation | **~24 tok/s** |
| Prompt processing | 250–340 tok/s |
| Cold start | ~15 s |
| Weight VRAM | 4717.38 MiB |
| Host side | 545.63 MiB |

**What 24 tok/s feels like**: faster than human reading speed — no perceptible wait in
interactive use. For contrast, the 27B Q3_K on the same card via llama.cpp ROCm is noticeably
slower; it has 3× the parameters.

---

## 2. Qwen-Image 2.1 A/B benchmark

Full numbers are in main doc §5.2; here are the log-side observations.

### 2.1 Weight verification (do this before benchmarking)

```json
[
  { "file": "...\\qwen-image-2.1-Q4_K_M.gguf",
    "sha256":   "833439e91bc1152d28f37aa198c7f6f4218b7de95754c2f7a318a2422ab4b2f8",
    "expected": "833439e91bc1152d28f37aa198c7f6f4218b7de95754c2f7a318a2422ab4b2f8",
    "ok": true },
  { "file": "...\\qwen3vl_8b_int8_convrot.safetensors",
    "sha256":   "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f",
    "expected": "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f",
    "ok": true },
  { "file": "...\\qwen_image_2.1_vae_bf16.safetensors",
    "sha256":   "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9",
    "expected": "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9",
    "ok": true }
]
```

> **Wait for all `ok: true` before benchmarking.** A file truncated by an interrupted download
> will still load successfully but produce broken images — a failure mode that wastes a lot of
> time if you don't check hashes.

### 2.2 VRAM sampling curve (route B, int8_convrot)

The benchmark script samples peak allocation every 30 seconds. Full curve:

```
 29 s   peak allocated  0.000 GiB   <- still reading from disk
 59 s   peak allocated  0.000 GiB
 97 s   peak allocated  5.182 GiB   <- weights start landing on the GPU
127 s   peak allocated  5.460 GiB
157 s   peak allocated  5.739 GiB
187 s   peak allocated  6.047 GiB
217 s   peak allocated  7.056 GiB
247 s   peak allocated 12.109 GiB   <- sampling begins, VRAM jumps
277 s   peak allocated 13.707 GiB
307 s   peak allocated 13.707 GiB   <- plateau
337 s   peak allocated 13.707 GiB
FINISHED 363.88 s
```

**Two features you can compare against directly:**

1. **VRAM is 0 for the first 90 seconds** — that is all disk read. Don't assume it's hung.
2. **At 247 s it jumps from 7 GB to 12 GB** — the sampling-phase spike. If your card OOMs
   at this step, you didn't leave enough headroom earlier.

### 2.3 Final A/B result

| | Cold | Warm | Resident VRAM | Free after |
|---|---|---|---|---|
| A · Q8_0 GGUF | 294.7 s | **40.4 s** | 14.03 GB | 2.38 GB |
| B · int8_convrot | 155.9 s | **97.4 s** | 6.02 GB | 10.49 GB |

**2.4× apart on warm time, 2.3× apart on VRAM — a clean inverse trade-off.**
1024×1024 / 20 steps / cfg 6.0 / euler.

---

## 3. MiniMax H3 text-to-video · full run

The one fully successful run: **277.87 seconds / 10 steps**.

### 3.1 Startup

```
Total VRAM 16304 MB, total RAM 32683 MB
Set vram state to: NORMAL_VRAM
[WARNING] Dynamic vram disabled with argument.
Found comfy_kitchen backend hip: {'available': True, 'disabled': False, ...}
                        triton: "ImportError: No module named 'triton'"
    0.0 seconds: custom_nodes\qwen_image21_gguf_encoder.py
    0.0 seconds: custom_nodes\Qwen21-VRAM-Scheduler-Clean
    0.0 seconds: custom_nodes\H3-VRAM-Gate
    0.1 seconds: custom_nodes\ComfyUI-GGUF
```

✅ All three custom nodes must appear in this list, or the patches didn't take effect.
✅ The `triton` ImportError is **expected** — there is no triton wheel for Windows ROCm.

### 3.2 The mmap patch firing

```
[H3-VRAM-Gate] mmap loading qwen3vl_32b_minimax_h3_int8_convrot.safetensors
Found quantization metadata version 1
Using MixedPrecisionOps for text encoder
[H3-VRAM-Gate] mmap loading minimax_h3_fl2va_pruned_int8_convrot.safetensors
Detected mixed precision quantization
Native ops:   float8_e5m2, int8_tensorwise, asym_w4a8_int8, convrot_w4a4, float8_e4m3fn
emulated ops: nvfp4, mxfp8
model weight dtype torch.bfloat16, manual cast: torch.bfloat16
model_type FLOW_AV
```

> **`mmap loading` should appear twice** — both the 25.3 GiB encoder and the 19.5 GiB
> diffusion model exceed the 16 GiB threshold. **If it never appears, the patch isn't
> installed and the process will be killed outright by `0xC0000005`** (see main doc §6.1).

**The `Native ops` / `emulated ops` pair is the only reliable way to tell whether a
quantization format is actually accelerated.** `nvfp4` and `mxfp8` sit on the emulated side —
they run, but without hardware acceleration.

### 3.3 Model loading: partial, but not degraded

```
Requested to load MiniMaxH3TEModel_
loaded partially; 14293.74 MB usable, 13418.44 MB loaded,
                  12465.00 MB offloaded, 875.29 MB buffer reserved,
                  lowvram patches: 0

Requested to load MiniMaxH3
loaded partially; 13866.61 MB usable, 13386.64 MB loaded,
                   6609.51 MB offloaded, 661.52 MB buffer reserved,
                  lowvram patches: 0
```

> ⭐ **`lowvram patches: 0` is the single most important line to grep in the whole log.**
> Models being split (about 19 GB offloaded to CPU in total) is normal and necessary.
> **But the moment `lowvram patches` is non-zero, you have fallen into ComfyUI's degradation
> path and throughput drops off a cliff.** Partial loading ≠ degradation — these two are
> frequently confused.

The AOTriton attention backend is confirmed active:

```
UserWarning: Using AOTriton backend for Efficient Attention forward...
(aten/src/ATen/native/transformers/hip/attention.hip:1452)
```

### 3.4 Sampling: a 3.4× slower first step is normal

```
  0%|          | 0/10 [00:00<?, ?it/s]
 10%|█         | 1/10 [00:13<02:00, 13.41s/it]   <- includes first kernel compile
 20%|██        | 2/10 [00:19<01:10,  8.82s/it]
 30%|███       | 3/10 [00:25<00:54,  7.81s/it]
 40%|████      | 4/10 [00:32<00:43,  7.30s/it]
 50%|█████     | 5/10 [00:37<00:32,  6.53s/it]
 60%|██████    | 6/10 [00:41<00:22,  5.57s/it]
 70%|███████   | 7/10 [00:44<00:15,  5.00s/it]
 80%|████████  | 8/10 [00:48<00:08,  4.45s/it]
 90%|█████████ | 9/10 [00:51<00:04,  4.14s/it]
100%|██████████| 10/10 [00:55<00:00,  3.98s/it]  <- converged
100%|██████████| 10/10 [00:55<00:00,  5.52s/it]  <- average
```

> **First step 13.41 s, last step 3.98 s.** The first includes MIOpen kernel selection and the
> AOTriton attention kernel compile. **Evaluate this card on the converged 3.98 s/it; the
> 5.52 s/it average understates it by 39%.** If your first step takes far more than 13 s,
> check that the MIOpen cache directory is writable (`MIOPEN_USER_DB_PATH`, main doc §4.1).

### 3.5 Decode stage

```
[H3-VRAM-Gate] free VRAM 15359 -> 15357 MB
Requested to load MiniMaxH3AudioVAE
loaded completely; 13513.42 MB usable, 577.08 MB loaded, full load: True
Requested to load MiniMaxH3VideoVAE
loaded completely; 12827.35 MB usable, 4966.19 MB loaded, full load: True
Prompt executed in 277.87 seconds
```

⚠️ **Honest caveat**: the gate node freed only 2 MB (15359 → 15357), meaning ComfyUI had
already cleared everything by the end of sampling. **This log does not prove the gate "saved"
the run** — its value is guaranteeing deterministic ordering rather than relying on ComfyUI
happening to get it right.

### 3.6 Where the time goes

| Stage | Time | Share |
|---|---|---|
| Model load + text encode | ~215 s | **77%** |
| Sampling (10 steps) | 55 s | 20% |
| Dual VAE decode | ~8 s | 3% |
| **Total** | **277.87 s** | |

> **The bottleneck is disk, not compute.** `fast_disk=False` in the log is the slow-disk flag
> — H3's 53 GiB of weights live on a mechanical drive here. **Moving to an SSD is the only
> meaningful optimization; no sampler setting touches that 77%.**

### 3.7 Failure record: the audio branch

```
Prompt executed in 148.23 seconds
!!! Exception during processing !!!
AttributeError: 'ModelSamplingAdvanced' object has no attribute 'audio_scale'
```

`model_type` is `FLOW_AV` (joint audio-video), so the sampling node must carry an
`audio_scale` attribute. **The video branch works completely; the audio branch does not.**
This error is **unrelated to RDNA4 and reproduces on NVIDIA** — it is a node selection issue.

---

## 4. Installation logs

The places most likely to stall during setup, each backed by a log.

### 4.1 The ROCm SDK is not on PyPI

```
Collecting rocm-sdk-core==7.2.1
  Downloading https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/
             rocm_sdk_core-7.2.1-py3-none-win_amd64.whl (644.8 MB)
```

> **The Windows ROCm SDK comes from `repo.radeon.com`, not PyPI.** A single 644.8 MB wheel.
> On this machine the download failed partway through with a network read error (the log shows
> an urllib3 `_error_catcher` traceback).
> **Be ready to resume** — pulling 644 MB in one shot is unrealistic on an unstable link.

### 4.2 Mirrors, and skipping 490 MB of unused assets

```
=== 1. build a filtered requirements list ===
  excluded: comfyui-workflow-templates==0.11.66
  written:  requirements-nomedia.txt

=== 2. template package (body + JSON only, skipping ~490MB of preview assets) ===
>>> templates body (--no-deps)  [1]  mirror.nju.edu.cn   OK
>>> templates core + json       [1]  mirror.nju.edu.cn   OK
=== 3. remaining ComfyUI dependencies ===
>>> ComfyUI requirements (filtered) [1] mirror.nju.edu.cn OK
=== 4. ComfyUI-GGUF dependencies ===
>>> GGUF requirements           [1]  mirror.nju.edu.cn   OK
```

**Two practical tricks:**

1. **`comfyui-workflow-templates` ships roughly 490 MB of preview thumbnails.**
   Install body + JSON only with `--no-deps` and save all 490 MB.
   Cost: empty thumbnails in the template browser. **Generation is unaffected** and local
   workflow files load normally.
2. **Route pip through a regional mirror** (`mirror.nju.edu.cn` here), or the large torch ROCm
   wheels are effectively undownloadable.

### 4.3 Post-install verification output

```
=== 5. verification ===
torch 2.9.1+rocm7.2.1
gpu_available True
device AMD Radeon RX 9070 XT
imports_ok True
```

✅ **All four lines must be correct.** The most common cause of `gpu_available False` is a
CPU-only torch overwriting the ROCm build — check that `torch.__version__` still ends in
`+rocm7.2.1`.

---

## 5. What to expect

**Given a setting → how long it should take and how much VRAM it should use.**
For pre-run comparison, not a quality judgment.

### 5.1 Chat

| Model | Context | VRAM | Speed | Feel |
|---|---|---|---|---|
| Qwen3.5 9B Q4 | 8K | 4.7 GB | 24 tok/s | No perceptible wait |
| 27B Q3_K | 16K (KV q4_0) | ~12.6 GB + KV | Clearly slower than 9B | The price of 3× parameters |
| 12B Q6_K | 32K (KV q8_0) | ~9.4 GB + KV | Between the two | Long context at higher quant quality |

### 5.2 Text-to-image

| Route | Resolution | Steps | Warm time | Resident VRAM | Room for anything else |
|---|---|---|---|---|---|
| Qwen-Image 2.1 · Q8_0 | 1024² | 20 | **40 s** | 14.0 GB | ❌ card is full |
| Qwen-Image 2.1 · int8 | 1024² | 20 | **97 s** | 6.0 GB | ✅ 10.5 GB left |
| + Pruna 8-step LoRA | 1024² | 8 | not timed separately | same | 60% fewer steps |
| SDXL RealVisXL V5.0 | 1024² | — | not timed | ~7 GB | defaults are fine |

⚠️ **1024×1024 is the ceiling on 16 GB without offload** — both Qwen-Image routes peak at
~15.9 GiB, the physical limit. Higher resolutions require discussing offload or scaling down
first.

### 5.3 Text-to-video

| Item | Value |
|---|---|
| Steps | 10 |
| Total time | **277.87 s** (~4 min 38 s) |
| Converged sampling speed | 3.98 s/it |
| Total weights | 53 GiB (needs a separate large drive) |
| Peak on GPU | 13.4 GB (plus ~19 GB offloaded to CPU) |
| System RAM requirement | 32 GB is just barely enough (read-only mmap avoids commit charge) |
| Audio branch | ⚠️ not working (§3.7) |

> **H3 on this card is "runnable", not "pleasant"** — a single 10-step video takes close to
> 5 minutes, 77% of it disk read. Putting the weights on an SSD helps substantially, but the
> 53 GiB footprint is a hard requirement.

---

## 6. Quick self-check

After following the main doc, grep your own logs in this order:

```bash
# 1. torch is the ROCm build
python -c "import torch;print(torch.__version__)"      # must contain +rocm

# 2. the card is detected
python -c "import torch;print(torch.cuda.get_device_properties(0).gcnArchName)"   # gfx1201

# 3. Ollama offloaded every layer
grep "offloaded" server.log                            # must be N/N

# 4. custom nodes loaded
grep "custom_nodes" comfyui.log                         # all three present

# 5. the H3 mmap patch fired
grep "mmap loading" comfyui.log                         # must appear twice

# 6. no degradation path
grep "lowvram patches" comfyui.log                      # must all be 0

# 7. weights aren't truncated
# run audit_environment.py from the main doc; every `matches` should be true
```

If any check fails, look the symptom up in main doc §9.

---

[← Back to main doc](README.md)
