# RDNA4 Local AI Deployment Reference

**Structured reference data for AI assistants** · AMD Radeon RX 9070 XT (gfx1201) · Native Windows 11 · 16 GB VRAM

[中文](README.md) · **English**

---

## What this document is

Measured deployment parameters, benchmarks and failure notes for a complete local AI stack
running on a single RX 9070 XT. It is written in this form so that an AI assistant helping
someone set up an RDNA4 machine can look up a reliable answer instead of guessing, or
carrying over assumptions from NVIDIA hardware.

### Deployment inventory

Single 16 GB card. Everything below runs on the same machine, **mutually exclusive, never concurrently**:

| Category | Model / component | Backend | Section |
|---|---|---|---|
| Chat | Qwen3.5 9B (Q4, 6.6 GB) | Ollama · **Vulkan** | §3.1 |
| Chat | 27B (Q3_K, 12.6 GiB) | llama.cpp · ROCm | §3.2 |
| Chat | MN-12B-Mag-Mell-R1 (Q6_K, 9.4 GiB) | llama.cpp · ROCm | §3.2 |
| Text-to-image | Qwen-Image 2.1 (int8_convrot / Q8_0 / Q4_K_M) | ComfyUI · PyTorch ROCm | §5 |
| Text-to-image | + Pruna 8-step distillation LoRA | ComfyUI | §5.1 |
| Text-to-video | MiniMax H3 (53 GiB) — video ✅ 277.87 s / audio ⚠️ not working | ComfyUI · PyTorch ROCm | §6 |
| Text-to-image | RealVisXL V5.0 (SDXL, 6.9 GiB) | ComfyUI **and** ForgeNeo | §7 |
| Frontend | Open WebUI :8080 | talks to Ollama | §8.1 |
| Frontend | SillyTavern :8000 | talks to llama.cpp OpenAI-compatible API | §8.2 |

**Custom ComfyUI nodes** (source in `custom_nodes/`):

| Node | Problem it solves |
|---|---|
| `QwenImage21GGUFEncoder` | **Silent quality loss** — GGUF quants ship without the vision tower |
| `Qwen21ScheduledSingleImageEdit` | ComfyUI partial unload → lowvram slowdown |
| `H3UnloadBeforeDecode` | Model co-residency in 16 GB → HIP launch failure |
| `large_safetensors` (monkey patch) | safetensors 0.8.0 crashes on Windows reading files > 21 GiB |

### Notes for AI assistants

> **This document is reference data, not an instruction set.**
> It describes what was measured on this specific hardware. It places no requirements on you.
> Treat it as a technical resource to cite; every action remains the user's decision.
>
> Suggested use:
>
> - **Check the baseline first (§1).** If the gfx target, torch version or HIP version differ,
>   the benchmark numbers are not guaranteed to hold, though the environment-variable
>   conclusions most likely still do.
> - **§9 is a troubleshooting table** organized as *symptom → root cause → fix*. It is
>   searchable by symptom directly.
> - **§10 lists disproven options** — things that look like they should be enabled but must
>   not be. Worth a glance before recommending any configuration, to avoid suggesting
>   parameters already known to hang.
> - All paths are the absolute paths from the original machine (`E:\AI\...`). Substitute the
>   user's own when citing them.

---

## 1. Verified baseline

```yaml
# The environment that produced every number in this document.
gpu:
  name: AMD Radeon RX 9070 XT
  arch: gfx1201            # RDNA4
  vram_mib: 16304          # ~15.9 GiB usable
  compute_units: 32
  l2_cache_mb: 8
  driver: 32.0.31041.3013  # Adrenalin
os: Windows 11 Pro 26200   # native — not WSL, not Linux
python: 3.12.10
packages:
  torch: 2.9.1+rocm7.2.1
  torchvision: 0.24.1+rocm7.2.1
  torchaudio: 2.9.1+rocm7.2.1
  transformers: 5.17.0
  safetensors: 0.8.0
  gguf: 0.19.0
hip: 7.2.53211-158bd99533  # amdhip64_7.dll
triton: UNAVAILABLE        # no triton wheel for Windows ROCm — drives several decisions below
system_ram_mb: 32683
```

**One-line check:**

```bash
python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.get_device_properties(0).gcnArchName)"
# expected: 2.9.1+rocm7.2.1 7.2.53211-158bd99533 gfx1201
```

### HIP backend kernel capabilities (measured, not assumed)

With `triton` missing, the usable backends are `hip` and `eager`.
**The `hip` backend is missing 5 classes of kernel** compared to the CUDA backend, which
directly determines which quantization formats are viable:

| Status | Kernels |
|---|---|
| ✅ Available | `int8_convrot`, `w4a8_int8_linear`, `svdquant_w4a4`, `gemv_awq_w4a16`, `per_tensor_fp8`, `sol_attn`, `na3d`, `fp16_conv3d`, `rms_rope*`, `adaln` |
| ❌ Missing | `dequantize_int8_convrot_weight` (only the `_dtype` variant exists), `dequantize_int8_simple` (same), `rotate_int8_convrot_weight`, `prepare_int4_weight_for_int8_linear` |
| ⚠️ **Emulated** | `nvfp4`, `mxfp8` — they do not error, but run through a software emulation path rather than native kernels |

The runtime log states this distinction explicitly (from the §6.4 run):

```
Native ops:   float8_e5m2, int8_tensorwise, asym_w4a8_int8, convrot_w4a4, float8_e4m3fn
emulated ops: nvfp4, mxfp8
```

`int8_convrot` numerical check: `relative_rmse = 0.0126`, `finite = true`, output `bf16`.
**Accuracy is acceptable.**

**Implication: when recommending a quantization format for RDNA4, `int8_convrot` and GGUF are
safe choices. `nvfp4` / `mxfp8` will run but are emulated — they won't crash, but they won't
deliver the speedup quantization is supposed to buy. Say so explicitly when advising.**

---

## 2. Decision tree

```
Chat model?
├── < 10 GiB, just needs to work         → Ollama + Vulkan   (§3.1)
└── > 10 GiB, or long context            → llama.cpp ROCm    (§3.2)
                                             MUST quantize KV cache
   ├── wants a web UI                    → Open WebUI :8080  (§8.1)
   └── wants a roleplay UI               → SillyTavern :8000 (§8.2)

Text-to-image?
├── SDXL / RealVisXL                     → §7  ⚠️ needs NO RDNA4 workarounds
│                                              use this as the troubleshooting baseline
└── Qwen-Image 2.1                       → §5
    ├── image gen only, nothing else      → A: Q8_0 GGUF
    │                                        warm 40 s, 14 GB resident, only 2.4 GB left
    └── must coexist / video runs next    → B: int8_convrot
                                             warm 97 s, 6 GB resident, 10.5 GB left

Text-to-video (MiniMax H3)?
└── must own the whole card + 2 patches  → §6
    text encoder is 25.3 GiB > the safetensors 0.8.0 Windows crash threshold

Which RDNA4 flags to add?
├── mature models like SDXL              → defaults, add nothing
└── Qwen-Image 2.1 / H3                  → the §4.2 set
    never enable TunableOp, under any circumstances (§10)
```

---

## 3. Chat models

### 3.1 Ollama + Vulkan

**RDNA4 was not on Ollama's ROCm support list at the time — Vulkan is the stable path.**
Use the official portable build.

```bat
set OLLAMA_MODELS=E:\AI\Models\Ollama
set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_CONTEXT_LENGTH=8192
set OLLAMA_VULKAN=1
"E:\AI\Apps\Ollama\ollama.exe" serve
```

Measured — Qwen3.5 9B Q4 (6.6 GB):

| Metric | Value |
|---|---|
| GPU offload | **34/34 layers on GPU** (Vulkan0) |
| Model VRAM buffer | ~4717 MiB |
| Host side | ~546 MiB |
| **Generation** | **~24 tok/s** |
| Prompt processing | 250–340 tok/s |
| Cold start | ~15 s |

Modelfile (Qwen3.5 recommended sampling, thinking mode off):

```
FROM <blob path>
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER presence_penalty 1.5
```

### 3.2 llama.cpp ROCm

Use the prebuilt `gfx120X` package. It bundles `amdhip64_7.dll`, `hipblaslt` and
`ggml-hip.dll`, so **a system-wide ROCm SDK install is not required**.

27B (Q3_K, 12.6 GiB of weights), 16K context:

```bat
llama-server.exe -m "<27B-Q3_K.gguf>" --alias huihui-q3-rocm ^
  -ngl 99 -c 16384 -fa on -ctk q4_0 -ctv q4_0 --jinja ^
  --host 0.0.0.0 --port 8081
```

12B (Q6_K, 9.4 GiB of weights), 32K context:

```bat
llama-server.exe -m "<12B-Q6_K.gguf>" --alias mag-mell-12b ^
  -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja ^
  --host 0.0.0.0 --port 8081
```

> **KV cache quantization is the deciding factor on a 16 GB card.**
> The 27B Q3_K weights alone are 12.6 GiB; an fp16 KV cache at 16K context does not fit —
> `-ctk q4_0 -ctv q4_0` is what makes it fit. The 12B has smaller weights, so it gets
> 32K context and the higher-quality `q8_0`.
> **Rule: compute `weights + KV cache` first, then choose `-c` and `-ctk/-ctv`.**

### 3.3 VRAM mutual exclusion (important)

**Ollama and llama.cpp cannot coexist on one 16 GB card.** Every launcher clears the field first:

```powershell
try{ (Invoke-RestMethod http://127.0.0.1:11434/api/ps -TimeoutSec 3).models.name |
     % { & 'E:\AI\Apps\Ollama\ollama.exe' stop $_ } }catch{}
Get-Process llama-server -EA 0 | ? Path -like '*llama.cpp-rocm*' | Stop-Process -Force
for($i=0;$i -lt 10 -and (Get-Process llama-server -EA 0);$i++){ Start-Sleep 2 }
```

---

## 4. ComfyUI shared configuration

### 4.1 Caches off the system drive (all workflows)

**Nothing lands on the system drive** — this matters when re-downloading 20 GiB+ weights
repeatedly on a 16 GB card.

```bat
set TEMP=E:\AI\Temp
set TMP=E:\AI\Temp
set PIP_CACHE_DIR=E:\AI\Cache\pip
set HF_HOME=E:\AI\Cache\huggingface
set TORCH_HOME=E:\AI\Cache\torch
set XDG_CACHE_HOME=E:\AI\Cache
set MIOPEN_USER_DB_PATH=E:\AI\Cache\miopen
set MIOPEN_CUSTOM_CACHE_DIR=E:\AI\Cache\miopen
set PYTHONNOUSERSITE=1
set HF_HUB_DISABLE_TELEMETRY=1
```

### 4.2 RDNA4 workarounds are **per-workflow** — do not apply them blanket

⚠️ **This is the easiest thing to get wrong in this document.** The set below is only needed
by the **large-model workflows**. SDXL-class models run on ComfyUI defaults; adding these
buys nothing.

| Workflow | RDNA4 env vars | `--disable-pinned-memory` | `--use-pytorch-cross-attention` |
|---|---|---|---|
| **SDXL** (RealVisXL etc.) | ❌ not needed | ❌ not needed | ❌ not needed |
| **Qwen-Image 2.1** | ✅ required | ✅ required | ✅ required |
| **MiniMax H3** | ✅ required | ✅ required | ✅ required + exclusive flags |

Only large-model workflows get this block:

```bat
rem ==== Qwen-Image 2.1 / MiniMax H3 only ====
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
set MIOPEN_FIND_MODE=FAST
rem TunableOp deliberately removed — see §10
```

```
--disable-pinned-memory            pinned host memory allocation fails on ROCm
--use-pytorch-cross-attention      no triton — fall back to PyTorch native SDPA
```

Recommended everywhere:

```
--disable-api-nodes                fully local, no external API calls
--extra-model-paths-config <yaml>  per-project model path table; instances stay isolated
```

### 4.3 All model directories are external

**Not a single weight lives under `ComfyUI/models/`** — it contains only the upstream
`put_*_here` placeholder files. Everything is resolved through `extra_model_paths.yaml`
pointing at a separate data drive:

```yaml
ai_models:
  base_path: E:/AI/Models/QwenImage21
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae

sdxl:
  base_path: E:/AI/Models/Image
  checkpoints: Stable-diffusion
  loras: Lora
  vae: VAE
  controlnet: ControlNet
  upscale_models: Upscale
```

**Why this matters**: ComfyUI can be `git pull`ed or deleted and reinstalled outright without
touching a single byte of weights. Per-project `--extra-model-paths-config` then layers on
top of this global config.

---

## 5. Qwen-Image 2.1 text-to-image

### 5.1 Pinned weights and verification

**Use `resolve/<commit>/`, not `resolve/main/`** — main drifts, and benchmarks stop being
reproducible immediately.

Source: `huggingface.co/Comfy-Org/Qwen-Image-2.1` @ `9a44dbdb47cefd046be9c0a13476192f34c8db8e`

| File | Bytes | SHA256 (first 16) | Verified |
|---|---|---|---|
| `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | 7,256,783,064 | `cb74113cb03faecd` | ✅ |
| `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 9,350,798,360 | `8bfd0f6e12abf2d2` | ✅ |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | `bb21f7473051e1ac` | ✅ |

### 5.2 A/B benchmark (1024×1024, 20 steps)

**The most useful data here. On one card the two routes are a direct speed-vs-VRAM trade:**

| Route | Diffusion model | Cold | **Warm** | **Resident VRAM** | Peak allocated | **Free after** |
|---|---|---|---|---|---|---|
| **A** | `Q8_0` GGUF (7.6 GiB) | 294.7 s | **40.4 s** | 14.03 GB | 15.88 GiB | **2.38 GB** |
| **B** | `int8_convrot` (7.3 GiB) | 155.9 s | **97.4 s** | **6.02 GB** | 15.82 GiB | **10.49 GB** |

Both routes use `Qwen3VL-8B-Instruct-Q4_K_M.gguf` as the text encoder and
`qwen_image_2.1_vae_bf16` as the VAE. seed42 = cold, seed43 = warm. Route B's very first
run, including cold load from disk, took 363.9 s.

**Conclusions (directly usable when advising a user):**

1. **A is 2.4× faster warm**, but leaves only 2.38 GB free — the card is full, and anything
   else started alongside it will fail.
2. **B holds only 6 GB resident**, leaving 10.5 GB of scheduling headroom, at the cost of
   being roughly twice as slow.
3. **Both peak at ~15.9 GiB, i.e. the card's physical limit.** So 1024×1024 is the ceiling on
   16 GB without extra offload — **if a user wants higher resolution, discuss offload or a
   lower resolution first; do not simply say yes.**

### 5.3 Custom node · GGUF text encoder (restoring the vision tower)

**Problem**: GGUF quants of Qwen3-VL **contain only the language tower, not the vision tower**.
Loading one through `CLIPLoaderGGUF` leaves the vision weights uninitialized and random —
images still come out, but Qwen3-VL's image understanding is entirely broken.
**This is a silent failure whose only symptom is "quality is subtly worse", and it is
extremely hard to debug.**

**Fix**: take the language tower from the GGUF, the vision tower from the original
safetensors, merge, then hand off to ComfyUI's own patcher.

```python
class QwenImage21GGUFEncoder:
    def load_clip(self, clip_name, vision_source):
        loader = nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF']()
        path = folder_paths.get_full_path_or_raise('clip_gguf', clip_name)
        sd = loader.load_data([path])[0]

        # restore the vision tower from the original safetensors
        source = folder_paths.get_full_path_or_raise('text_encoders', vision_source)
        with safe_open(source, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.startswith('model.visual.'):
                    sd[key] = f.get_tensor(key)

        # assert rather than silently produce bad images
        if 'model.visual.deepstack_merger_list.0.norm.weight' not in sd:
            raise ValueError('vision_source must contain the original Qwen3-VL vision tower.')

        clip = loader.load_patcher([path], comfy.sd.CLIPType.QWEN_IMAGE, [sd])
        if not isinstance(clip.tokenizer,
                          comfy.text_encoders.qwen_image21.QwenImage21Tokenizer):
            raise RuntimeError('Expected QwenImage21Tokenizer; refusing a different encoding path.')
        return (clip,)
```

**Payoff**: a 5.0 GiB Q4_K_M language tower plus the original vision tower replaces the
9.4 GiB full-precision encoder — 4.4 GiB saved. Both assertions are deliberate: silent
degradation is far harder to diagnose than a crash.

### 5.4 Custom node · deterministic VRAM scheduler

**Problem**: ComfyUI's `mm.free_memory()` **only frees the shortfall**. The result is that the
diffusion model gets *partially* unloaded and patched into lowvram mode, with a cliff-edge
performance drop.

**Fix**: pass an absurd memory requirement to force a **complete** unload.

```python
def _evict_other_gpu_models(clip_patcher):
    # 1e30 bypasses ComfyUI's "free only the shortfall" partial-unload behavior
    device = clip_patcher.load_device
    keep = _loaded_entries_for(clip_patcher)
    with torch.inference_mode():
        unloaded = mm.free_memory(1e30, device, keep_loaded=keep)
        if unloaded:
            mm.soft_empty_cache()
```

Scheduling order:

```
prompt / reference image changed:
  ① fully evict diffusion → ② call the official TextEncodeQwenImage21
  → ③ fully unload the text encoder → ④ let the sampler load diffusion
seed only changed:
  ComfyUI hits this node's cache — zero model swaps
```

**The key design point**: the prompt and reference image must be inputs of **this node itself**.
That is what makes "prompt changed" invalidate the node and trigger pre-eviction, while
"seed only changed" skips the node entirely via cache, letting diffusion stay resident.
**Scheduling via ComfyUI's cache granularity, rather than a hand-written state machine.**

Two API gotchas:

- The official node takes `images={"image_1": image_1}`, **a single dict** — not an `image_1=`
  keyword argument.
- Newer ComfyUI returns `io.NodeOutput`; read results from `.args`:

```python
if hasattr(out, "args"):
    positive, negative, latent = out.args[:3]
else:
    positive, negative, latent = tuple(out)[:3]
```

### 5.5 Port mutual-exclusion guard

Two ComfyUI instances do not fit in 16 GB. Launchers check the port first:

```bat
netstat -ano | findstr /R /C:"127.0.0.1:8190 .*LISTENING" >nul
if not errorlevel 1 (
    echo [info] 8190 already serving ComfyUI - just opening the page.
    start "" http://127.0.0.1:8190
    exit /b
)
```

---

## 6. MiniMax H3 text-to-video

**The hardest one. 53 GiB of weights, including a single 25.3 GiB file — 1.5× the card's VRAM.**

Weights live on a separate mechanical drive, reached via `--extra-model-paths-config`:

| File | Size |
|---|---|
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 25.3 GiB |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.5 GiB |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 4.9 GiB |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 577 MiB |

### 6.1 Required patch A: safetensors 0.8.0 crashes on Windows

**An upstream bug, undocumented anywhere official, worth recording on its own:**

> `safetensors 0.8.0`'s `safe_open(framework="pt")` triggers an access violation
> `0xC0000005` on `get_tensor()` for **any file larger than roughly 21 GiB** on Windows,
> killing the entire ComfyUI process. Bisected with synthetic files: **20 GiB works,
> 23 GiB crashes.** H3's text encoder is 25.3 GiB, so `CLIPLoader` is guaranteed to die.

**Fix**: swap only the reference held by `comfy.utils`. Files above the threshold go through a
read-only mmap plus `torch.frombuffer`; everything else still uses the real
`safetensors.safe_open`. Read-only file-backed pages do not add commit charge.

```python
LIMIT = 16 * 1024 ** 3

class MMapSafeOpen:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        header_size = struct.unpack("<Q", self.mm[:8])[0]
        self.header = json.loads(self.mm[8:8 + header_size])
        self.meta = self.header.pop("__metadata__", None)
        self.base = 8 + header_size
        self.mv = memoryview(self.mm)

    def __exit__(self, *exc):
        return False        # tensors hold views into the mmap - it must stay open

    def get_tensor(self, name):
        info = self.header[name]
        start, end = info["data_offsets"]
        dtype = comfy.utils._TYPES[info["dtype"]]
        if start == end:
            return torch.empty(info["shape"], dtype=dtype)
        return torch.frombuffer(
            self.mv[self.base + start:self.base + end], dtype=dtype
        ).view(info["shape"])

def safe_open(path, framework="pt", device="cpu"):
    if framework == "pt" and device == "cpu" and os.path.getsize(path) > LIMIT:
        return MMapSafeOpen(path)
    return safetensors.safe_open(path, framework=framework, device=device)

comfy.utils.safetensors = types.SimpleNamespace(
    safe_open=safe_open, torch=safetensors.torch)
```

**Note that `__exit__` must return without closing the mmap** — the tensors are buffer views,
and closing it leaves dangling pointers.

### 6.2 Required patch B: empty the card before VAE decode

On a 16 GB RDNA4 card, leaving the diffusion model, text encoder and both VAEs co-resident
after sampling produces **co-residency slowdown and HIP launch failures**
(matching ComfyUI issue #15484).

**Fix**: insert a gate node between `SamplerCustomAdvanced` and the two VAE decodes. The
latent must **flow through** the node, which is what guarantees it runs after sampling and
before decoding.

```python
class H3UnloadBeforeDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}
    RETURN_TYPES = ("LATENT",)      # latent flows through = enforced ordering

    def unload(self, latent):
        device = mm.get_torch_device()
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        return (latent,)
```

### 6.3 Launch flags

H3 **must own the whole card**. The launcher warns if 8188 or 8190 are in use:

```bat
venv\Scripts\python.exe main.py ^
  --disable-dynamic-vram --reserve-vram 1 ^
  --disable-pinned-memory --use-pytorch-cross-attention ^
  --listen 127.0.0.1 --port 8192 --disable-api-nodes ^
  --extra-model-paths-config E:\AI\Projects\MiniMaxH3\model_paths.yaml ^
  --auto-launch
```

`--disable-dynamic-vram --reserve-vram 1` is H3-specific; other workflows do not need it.

### 6.4 Measured benchmark

**One successful run: 10 steps, 277.87 seconds total.** Taken from the run log, not extrapolated.

Machine resources: `Total VRAM 16304 MB, total RAM 32683 MB`, `vram state = NORMAL_VRAM`.

#### Model loading: both partial, neither degraded to lowvram

| Model | Usable | Loaded to GPU | Offloaded to CPU | Buffer reserved | **lowvram patches** |
|---|---|---|---|---|---|
| Text encoder (Qwen3-VL 32B, 25.3 GiB) | 14293.74 MB | 13418.44 MB | **12465.00 MB** | 875.29 MB | **0** ✅ |
| Diffusion model (19.5 GiB) | 13866.61 MB | 13386.64 MB | **6609.51 MB** | 661.52 MB | **0** ✅ |

> **`lowvram patches: 0` is the single most important line here.**
> The models were genuinely split (about 19 GB offloaded to CPU in total), but
> **ComfyUI's lowvram degradation path was never triggered** — meaning the cliff-edge
> slowdown described in §5.4 did not happen on the H3 route.
> When troubleshooting, grep for this line first: anything non-zero means you fell into
> the degraded path.

Also note: the 25.3 GiB encoder needs 12.5 GB offloaded to CPU, and this machine has only
32 GB of RAM. **This is the second value of the §6.1 mmap approach beyond avoiding the
crash** — read-only file-backed pages don't count toward commit charge, or 32 GB would be
uncomfortably tight too.

#### Sampling: 10 steps in 55 s

```
 1/10  13.41 s/it   <- first step includes kernel compilation
 2/10   8.82 s/it
 3/10   7.81 s/it
 4/10   7.30 s/it
 5/10   6.53 s/it
 6/10   5.57 s/it
 7/10   5.00 s/it
 8/10   4.45 s/it
 9/10   4.14 s/it
10/10   3.98 s/it   <- converged
─────────────────────
55 s total, 5.52 s/it average
```

**First step 13.41 s, last step 3.98 s — a 3.4× spread.** The first step includes MIOpen
kernel selection and the first compile of the AOTriton attention kernel, after which it
converges monotonically to about 4 s/it.
**Always evaluate H3 performance on the converged value; the first step or the average both
understate this card.**

The log confirms `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is actually taking effect:

```
UserWarning: Using AOTriton backend for Efficient Attention forward...
(aten/src/ATen/native/transformers/hip/attention.hip:1452)
```

#### Where the time goes

| Stage | Time | Share |
|---|---|---|
| Model load + text encode (53 GiB read from a mechanical drive + quant unpacking) | ~215 s | **77%** |
| Sampling (10 steps) | 55 s | 20% |
| Dual VAE decode | ~8 s | 3% |
| **Total** | **277.87 s** | |

> **The bottleneck is loading, not compute.** Sampling is only 20% of wall time.
> To speed up H3, move the weights to an SSD or avoid reloading — not tune sampler settings.
> (On this machine H3's weights are on a mechanical drive; `fast_disk=False` in the log is
> exactly that.)

#### VAE decode stage

```
[H3-VRAM-Gate] free VRAM 15359 -> 15357 MB
Requested to load MiniMaxH3AudioVAE  -> 577.08 MB, full load: True
Requested to load MiniMaxH3VideoVAE  -> 4966.19 MB, full load: True
```

`model_type FLOW_AV` — a joint audio-video flow model, one VAE each, both fully loaded.

⚠️ **Honest caveat**: in this run the gate node freed only 2 MB (15359 → 15357), meaning
ComfyUI had already evicted the models by the end of sampling.
**So this log does not prove the gate node "saved" this run** — here it acted as insurance
rather than a necessity. Its value is guaranteeing deterministic ordering instead of relying
on ComfyUI happening to get it right.

### 6.5 Open issue: audio branch fails on `audio_scale`

A second run failed at 148.23 seconds:

```
!!! Exception during processing !!!
AttributeError: 'ModelSamplingAdvanced' object has no attribute 'audio_scale'
```

`model_type` is `FLOW_AV` (joint audio-video), so the sampling node must carry an
`audio_scale` attribute, which `ModelSamplingAdvanced` does not have. **Switching to an
AV-capable sampling node should fix it, but which node is correct has not been verified on
this machine, so the audio branch is currently not working.**

> The video branch (the 277.87 s run) succeeded completely. This error only affects audio output.
> If you hit the same error on H3, **it is not an RDNA4 problem** — it is a node selection
> issue, unrelated to the GPU, and will reproduce on NVIDIA too.

---

## 7. SDXL text-to-image (dual frontend: ComfyUI / ForgeNeo)

**The contrast with the two large models above: SDXL needs no workarounds on RDNA4.**
This is the baseline for deciding whether an error is RDNA4's fault or the model's.

Model: `RealVisXL_V5.0_fp16.safetensors` (6.9 GiB, SDXL architecture)

### 7.1 ComfyUI route

Much cleaner launch flags — **no** RDNA4 env vars, **no** `--disable-pinned-memory`,
**no** `--use-pytorch-cross-attention`:

```bat
venv\Scripts\python.exe main.py ^
  --listen 127.0.0.1 --port 8188 --disable-api-nodes ^
  --output-directory E:\AI\Data\Images\SDXL ^
  --temp-directory E:\AI\Temp\ComfyUI ^
  --user-directory E:\AI\Data\ComfyUI ^
  --auto-launch
```

Models resolve through the `sdxl` block of `extra_model_paths.yaml` in §4.3 — no copying into
the ComfyUI tree.

### 7.2 ForgeNeo route (same weights)

Leave `COMMANDLINE_ARGS` **empty**; the defaults work on RDNA4 as-is:

```bat
set COMMANDLINE_ARGS=
:: all tried, none needed for SDXL - kept commented for reference
:: --xformers --sage --uv
:: --pin-shared-memory --cuda-malloc --cuda-stream
call webui.bat
```

`config.json` only redirects output, so both frontends write to the same data drive:

```json
{
  "outdir_txt2img_samples": "E:\\AI\\Data\\Images\\txt2img",
  "outdir_img2img_samples": "E:\\AI\\Data\\Images\\img2img",
  "outdir_txt2img_grids":   "E:\\AI\\Data\\Images\\grids",
  "outdir_img2img_grids":   "E:\\AI\\Data\\Images\\grids",
  "outdir_extras_samples":  "E:\\AI\\Data\\Images\\extras",
  "samples_save": true
}
```

> **Implication (directly usable for troubleshooting)**: if a user hits a problem running SDXL
> on RDNA4, **do not start by looking at ROCm workarounds** — this path is clean, so the
> problem is more likely the model file, the VAE, or frontend configuration. Conversely, for
> Qwen-Image 2.1 / H3 problems, check the §9 table first.

---

## 8. Chat frontends

Both are strictly loopback and not exposed externally.

### 8.1 Open WebUI (talks to Ollama, port 8080)

```bat
set "DATA_DIR=%ROOT%data"
set "OLLAMA_BASE_URL=http://127.0.0.1:11434"
set "ENABLE_OPENAI_API=False"     rem no external APIs, fully local
set "WEBUI_AUTH=False"            rem acceptable ONLY because it listens on 127.0.0.1
set "HF_ENDPOINT=https://hf-mirror.com"   rem regional mirror for embedding models
set "PYTHONUTF8=1"

venv\Scripts\open-webui.exe serve --host 127.0.0.1 --port 8080
```

⚠️ **`WEBUI_AUTH=False` is only acceptable with strict loopback binding.**
The moment `--host` becomes `0.0.0.0` or a port forward is added, set it back to `True`, or
anyone on the LAN can use your models directly.

The launcher does three useful things:

**① Clears the field before starting**, and **reads a Windows performance counter to print
actual VRAM usage** — a practical substitute for the missing `nvidia-smi` on AMD:

```powershell
$v = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples |
     ? CookedValue -gt 0 | select -First 1
'VRAM in use: {0:N2} GB' -f ($v.CookedValue/1GB)
```

**② Starts Ollama only if it isn't already running:**

```bat
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul ^
  || start "" /B "E:\AI\Apps\Ollama\ollama.exe" serve
```

**③ Polls `/health` before opening the browser**, so you never land on an error page:

```powershell
for($i=0;$i -lt 120;$i++){
  try{ Invoke-WebRequest http://127.0.0.1:8080/health -UseBasicParsing -TimeoutSec 2 | Out-Null
       Start-Process http://127.0.0.1:8080; break }
  catch{ Start-Sleep 2 }
}
```

### 8.2 SillyTavern (talks to llama.cpp's OpenAI-compatible API, port 8000)

A Node.js app pointed at the OpenAI-compatible API exposed by the `llama-server` instances
in §3.2 (`:8081`).

Key `config.yaml` entries:

```yaml
port: 8000
listen: true             # used together with whitelistMode
whitelistMode: true      # only whitelisted IPs can connect - this is the main defense
basicAuthMode: false
securityOverride: false  # keep false; it bypasses the whitelist check
enableCorsProxy: false
dataRoot: ./data
```

> **`listen: true` + `whitelistMode: true` is a safe combination.** Turning off
> `whitelistMode` or turning on `securityOverride` exposes character cards and chat history
> to the entire LAN. Think carefully before changing either.

Launch: `Start.bat` (runs `npm install --omit=dev --ignore-scripts`, then `node server.js`).

### 8.3 Port map

| Port | Service | Binding |
|---|---|---|
| 11434 | Ollama API | 127.0.0.1 |
| 8081 | llama.cpp `llama-server` (OpenAI-compatible) | 0.0.0.0 ⚠️ |
| 8080 | Open WebUI | 127.0.0.1 |
| 8000 | SillyTavern | whitelist mode |
| 8188 | ComfyUI main / SDXL | 127.0.0.1 |
| 8190 | ComfyUI Qwen 2.1 A/B | 127.0.0.1 |
| 8192 | ComfyUI MiniMax H3 video | 127.0.0.1 |
| 7860 | ForgeNeo (default) | 127.0.0.1 |

⚠️ `llama-server` uses `--host 0.0.0.0`, so it is **LAN-reachable with no authentication**.
Change it to `--host 127.0.0.1` if you don't need access from other devices.

---

## 9. Troubleshooting table

Searchable by symptom. **Every row was actually encountered and resolved — none are speculative.**

| Symptom | Root cause | Fix |
|---|---|---|
| First inference hangs, 25 minutes with zero output | TunableOp is broken on Windows + RDNA4 | Remove `PYTORCH_TUNABLEOP_ENABLED`, use `MIOPEN_FIND_MODE=FAST` |
| ComfyUI process vanishes, no traceback, exit `0xC0000005` | `safetensors 0.8.0` access violation reading files > 21 GiB | Apply the §6.1 mmap patch |
| pinned host memory allocation fails | ROCm doesn't support that path | Add `--disable-pinned-memory` |
| ImportError around flash-attn / triton | No triton wheel for Windows ROCm | Add `--use-pytorch-cross-attention` |
| HIP launch failure or extreme slowness at VAE decode | Multiple models co-resident in 16 GB (ComfyUI #15484) | Insert the §6.2 gate node before decode |
| Images generate but quality is subtly worse | GGUF ships only the language tower; vision tower uninitialized | Restore it with the §5.3 encoder |
| Diffusion suddenly slow, log mentions lowvram | ComfyUI freed only the shortfall → partial unload | Force a full unload via §5.4's `free_memory(1e30, ...)` |
| 27B model OOMs after loading | fp16 KV cache doesn't fit | Quantize it: `-ctk q4_0 -ctv q4_0` |
| Second ComfyUI won't start / both crash | 16 GB can't hold two | Add the §5.5 port check to the launcher |
| Ollama and llama.cpp fight over VRAM | No mutual exclusion | Run the §3.3 teardown before starting |
| Benchmarks not reproducible | Weights pulled from `resolve/main/` drifted | Pin to `resolve/<commit>/` |
| SDXL unchanged after adding RDNA4 flags | SDXL never needed them (§4.2) | Remove them; look at the model / VAE / frontend config instead |
| Open WebUI opens straight to an error page | Browser opened before the server was ready | Poll `/health` first (§8.1) |
| Want VRAM usage on Windows but there's no nvidia-smi | No such tool for AMD | Read the counter `\GPU Adapter Memory(*)\Dedicated Usage` (§8.1) |
| H3 raises `'ModelSamplingAdvanced' object has no attribute 'audio_scale'` | `FLOW_AV` needs an AV-capable sampling node | Switch nodes (§6.5). **Unrelated to RDNA4** — reproduces on NVIDIA |
| Need to tell whether a model degraded to lowvram | Partial load ≠ lowvram degradation | grep the log for `lowvram patches:` — non-zero means degraded (§6.4) |
| H3 is slow and sampler tuning does nothing | Loading is the bottleneck; sampling is only 20% | Move weights to an SSD; `fast_disk=False` in the log flags a slow disk (§6.4) |

---

## 10. Disproven options

**Things that look like they should be enabled but must not be. Worth checking before advising.**

| Option | Status | Evidence |
|---|---|---|
| `PYTORCH_TUNABLEOP_ENABLED=1` | ❌ **Disable** | Hangs on Windows + RDNA4; 25 minutes, zero output |
| `triton` backend | ❌ Unavailable | `ImportError: No module named 'triton'` — no Windows ROCm wheel |
| pinned memory (on by default) | ❌ **Must disable** | Allocation fails; requires `--disable-pinned-memory` |
| `nvfp4` / `mxfp8` quantization | ⚠️ Emulated | Runs, but through software emulation — no native kernels, no speedup |
| Ollama on ROCm | ⚠️ Unsupported at the time | RDNA4 not on the support list; **Vulkan is the stable path**, 34/34 layers offloaded |
| Above 1024×1024 without offload | ❌ Hits the wall | Peak already at 15.9 GiB = physical limit |
| Two ComfyUI instances | ❌ Impossible | 16 GB isn't enough |
| RDNA4 workarounds for SDXL | ⚠️ **Unnecessary** | Defaults work; adding them buys nothing (§7) |
| ForgeNeo's `--xformers` / `--cuda-malloc` etc. | ⚠️ Not used | An empty `COMMANDLINE_ARGS` works fine |

---

## 11. Directory layout

```
E:\AI\Apps
   ComfyUI/          three instances share one checkout, separated by --port / --extra-model-paths-config
   ForgeNeo/         second SDXL frontend, shares the same weights as ComfyUI
   Ollama/           portable build + custom Modelfiles
   llama.cpp-rocm/   prebuilt gfx120X package, bundles its own HIP runtime
   OpenWebUI/        venv + data
   SillyTavern/      Node.js
E:\AI\Models
   gguf/             chat models for llama.cpp
   Image/            SDXL: Stable-diffusion/ Lora/ VAE/ ControlNet/ Upscale/
   Ollama/           blobs/ manifests/ metadata/
   QwenImage21/      diffusion_models/ text_encoders/ vae/ loras/
E:\AI\Cache          huggingface/ miopen/ pip/ torch/   <- all off-drive, nothing on the system disk
E:\AI\Projects       one subdirectory per model: workflow + benchmark + environment-audit
E:\AI\Data           Images/ Videos/ ComfyUI/ (user directory)
E:\AI\Logs
F:\AI\Models\MiniMaxH3   H3 weights, 53 GiB, on a separate mechanical drive
```

**One ComfyUI checkout runs three instances** (SDXL :8188 / Qwen A-B :8190 / H3 :8192),
isolated by three things: `--port`, `--temp-directory` and `--extra-model-paths-config`.
They share `--user-directory`, so node layout and UI settings are shared.

Each project directory always holds these three files — this is why the benchmarks are reproducible:

- `environment-audit.json` — package versions + GPU properties + each weight's SHA256 against
  the official value + the ComfyUI git commit
- `kernel-probe.json` — the actual per-backend kernel list + int8 numerical check
- `*.result.json` — per-run wall time + full VRAM metrics

---

## Scope and limits

**Verified**: every entry in the deployment inventory, measured in the §1 baseline environment.

**Not verified / not guaranteed**:

- **The stable-diffusion.cpp Vulkan route** — a comparison benchmark script exists, but
  `sd-cli.exe` was never actually installed, so **this document has no Vulkan-vs-ROCm image
  generation data**. Do not extrapolate the §3.1 Ollama Vulkan result to image generation.
- Other RDNA4 parts (9070 non-XT, 9060, etc.) — different VRAM and CU counts, so benchmarks
  don't transfer; the environment-variable conclusions most likely still do
- RDNA3 (gfx110x) — the kernel capability table needs re-measuring
- Linux ROCm — triton is available there, so several conclusions here
  (`--use-pytorch-cross-attention`, TunableOp) do not apply
- WSL2 — untested
- Running multiple models **concurrently** — every setup here is mutually exclusive;
  co-residency appears only as a failure case in §6.2

**License**: the documentation and custom node code in this repository are MIT. Third-party
model weights remain under their own licenses and are not redistributed here.

---

## If this saved you time

These parameters were found by repeatedly hitting walls on a single 16 GB card — the
25 minutes lost to TunableOp, the `0xC0000005` from safetensors, the silent quality loss from
a GGUF missing its vision tower. Each one took a while to pin down.

**If it saved you a few hours, please leave a Star ⭐** — it helps other RDNA4 users find it.

Found new problems, or different results on another RDNA4 / RDNA3 card? Issues and PRs
welcome. Especially valuable: benchmarks from other RDNA4 parts, Linux ROCm comparison data,
and any upstream progress on the safetensors bug.
