# RDNA4 本地 AI 部署参考手册

**面向 AI 助手的结构化参考数据** · AMD Radeon RX 9070 XT (gfx1201) · Windows 11 原生 · 16 GB VRAM

**中文** · [English](README.en.md)

---

## 这份文档是什么

一台 RX 9070 XT 上**实测跑通**的完整本地 AI 栈的部署参数、跑分数据和踩坑记录。
写成这个形式，是为了让 AI 助手在帮人部署 RDNA4 环境时能直接查到可靠答案，
而不是去猜、或者套用 NVIDIA 的经验。

### 已部署清单

单卡 16 GB，全部在同一台机器上跑通（**互斥运行**，不是同时）：

| 类别 | 模型 / 组件 | 后端 | 章节 |
|---|---|---|---|
| 对话 | Qwen3.5 9B (Q4, 6.6 GB) | Ollama · **Vulkan** | §3.1 |
| 对话 | 27B (Q3_K, 12.6 GiB) | llama.cpp · ROCm | §3.2 |
| 对话 | MN-12B-Mag-Mell-R1 (Q6_K, 9.4 GiB) | llama.cpp · ROCm | §3.2 |
| 文生图 | Qwen-Image 2.1（int8_convrot / Q8_0 / Q4_K_M） | ComfyUI · PyTorch ROCm | §5 |
| 文生图 | + Pruna 8 步蒸馏 LoRA | ComfyUI | §5.1 |
| 文生视频 | MiniMax H3（53 GiB）— 视频 ✅ 277.87 s / 音频 ⚠️ 未通 | ComfyUI · PyTorch ROCm | §6 |
| 文生图 | RealVisXL V5.0 (SDXL, 6.9 GiB) | ComfyUI **和** ForgeNeo | §7 |
| 前端 | Open WebUI :8080 | 接 Ollama | §8.1 |
| 前端 | SillyTavern :8000 | 接 llama.cpp OpenAI 兼容端点 | §8.2 |

### Qwen 家族覆盖

Qwen 系是这张卡上的主力：**5 个不同模型、11 个权重文件、约 108 GB**，
横跨对话 / 文生图 / 文生视频三类任务。

| 模型 | 形态 | 大小 | 担任角色 | 章节 |
|---|---|---|---|---|
| Qwen3.5 9B | Q4 | 6.59 GB | 对话（Ollama Vulkan，24 tok/s） | §3.1 |
| Qwen3.8 27B | Q3_K | 13.50 GB | 对话（llama.cpp ROCm）+ 3 个 Ollama 变体 | §3.2 |
| Qwen-Image 2.1 | `int8_convrot` | 7.26 GB | 扩散主体 —— 省显存路线（稳态 6 GB） | §5.2 B |
| Qwen-Image 2.1 | `Q8_0` GGUF | 7.64 GB | 扩散主体 —— 快速路线（热跑 40 s） | §5.2 A |
| Qwen-Image 2.1 | `Q4_K_M` GGUF | 4.60 GB | 扩散主体 —— 备选 | §5.1 |
| Qwen-Image 2.1 VAE | bf16 | 0.68 GB | 图像解码 | §5.1 |
| └ Pruna 8 步蒸馏 LoRA | — | 0.34 GB | 采样步数压缩 | §5.1 |
| Qwen3-VL 8B | `int8_convrot` | 9.35 GB | 文本编码器 —— **视觉塔来源** | §5.3 |
| Qwen3-VL 8B Instruct | `Q4_K_M` GGUF | 5.03 GB | 文本编码器 —— 语言塔来源 | §5.3 |
| **Qwen3-VL 32B** | `int8_convrot` | **27.14 GB** | **H3 的文本编码器，全机最大单文件** | §6 |
| Qwen3-VL 32B（第二变体） | `int8_convrot` | 26.36 GB | H3 备用编码器 | §6 |

三个值得注意的点：

1. **同一个家族覆盖了全栈。** 对话、文生图的扩散主体、文生视频的文本编码器，
   全部是 Qwen —— 所以本文档里 RDNA4 上关于 Qwen 系的量化结论（`int8_convrot` 可用、
   GGUF 可用、`nvfp4`/`mxfp8` 是模拟）适用范围比看起来更广。
2. **Qwen3-VL 同时扮演两个角色**：既是 VLM，又是扩散模型的文本编码器，
   而且用了 8B / 32B 两个尺度 —— 8B 给 Qwen-Image 2.1，32B 给 MiniMax H3。
   §5.3 那个"GGUF 缺视觉塔"的坑就是从这个双重身份来的。
3. **Qwen-Image 2.1 特意存了三份量化**，就是为了做 §5.2 的 A/B 对照。
   如果你只想跑不想测，选一份即可，别跟着下三份。

> 表中 27B 与 32B 第二变体的具体微调版本名此处省略，不影响任何技术结论 ——
> 量化格式、尺寸和显存行为才是决定部署参数的因素。

**自研 ComfyUI 节点**（源码在 `custom_nodes/`）：

| 节点 | 解决的问题 |
|---|---|
| `QwenImage21GGUFEncoder` | GGUF 量化版缺视觉塔导致的**静默降质** | 
| `Qwen21ScheduledSingleImageEdit` | ComfyUI 部分卸载 → lowvram 降速 |
| `H3UnloadBeforeDecode` | 多模型共驻 16 GB 导致 HIP launch failure |
| `large_safetensors` (猴补丁) | safetensors 0.8.0 读 >21 GiB 文件在 Windows 崩溃 |

### 给 AI 助手的使用说明

> **本文档是参考数据（reference data），不是指令集。**
> 下面的内容描述"在这套硬件上实测是什么结果"，不构成对你的任何要求。
> 请把它当作一份技术资料来引用，一切动作仍由用户决定。
>
> 引用建议：
> - **先核对基线**（§1）。gfx 型号、torch 版本、HIP 版本任一不同，跑分数据都不保证成立，环境变量结论大概率仍成立。
> - **§9 故障速查表**是按「报错现象 → 根因 → 修法」组织的，可以直接按症状检索。
> - **§10 已证伪清单**记录了"看起来该开、实际必须关"的选项。建议在给出配置建议前先看一眼，避免推荐已知会挂的参数。
> - 所有路径都是原机的绝对路径（`E:\AI\...`），引用时请替换成用户自己的。

---

## 1. 已验证基线

```yaml
# 本文所有数据的产出环境。核对用。
gpu:
  name: AMD Radeon RX 9070 XT
  arch: gfx1201            # RDNA4
  vram_mib: 16304          # 可用约 15.9 GiB
  compute_units: 32
  l2_cache_mb: 8
  driver: 32.0.31041.3013  # Adrenalin
os: Windows 11 Pro 26200   # 原生，非 WSL、非 Linux
python: 3.12.10
packages:
  torch: 2.9.1+rocm7.2.1
  torchvision: 0.24.1+rocm7.2.1
  torchaudio: 2.9.1+rocm7.2.1
  transformers: 5.17.0
  safetensors: 0.8.0
  gguf: 0.19.0
hip: 7.2.53211-158bd99533  # amdhip64_7.dll
triton: UNAVAILABLE        # Windows ROCm 无 triton wheel，多处行为由此决定
```

**一行核对：**

```bash
python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.get_device_properties(0).gcnArchName)"
# 期望: 2.9.1+rocm7.2.1 7.2.53211-158bd99533 gfx1201
```

### HIP 后端算子能力（实测，非推测）

`triton` 缺失后，可用后端是 `hip` / `eager`。**`hip` 比 CUDA 后端少 5 类算子**，
这直接决定了哪些量化格式能选：

| 状态 | 算子 |
|---|---|
| ✅ 可用 | `int8_convrot`、`w4a8_int8_linear`、`svdquant_w4a4`、`gemv_awq_w4a16`、`per_tensor_fp8`、`sol_attn`、`na3d`、`fp16_conv3d`、`rms_rope*`、`adaln` |
| ❌ 缺失 | `dequantize_int8_convrot_weight`（仅有 `_dtype` 变体）、`dequantize_int8_simple`（同）、`rotate_int8_convrot_weight`、`prepare_int4_weight_for_int8_linear` |
| ⚠️ **模拟执行** | `nvfp4`、`mxfp8` —— 不报错，但走软件模拟路径，不是原生算子 |

运行时日志会明确列出这个区分（来自 §6.4 的实测日志）：

```
Native ops:   float8_e5m2, int8_tensorwise, asym_w4a8_int8, convrot_w4a4, float8_e4m3fn
emulated ops: nvfp4, mxfp8
```

`int8_convrot` 数值校验：`relative_rmse = 0.0126`，`finite = true`，输出 `bf16`。**精度可接受。**

**推论：给 RDNA4 推荐量化格式时，`int8_convrot` 和 GGUF 是安全选择。
`nvfp4` / `mxfp8` 能跑但是模拟的 —— 不会崩，但拿不到量化本该带来的加速，属于"看起来能用实际白费"，
给建议时要说清楚。**

---

## 2. 决策树：该走哪条路

```
对话模型？
├── < 10 GiB，只要能用、省事        → Ollama + Vulkan   (§3.1)
└── > 10 GiB，或要长上下文          → llama.cpp ROCm    (§3.2)
                                       必须量化 KV cache
   ├── 要网页界面                   → Open WebUI :8080  (§8.1)
   └── 要角色扮演界面               → SillyTavern :8000 (§8.2)

文生图？
├── SDXL / RealVisXL               → §7  ⚠️ 不需要任何 RDNA4 workaround
│                                      这是排障基准线
└── Qwen-Image 2.1                 → §5
    ├── 只生图，不和别的模型共存     → A: Q8_0 GGUF
    │                                 热跑 40 s，占 14 GB，跑完只剩 2.4 GB
    └── 要共存 / 之后要跑视频        → B: int8_convrot
                                      热跑 97 s，稳态仅 6 GB，剩 10.5 GB

文生视频（MiniMax H3）？
└── 必须独占整卡 + 两个补丁         → §6
    文本编码器 25.3 GiB > safetensors 0.8.0 的 Windows 崩溃阈值

RDNA4 参数怎么加？
├── SDXL 等成熟模型                → 默认参数，什么都不用加
└── Qwen-Image 2.1 / H3            → §4.2 那一套
    任何情况都绝不开 TunableOp（§10）
```

---

## 3. 对话模型

### 3.1 Ollama + Vulkan

**RDNA4 当时不在 Ollama 的 ROCm 支持列表里，走 Vulkan 反而稳定。** 用官方便携版。

```bat
set OLLAMA_MODELS=E:\AI\Models\Ollama
set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_CONTEXT_LENGTH=8192
set OLLAMA_VULKAN=1
"E:\AI\Apps\Ollama\ollama.exe" serve
```

实测 —— Qwen3.5 9B Q4（6.6 GB）：

| 指标 | 数值 |
|---|---|
| GPU 卸载层数 | **34/34 全部上 GPU**（Vulkan0） |
| 模型显存缓冲 | ~4717 MiB |
| 主机端占用 | ~546 MiB |
| **生成速度** | **~24 tok/s** |
| 提示处理 | 250–340 tok/s |
| 冷启动 | ~十几秒 |

Modelfile（Qwen3.5 推荐采样参数 + 关思考模式）：

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

用 `gfx120X` 预编译包，自带 `amdhip64_7.dll` / `hipblaslt` / `ggml-hip.dll`，
**不依赖系统装 ROCm SDK**。

27B（Q3_K，12.6 GiB 本体），16K 上下文：

```bat
llama-server.exe -m "<27B-Q3_K.gguf>" --alias huihui-q3-rocm ^
  -ngl 99 -c 16384 -fa on -ctk q4_0 -ctv q4_0 --jinja ^
  --host 0.0.0.0 --port 8081
```

12B（Q6_K，9.4 GiB 本体），32K 上下文：

```bat
llama-server.exe -m "<12B-Q6_K.gguf>" --alias mag-mell-12b ^
  -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja ^
  --host 0.0.0.0 --port 8081
```

> **量化 KV cache 是 16 GB 卡的胜负手。**
> 27B Q3_K 本体 12.6 GiB，16K 的 fp16 KV cache 绝对放不下 —— `-ctk q4_0 -ctv q4_0` 才塞得进。
> 12B 本体小，所以可以给 32K + 更高质量的 `q8_0`。
> **规则：先算 `本体大小 + KV cache`，再决定 `-c` 和 `-ctk/-ctv`。**

### 3.3 显存互斥（重要）

**一张 16 GB 卡上，Ollama 和 llama.cpp 不能共存。** 每个启动脚本开头都要清场：

```powershell
try{ (Invoke-RestMethod http://127.0.0.1:11434/api/ps -TimeoutSec 3).models.name |
     % { & 'E:\AI\Apps\Ollama\ollama.exe' stop $_ } }catch{}
Get-Process llama-server -EA 0 | ? Path -like '*llama.cpp-rocm*' | Stop-Process -Force
for($i=0;$i -lt 10 -and (Get-Process llama-server -EA 0);$i++){ Start-Sleep 2 }
```

端口分配：`11434` Ollama / `8081` llama.cpp / `8188` 主生图 / `8190` A-B 测试 / `8192` H3 视频。

---

## 4. ComfyUI 公共配置

### 4.1 缓存离盘（所有工作流通用）

**系统盘不落任何东西** —— 在 16 GB 卡上反复下载 20 GiB+ 权重时这很关键。

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

### 4.2 RDNA4 workaround 是**分场景**的，不要一律套用

⚠️ **这是本文档一个容易搞错的点。** 下面这套只有**大模型工作流**需要；
SDXL 这类成熟模型用 ComfyUI 默认参数就能跑，加了反而是白搭。

| 工作流 | RDNA4 环境变量 | `--disable-pinned-memory` | `--use-pytorch-cross-attention` |
|---|---|---|---|
| **SDXL**（RealVisXL 等） | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 |
| **Qwen-Image 2.1** | ✅ 需要 | ✅ 需要 | ✅ 需要 |
| **MiniMax H3** | ✅ 需要 | ✅ 需要 | ✅ 需要 + 独占参数 |

大模型工作流才加这一段：

```bat
rem ==== 仅 Qwen-Image 2.1 / MiniMax H3 需要 ====
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
set MIOPEN_FIND_MODE=FAST
rem TunableOp 已移除，原因见 §10
```

```
--disable-pinned-memory            ROCm 上 pinned host memory 分配会失败
--use-pytorch-cross-attention      无 triton，退回 PyTorch 原生 SDPA
```

全场景都建议加的：

```
--disable-api-nodes                纯本地，不联外部 API
--extra-model-paths-config <yaml>  每个项目一份独立模型路径表，多实例互不干扰
```

### 4.3 模型目录全部外置

`ComfyUI/models/` 下**一个权重都不放**，全是官方的 `put_*_here` 占位文件。
所有模型通过 `extra_model_paths.yaml` 指到独立数据盘：

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

**好处**：ComfyUI 可以随时 `git pull` 或整个删掉重装，权重一个字节都不用动。
项目级的 `--extra-model-paths-config` 再叠加在这个全局配置之上。

---

## 5. Qwen-Image 2.1 文生图

### 5.1 权重固定与校验

**用 `resolve/<commit>/` 而不是 `resolve/main/`** —— main 会漂移，跑分立刻不可复现。

来源：`huggingface.co/Comfy-Org/Qwen-Image-2.1` @ `9a44dbdb47cefd046be9c0a13476192f34c8db8e`

| 文件 | 字节 | SHA256 (前 16) | 校验 |
|---|---|---|---|
| `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | 7,256,783,064 | `cb74113cb03faecd` | ✅ |
| `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 9,350,798,360 | `8bfd0f6e12abf2d2` | ✅ |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | `bb21f7473051e1ac` | ✅ |

### 5.2 A/B 跑分（1024×1024，20 步）

**本文最核心的数据。同一张卡上两条路线是速度与显存的直接取舍：**

| 方案 | 扩散模型 | 冷跑 | **热跑** | **稳态显存** | 峰值分配 | **跑完剩余** |
|---|---|---|---|---|---|---|
| **A** | `Q8_0` GGUF (7.6 GiB) | 294.7 s | **40.4 s** | 14.03 GB | 15.88 GiB | **2.38 GB** |
| **B** | `int8_convrot` (7.3 GiB) | 155.9 s | **97.4 s** | **6.02 GB** | 15.82 GiB | **10.49 GB** |

两方案的文本编码器都是 `Qwen3VL-8B-Instruct-Q4_K_M.gguf`，VAE 都是 `qwen_image_2.1_vae_bf16`。
seed42 = 冷跑，seed43 = 热跑。B 的首次运行含磁盘冷加载为 363.9 s。

**结论（可直接用于给用户建议）：**

1. **A 热跑快 2.4 倍**，但跑完只剩 2.38 GB —— 卡被占满，同时开任何别的模型必崩。
2. **B 稳态只占 6 GB**，留出 10.5 GB 调度余量，代价是慢一倍多。
3. **两者峰值都顶到约 15.9 GiB，即这张卡的物理上限。**
   意味着 1024×1024 就是 16 GB 上不加额外 offload 的天花板 ——
   **用户想要更高分辨率时，必须先谈 offload 或降分辨率，不要直接答"可以"。**

### 5.3 自研节点 · GGUF 文本编码器（补回视觉塔）

**问题**：Qwen3-VL 的 GGUF 量化版**只含语言塔，不含视觉塔**。
直接用 `CLIPLoaderGGUF` 加载，视觉权重是未初始化的随机值 —— 图能出，但 Qwen3-VL 的图像理解全废。
**这是静默失败，表现只是"质量微妙地差"，极难 debug。**

**解法**：GGUF 出语言塔，原始 safetensors 出视觉塔，合并后交给 ComfyUI 官方 patcher。

```python
class QwenImage21GGUFEncoder:
    def load_clip(self, clip_name, vision_source):
        loader = nodes.NODE_CLASS_MAPPINGS['CLIPLoaderGGUF']()
        path = folder_paths.get_full_path_or_raise('clip_gguf', clip_name)
        sd = loader.load_data([path])[0]

        # 从原始 safetensors 补回视觉塔
        source = folder_paths.get_full_path_or_raise('text_encoders', vision_source)
        with safe_open(source, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.startswith('model.visual.'):
                    sd[key] = f.get_tensor(key)

        # 断言式校验：宁可崩，不要静默出坏图
        if 'model.visual.deepstack_merger_list.0.norm.weight' not in sd:
            raise ValueError('vision_source must contain the original Qwen3-VL vision tower.')

        clip = loader.load_patcher([path], comfy.sd.CLIPType.QWEN_IMAGE, [sd])
        if not isinstance(clip.tokenizer,
                          comfy.text_encoders.qwen_image21.QwenImage21Tokenizer):
            raise RuntimeError('Expected QwenImage21Tokenizer; refusing a different encoding path.')
        return (clip,)
```

**收益**：5.0 GiB 的 Q4_K_M 语言塔 + 原始视觉塔，替代 9.4 GiB 全精度编码器，省 4.4 GiB。
两处断言都是故意的 —— 静默降质比崩溃难查得多。

### 5.4 自研节点 · 确定性 VRAM 调度器

**问题**：ComfyUI 默认的 `mm.free_memory()` **只释放差额**。
结果是扩散模型被 *部分* 卸载并打上 lowvram 补丁，性能断崖式下降。

**解法**：传一个荒谬的显存需求值进去，强制**完整**卸载。

```python
def _evict_other_gpu_models(clip_patcher):
    # 传 1e30 绕开 ComfyUI 只释放差额的部分卸载行为
    device = clip_patcher.load_device
    keep = _loaded_entries_for(clip_patcher)
    with torch.inference_mode():
        unloaded = mm.free_memory(1e30, device, keep_loaded=keep)
        if unloaded:
            mm.soft_empty_cache()
```

调度时序：

```
prompt / 参考图变了:
  ① 完整驱逐扩散模型 → ② 调官方 TextEncodeQwenImage21
  → ③ 完整卸载文本编码器 → ④ 采样器加载扩散模型
只改 seed:
  ComfyUI 命中本节点缓存，零次模型交换
```

**关键设计**：prompt 和参考图必须是**这个节点自己的输入**。
这样"改 prompt"才会让本节点缓存失效、触发预驱逐；"只改 seed"则整个节点被缓存跳过，
扩散模型可以一直驻留。**用 ComfyUI 的缓存粒度做调度，而不是手写状态机。**

两个 API 坑：

- 官方节点收的是 `images={"image_1": image_1}` **一个字典**，不是 `image_1=` 关键字参数。
- 新版 ComfyUI 的 `io.NodeOutput` 要从 `.args` 取返回值：

```python
if hasattr(out, "args"):
    positive, negative, latent = out.args[:3]
else:
    positive, negative, latent = tuple(out)[:3]
```

### 5.5 端口互斥保护

16 GB 装不下两份 ComfyUI。启动脚本先查端口，已在跑就只开网页：

```bat
netstat -ano | findstr /R /C:"127.0.0.1:8190 .*LISTENING" >nul
if not errorlevel 1 (
    echo [提示] 8190 已有 ComfyUI 在跑，直接打开网页。
    start "" http://127.0.0.1:8190
    exit /b
)
```

---

## 6. MiniMax H3 文生视频

**最难的一个。53 GiB 权重，其中单个文件 25.3 GiB，超过显存容量的 1.5 倍。**

权重清单（单独放机械盘，用 `--extra-model-paths-config` 指过去）：

| 文件 | 大小 |
|---|---|
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 25.3 GiB |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.5 GiB |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 4.9 GiB |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 577 MiB |

### 6.1 必须打的补丁 A：safetensors 0.8.0 在 Windows 上崩溃

**这是上游 bug，官方文档没有，单独记一笔：**

> `safetensors 0.8.0` 的 `safe_open(framework="pt")` 在 Windows 上对**任何超过约 21 GiB
> 的文件**执行 `get_tensor()` 时触发访问违规 `0xC0000005`，直接杀掉整个 ComfyUI 进程。
> 用合成文件二分验证过：**20 GiB 正常，23 GiB 崩溃**。
> H3 的文本编码器 25.3 GiB，所以 `CLIPLoader` 必挂。

**解法**：只替换 `comfy.utils` 持有的那个引用。超过阈值的文件走只读 mmap + `torch.frombuffer`，
其余照旧走真正的 `safetensors.safe_open`。只读的文件映射页不计入 commit charge。

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
        return False        # 张量持有 mmap 的视图，不能关闭

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

**注意 `__exit__` 必须返回而不关闭 mmap** —— 张量是 buffer 的视图，关了就是悬垂指针。

### 6.2 必须打的补丁 B：VAE 解码前清空整卡

16 GB RDNA4 上，采样完成后若扩散模型、文本编码器、两个 VAE 共存，
会出现**共驻减速和 HIP launch failure**（对应 ComfyUI issue #15484）。

**解法**：在 `SamplerCustomAdvanced` 和两个 VAE decode 之间插一个门控节点。
latent 必须**穿过**这个节点，以此保证它在采样之后、解码之前执行。

```python
class H3UnloadBeforeDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}
    RETURN_TYPES = ("LATENT",)      # latent 穿过本节点 = 强制执行时序

    def unload(self, latent):
        device = mm.get_torch_device()
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        return (latent,)
```

### 6.3 启动参数

H3 **必须独占整张卡**。启动脚本先检查 8188/8190 是否占用并警告：

```bat
venv\Scripts\python.exe main.py ^
  --disable-dynamic-vram --reserve-vram 1 ^
  --disable-pinned-memory --use-pytorch-cross-attention ^
  --listen 127.0.0.1 --port 8192 --disable-api-nodes ^
  --extra-model-paths-config E:\AI\Projects\MiniMaxH3\model_paths.yaml ^
  --auto-launch
```

`--disable-dynamic-vram --reserve-vram 1` 是 H3 专用，其他工作流不需要加。

### 6.4 实测跑分

**单次成功运行：10 步，总计 277.87 秒。** 数据来自运行日志，非推算。

机器可用资源：`Total VRAM 16304 MB, total RAM 32683 MB`，`vram state = NORMAL_VRAM`。

#### 模型加载：两个都是"部分加载"，但都没退化成 lowvram

| 模型 | 可用 | 实际载入 GPU | 卸载到 CPU | 缓冲保留 | **lowvram patches** |
|---|---|---|---|---|---|
| 文本编码器（Qwen3-VL 32B, 25.3 GiB） | 14293.74 MB | 13418.44 MB | **12465.00 MB** | 875.29 MB | **0** ✅ |
| 扩散模型（19.5 GiB） | 13866.61 MB | 13386.64 MB | **6609.51 MB** | 661.52 MB | **0** ✅ |

> **`lowvram patches: 0` 是这里最关键的一行。**
> 模型确实被拆开了（合计约 19 GB 卸到 CPU），但**没有触发 ComfyUI 的 lowvram 降级路径** ——
> 也就是说 §5.4 说的那种断崖式降速在 H3 这条路上没有发生。
> 排障时先 grep 这一行：非 0 就说明落进了降级路径。

另外注意：25.3 GiB 的编码器需要把 12.5 GB 卸到 CPU，而这台机器只有 32 GB RAM。
**这正是 §6.1 的 mmap 方案除了绕过崩溃之外的第二个价值** ——
只读的文件映射页不计入 commit charge，否则 32 GB 内存也会很紧张。

#### 采样：10 步 55 秒

```
 1/10  13.41 s/it   ← 首步含 kernel 编译
 2/10   8.82 s/it
 3/10   7.81 s/it
 4/10   7.30 s/it
 5/10   6.53 s/it
 6/10   5.57 s/it
 7/10   5.00 s/it
 8/10   4.45 s/it
 9/10   4.14 s/it
10/10   3.98 s/it   ← 收敛值
────────────────────
总计 55 s，平均 5.52 s/it
```

**首步 13.41 s、末步 3.98 s，相差 3.4 倍。** 首步包含 MIOpen kernel 选择和
AOTriton 注意力内核的首次编译，之后单调收敛到约 4 s/it。
**评估 H3 性能时必须用收敛值，用首步或平均值都会低估这张卡。**

日志里能看到 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 确实生效了：

```
UserWarning: Using AOTriton backend for Efficient Attention forward...
(aten/src/ATen/native/transformers/hip/attention.hip:1452)
```

#### 时间去向拆解

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 模型加载 + 文本编码（53 GiB 权重从机械盘读入 + 量化解包） | ~215 s | **77%** |
| 采样（10 步） | 55 s | 20% |
| 双 VAE 解码 | ~8 s | 3% |
| **合计** | **277.87 s** | |

> **瓶颈是加载，不是计算。** 采样只占 20%。
> 想加快 H3，方向是把权重挪到 SSD / 减少重复加载，而不是调采样参数。
> （本机 H3 权重在机械盘上，日志里 `fast_disk=False` 就是这个意思。）

#### VAE 解码阶段

```
[H3-VRAM-Gate] free VRAM 15359 -> 15357 MB
Requested to load MiniMaxH3AudioVAE  → 577.08 MB, full load: True
Requested to load MiniMaxH3VideoVAE  → 4966.19 MB, full load: True
```

`model_type FLOW_AV` —— 音视频联合流模型，音频和视频各一个 VAE，都是完整加载。

⚠️ **诚实说明**：这次运行里门控节点只释放了 2 MB（15359 → 15357），
说明采样结束时 ComfyUI 已经自己把模型驱逐干净了。
**所以这条日志不能证明门控节点"救了"这次运行** —— 它在这一次更像是保险而非必需。
它的价值在于保证时序确定：不依赖 ComfyUI 恰好做对。

### 6.5 未解决问题：音频分支报 `audio_scale`

第二次运行在 148.23 秒时失败：

```
!!! Exception during processing !!!
AttributeError: 'ModelSamplingAdvanced' object has no attribute 'audio_scale'
```

`model_type` 是 `FLOW_AV`（音视频联合），采样节点需要携带 `audio_scale` 属性，
而 `ModelSamplingAdvanced` 没有这个属性。**换用支持 AV 的采样节点即可，
但本机尚未验证哪个节点是对的，所以音频分支目前算未跑通。**

> 视频分支（277.87 s 那次）是完整成功的。这个报错只影响音频输出。
> 如果你在 H3 上遇到同样的报错，**不是 RDNA4 的问题** —— 是节点选择问题，
> 和显卡无关，NVIDIA 上同样会报。

---

## 7. SDXL 文生图（ComfyUI / ForgeNeo 双前端）

**和上面两个大模型形成对照：SDXL 在 RDNA4 上不需要任何 workaround。**
这是判断"某个报错是 RDNA4 的锅、还是这个模型自己的锅"的基准线。

模型：`RealVisXL_V5.0_fp16.safetensors`（6.9 GiB，SDXL 架构）

### 7.1 ComfyUI 路线

启动参数干净得多 —— **没有** RDNA4 环境变量，**没有** `--disable-pinned-memory`，
**没有** `--use-pytorch-cross-attention`：

```bat
venv\Scripts\python.exe main.py ^
  --listen 127.0.0.1 --port 8188 --disable-api-nodes ^
  --output-directory E:\AI\Data\Images\SDXL ^
  --temp-directory E:\AI\Temp\ComfyUI ^
  --user-directory E:\AI\Data\ComfyUI ^
  --auto-launch
```

模型走 §4.3 里 `extra_model_paths.yaml` 的 `sdxl` 段，不用复制进 ComfyUI 目录。

### 7.2 ForgeNeo 路线（同一份权重）

`COMMANDLINE_ARGS` **留空**即可，默认参数在 RDNA4 上直接能跑：

```bat
set COMMANDLINE_ARGS=
:: 下面这些都试过，SDXL 场景不需要，注释掉备查
:: --xformers --sage --uv
:: --pin-shared-memory --cuda-malloc --cuda-stream
call webui.bat
```

`config.json` 只改输出目录，让两个前端的产物都落到同一个数据盘：

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

> **推论（可直接用于排障）**：如果用户在 RDNA4 上跑 SDXL 遇到问题，
> **先不要往 ROCm workaround 的方向查** —— SDXL 这条路是干净的，
> 问题更可能在模型文件、VAE 或前端配置上。反过来，Qwen-Image 2.1 / H3 出问题，
> 优先查 §9 那张表。

---

## 8. 对话前端

两个前端都是纯本地回环，不对外暴露。

### 8.1 Open WebUI（接 Ollama，端口 8080）

```bat
set "DATA_DIR=%ROOT%data"
set "OLLAMA_BASE_URL=http://127.0.0.1:11434"
set "ENABLE_OPENAI_API=False"     rem 不连外部 API，纯本地
set "WEBUI_AUTH=False"            rem 仅因为只监听 127.0.0.1
set "HF_ENDPOINT=https://hf-mirror.com"   rem 国内镜像，拉 embedding 模型用
set "PYTHONUTF8=1"

venv\Scripts\open-webui.exe serve --host 127.0.0.1 --port 8080
```

⚠️ **`WEBUI_AUTH=False` 只在严格回环监听时才可接受。**
一旦把 `--host` 改成 `0.0.0.0` 或做端口转发，必须先把它改回 `True`，
否则局域网内任何人都能直接用你的模型。

启动脚本做了三件有用的事：

**① 先清场再启动**，并且**读 Windows 性能计数器把实际显存占用打出来**
（Windows 上没有 `nvidia-smi`，这是个实用替代）：

```powershell
$v = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples |
     ? CookedValue -gt 0 | select -First 1
'VRAM in use: {0:N2} GB' -f ($v.CookedValue/1GB)
```

**② Ollama 没跑就拉起来**，不重复启动：

```bat
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul ^
  || start "" /B "E:\AI\Apps\Ollama\ollama.exe" serve
```

**③ 轮询 `/health` 就绪后才开浏览器**，避免开出一个报错页：

```powershell
for($i=0;$i -lt 120;$i++){
  try{ Invoke-WebRequest http://127.0.0.1:8080/health -UseBasicParsing -TimeoutSec 2 | Out-Null
       Start-Process http://127.0.0.1:8080; break }
  catch{ Start-Sleep 2 }
}
```

### 8.2 SillyTavern（接 llama.cpp 的 OpenAI 兼容端点，端口 8000）

Node.js 应用，接 §3.2 那两个 `llama-server` 暴露的 OpenAI 兼容 API（`:8081`）。

`config.yaml` 关键项：

```yaml
port: 8000
listen: true             # 配合 whitelistMode 使用
whitelistMode: true      # ✅ 只有白名单 IP 能连，这是主要防线
basicAuthMode: false
securityOverride: false  # ✅ 保持 false，它会绕过白名单检查
enableCorsProxy: false
dataRoot: ./data
```

> **`listen: true` + `whitelistMode: true` 是安全的组合**；
> 把 `whitelistMode` 关掉或把 `securityOverride` 打开，就等于把角色卡和对话历史
> 开放给整个局域网。改这两项前务必想清楚。

启动：`Start.bat`（先 `npm install --omit=dev --ignore-scripts` 再 `node server.js`）。

### 8.3 端口总表

| 端口 | 服务 | 监听 |
|---|---|---|
| 11434 | Ollama API | 127.0.0.1 |
| 8081 | llama.cpp `llama-server`（OpenAI 兼容） | 0.0.0.0 ⚠️ |
| 8080 | Open WebUI | 127.0.0.1 |
| 8000 | SillyTavern | 白名单模式 |
| 8188 | ComfyUI 主生图 / SDXL | 127.0.0.1 |
| 8190 | ComfyUI Qwen 2.1 A-B 测试 | 127.0.0.1 |
| 8192 | ComfyUI MiniMax H3 视频 | 127.0.0.1 |
| 7860 | ForgeNeo（默认） | 127.0.0.1 |

⚠️ `llama-server` 用的是 `--host 0.0.0.0`，**局域网可达且无鉴权**。
如果不需要从别的设备连，改成 `--host 127.0.0.1`。

---

## 9. 故障速查表

按现象检索。**这些全部是实测遇到并解决的，不是推测。**

| 现象 | 根因 | 修法 |
|---|---|---|
| 首次推理挂死，跑 25 分钟零输出 | TunableOp 在 Windows+RDNA4 上损坏 | 删掉 `PYTORCH_TUNABLEOP_ENABLED`，改用 `MIOPEN_FIND_MODE=FAST` |
| ComfyUI 进程直接消失，无 traceback，退出码 `0xC0000005` | `safetensors 0.8.0` 读 >21 GiB 文件访问违规 | 打 §6.1 的 mmap 补丁 |
| pinned host memory 分配失败 | ROCm 不支持该路径 | 加 `--disable-pinned-memory` |
| flash-attn / triton 相关 ImportError | Windows ROCm 无 triton wheel | 加 `--use-pytorch-cross-attention` |
| VAE decode 阶段 HIP launch failure 或极慢 | 多模型共驻 16 GB（ComfyUI #15484） | 解码前插 §6.2 的门控节点 |
| 出图能出但质量微妙地差 | GGUF 只含语言塔，视觉塔未初始化 | 用 §5.3 的编码器补回视觉塔 |
| 扩散模型突然变慢、日志出现 lowvram | ComfyUI 只释放差额 → 部分卸载 | 用 §5.4 的 `free_memory(1e30, ...)` 强制完整卸载 |
| 27B 模型加载后 OOM | fp16 KV cache 放不下 | `-ctk q4_0 -ctv q4_0` 量化 KV cache |
| 第二个 ComfyUI 起不来 / 两个都崩 | 16 GB 装不下两份 | 启动脚本加 §5.5 端口检查 |
| Ollama 和 llama.cpp 互相抢显存 | 无互斥 | 启动前跑 §3.3 的清场脚本 |
| 跑分不可复现 | 从 `resolve/main/` 下的权重漂移了 | 固定到 `resolve/<commit>/` |
| SDXL 加了 RDNA4 参数没变化 | SDXL 本来就不需要那套（§4.2） | 去掉，从模型/VAE/前端配置方向查 |
| Open WebUI 打开就是报错页 | 服务还没就绪就开了浏览器 | 轮询 `/health` 就绪后再开（§8.1） |
| Windows 上想看显存占用但没有 nvidia-smi | AMD 卡无此工具 | 读性能计数器 `\GPU Adapter Memory(*)\Dedicated Usage`（§8.1） |
| H3 报 `'ModelSamplingAdvanced' object has no attribute 'audio_scale'` | `FLOW_AV` 需要支持音视频的采样节点 | 换采样节点（§6.5）。**与 RDNA4 无关**，NVIDIA 上同样报 |
| 想判断模型有没有退化成 lowvram | 部分加载 ≠ lowvram 降级 | grep 日志 `lowvram patches:`，非 0 才是降级（§6.4） |
| H3 很慢，调采样参数没用 | 瓶颈是加载不是计算，采样只占 20% | 权重挪到 SSD；日志里 `fast_disk=False` 就是慢盘标志（§6.4） |

---

## 10. 已证伪清单

**看起来该开、实测必须关的选项。给建议前先看这里。**

| 选项 | 状态 | 证据 |
|---|---|---|
| `PYTORCH_TUNABLEOP_ENABLED=1` | ❌ **禁用** | Windows+RDNA4 上挂死，25 分钟零输出 |
| `triton` 后端 | ❌ 不可用 | `ImportError: No module named 'triton'`，Windows ROCm 无 wheel |
| pinned memory（默认开） | ❌ **必须关** | 分配失败，须 `--disable-pinned-memory` |
| `nvfp4` / `mxfp8` 量化 | ❌ 不适用 | HIP 后端无对应算子，NVIDIA 专属 |
| Ollama 走 ROCm | ⚠️ 当时不支持 | RDNA4 不在支持列表，**Vulkan 反而稳**，34/34 层全卸载 |
| 1024×1024 以上不加 offload | ❌ 撞墙 | 峰值已达 15.9 GiB = 物理上限 |
| 两份 ComfyUI 共存 | ❌ 不可能 | 16 GB 不够 |
| 给 SDXL 套 RDNA4 workaround | ⚠️ **没必要** | 默认参数即可跑通，加了是白搭（§7） |
| ForgeNeo 的 `--xformers` / `--cuda-malloc` 等 | ⚠️ 未采用 | `COMMANDLINE_ARGS` 留空就能跑，无需这些 |

---

## 11. 目录布局参考

```
E:\AI\Apps
   ComfyUI/          三个实例共用一份代码，靠 --port / --extra-model-paths-config 区分
   ForgeNeo/         SDXL 第二前端，与 ComfyUI 共用同一份权重
   Ollama/           便携版 + 自定义 Modelfile
   llama.cpp-rocm/   gfx120X 预编译包，自带 HIP 运行时
   OpenWebUI/        venv + data
   SillyTavern/      Node.js
E:\AI\Models
   gguf/             llama.cpp 用的对话模型
   Image/            SDXL: Stable-diffusion/ Lora/ VAE/ ControlNet/ Upscale/
   Ollama/           blobs/ manifests/ metadata/
   QwenImage21/      diffusion_models/ text_encoders/ vae/ loras/
E:\AI\Cache          huggingface/ miopen/ pip/ torch/   ← 全部离盘，系统盘不落缓存
E:\AI\Projects       每个模型一个子目录：workflow + benchmark + environment-audit
E:\AI\Data           Images/ Videos/ ComfyUI/（用户目录）
E:\AI\Logs
F:\AI\Models\MiniMaxH3   H3 权重 53 GiB，单独放机械盘
```

**ComfyUI 一份代码跑三个实例**（SDXL :8188 / Qwen A-B :8190 / H3 :8192），
靠三样东西隔离：`--port`、`--temp-directory`、`--extra-model-paths-config`。
共用 `--user-directory`，所以节点布局和界面设置是共享的。

每个项目目录里固定放三样东西，这是能复现跑分的原因：

- `environment-audit.json` —— 包版本 + GPU 属性 + 每个权重的 SHA256 与官方值对比 + ComfyUI git commit
- `kernel-probe.json` —— 各后端实际可用算子清单 + int8 数值校验
- `*.result.json` —— 每次跑分的 wall time + 完整显存指标

---

## 覆盖范围与边界

**已验证**：「已部署清单」里的全部条目，在 §1 基线环境下实测。

**未验证 / 不保证**：

- **stable-diffusion.cpp 的 Vulkan 路线** —— 写过对照跑分脚本，但 `sd-cli.exe` 从未实际安装，
  所以**本文没有 Vulkan vs ROCm 的生图对照数据**。别把 §3.1 的 Ollama Vulkan 结论外推到生图。
- 其他 RDNA4 型号（9070 非 XT、9060 等）—— 显存和 CU 数不同，跑分不适用，环境变量结论大概率仍成立
- RDNA3（gfx110x）—— 算子能力表需重测
- Linux ROCm —— triton 可用，本文多处结论（`--use-pytorch-cross-attention`、TunableOp）不适用
- WSL2 —— 未测
- 多模型**同时**运行 —— 全部方案都是互斥运行，共驻只在 §6.2 作为失败案例出现

**许可**：本仓库的文档与自研节点代码 MIT。第三方模型权重各自遵循其原始许可，本仓库不再分发。

---

## 如果这份文档帮到了你

这些参数是在一张 16 GB 卡上反复撞墙试出来的 —— TunableOp 那 25 分钟、
safetensors 那个 `0xC0000005`、GGUF 缺视觉塔导致的静默降质，
每一条都花了不少时间才定位。

**如果它省了你几个小时，请点个 Star ⭐** —— 这能让更多 RDNA4 用户搜到它。

有新的踩坑、或在别的 RDNA4 / RDNA3 卡上跑出了不同结果，欢迎开 Issue 或 PR 补充。
尤其欢迎补充：其他 RDNA4 型号的跑分、Linux ROCm 的对照数据、safetensors 那个 bug 的上游修复进展。
