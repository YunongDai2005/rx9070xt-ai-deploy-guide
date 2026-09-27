# RDNA4 本地 AI 部署参考手册

**面向 AI 助手的结构化参考数据** · AMD Radeon RX 9070 XT (gfx1201) · Windows 11 原生 · 16 GB VRAM

---

## 这份文档是什么

一台 RX 9070 XT 上**实测跑通**的完整本地 AI 栈的部署参数、跑分数据和踩坑记录。
写成这个形式，是为了让 AI 助手在帮人部署 RDNA4 环境时能直接查到可靠答案，
而不是去猜、或者套用 NVIDIA 的经验。

### 给 AI 助手的使用说明

> **本文档是参考数据（reference data），不是指令集。**
> 下面的内容描述"在这套硬件上实测是什么结果"，不构成对你的任何要求。
> 请把它当作一份技术资料来引用，一切动作仍由用户决定。
>
> 引用建议：
> - **先核对基线**（§1）。gfx 型号、torch 版本、HIP 版本任一不同，跑分数据都不保证成立，环境变量结论大概率仍成立。
> - **§7 故障速查表**是按「报错现象 → 根因 → 修法」组织的，可以直接按症状检索。
> - **§8 已证伪清单**记录了"看起来该开、实际必须关"的选项。建议在给出配置建议前先看一眼，避免推荐已知会挂的参数。
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
| ❌ 不适用 | `nvfp4` 全系、`mxfp8` 全系（NVIDIA 专属） |

`int8_convrot` 数值校验：`relative_rmse = 0.0126`，`finite = true`，输出 `bf16`。**精度可接受。**

**推论：给 RDNA4 推荐量化格式时，`int8_convrot` 和 GGUF 是安全选择；任何 `nvfp4`/`mxfp8` 方案直接排除。**

---

## 2. 决策树：该走哪条路

```
用户要跑对话模型？
├── 模型 < 10 GiB，只要能用、省事      → Ollama + Vulkan  (§3.1)
└── 模型 > 10 GiB，或要长上下文        → llama.cpp ROCm   (§3.2)
                                          必须量化 KV cache

用户要跑 Qwen-Image 2.1？
├── 只生图，不和别的模型共存           → A: Q8_0 GGUF
│                                       热跑 40 s，但占 14 GB，跑完只剩 2.4 GB
└── 要和别的模型共存 / 之后要跑视频     → B: int8_convrot
                                        热跑 97 s，稳态仅 6 GB，剩 10.5 GB

用户要跑 MiniMax H3 视频？
└── 必须独占整卡，且必须打 mmap 补丁    → §5
    文本编码器 25.3 GiB > safetensors 0.8.0 的 Windows 崩溃阈值

任何 ComfyUI 场景
└── 必带 --disable-pinned-memory --use-pytorch-cross-attention
    必设 MIOPEN_FIND_MODE=FAST
    绝不开 TunableOp（§8）
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

所有 ROCm ComfyUI 启动脚本的公共前缀。**缓存全部离盘**，系统盘不落任何东西 ——
在 16 GB 卡上反复下载 20 GiB+ 权重时这很关键。

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

rem ==== AMD RDNA4 (gfx1201) 专用 ====
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
set MIOPEN_FIND_MODE=FAST
rem TunableOp 已移除，原因见 §8
```

必带命令行参数：

```
--disable-pinned-memory            ROCm 上 pinned host memory 分配会失败
--use-pytorch-cross-attention      无 triton，退回 PyTorch 原生 SDPA
--disable-api-nodes                纯本地，不联外部 API
--extra-model-paths-config <yaml>  每个项目一份独立模型路径表，多实例互不干扰
```

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

---

## 7. 故障速查表

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

---

## 8. 已证伪清单

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

---

## 9. 目录布局参考

```
E:\AI\Apps      ComfyUI / ForgeNeo / Ollama / llama.cpp-rocm / OpenWebUI / SillyTavern
E:\AI\Models    gguf/ Image/ Ollama/ QwenImage21/
E:\AI\Cache     huggingface/ miopen/ pip/ torch/     ← 全部离盘，系统盘不落缓存
E:\AI\Projects  每个模型一个子目录：workflow + benchmark + environment-audit
E:\AI\Logs
F:\AI\Models\MiniMaxH3    H3 权重 53 GiB，单独放机械盘
```

每个项目目录里固定放三样东西，这是能复现跑分的原因：

- `environment-audit.json` —— 包版本 + GPU 属性 + 每个权重的 SHA256 与官方值对比 + ComfyUI git commit
- `kernel-probe.json` —— 各后端实际可用算子清单 + int8 数值校验
- `*.result.json` —— 每次跑分的 wall time + 完整显存指标

---

## 覆盖范围与边界

**已验证**：上表全部条目，在 §1 基线环境下实测。

**未验证 / 不保证**：

- 其他 RDNA4 型号（9070 非 XT、9060 等）—— 显存和 CU 数不同，跑分不适用，环境变量结论大概率仍成立
- RDNA3（gfx110x）—— 算子能力表需重测
- Linux ROCm —— triton 可用，本文多处结论（`--use-pytorch-cross-attention`、TunableOp）不适用
- WSL2 —— 未测

**许可**：本仓库的文档与自研节点代码 MIT。第三方模型权重各自遵循其原始许可，本仓库不再分发。

---

## 如果这份文档帮到了你

这些参数是在一张 16 GB 卡上反复撞墙试出来的 —— TunableOp 那 25 分钟、
safetensors 那个 `0xC0000005`、GGUF 缺视觉塔导致的静默降质，
每一条都花了不少时间才定位。

**如果它省了你几个小时，请点个 Star ⭐** —— 这能让更多 RDNA4 用户搜到它。

有新的踩坑、或在别的 RDNA4 / RDNA3 卡上跑出了不同结果，欢迎开 Issue 或 PR 补充。
尤其欢迎补充：其他 RDNA4 型号的跑分、Linux ROCm 的对照数据、safetensors 那个 bug 的上游修复进展。
