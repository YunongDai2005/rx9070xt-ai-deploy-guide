# 运行日志实录 · RX 9070 XT

真实运行日志 + 逐段注解。[← 返回主文档](README.md) · [English](RUNLOGS.en.md)

这份文件的用途：让你**在自己跑之前就知道正常的日志长什么样**。
照着主文档部署时，把你的日志和这里对比，偏差在哪一眼就能看出来。

> **关于"效果"的说明**：本文件只记录**可测量的**指标 —— 耗时、显存、
> 层数、吞吐、报错。**不包含任何生成内容的画质/观感评价**，因为撰写时
> 未查看任何输出图像或视频。画质请你自己判断，本文只负责让你知道
> 什么设置下该花多久、占多少显存。

---

## 1. Ollama + Vulkan 加载 Qwen3.5 9B

### 1.1 设备识别

```
"inference compute" id=0 library=Vulkan name=Vulkan0
    description="AMD Radeon RX 9070 XT" type=discrete
    total="15.9 GiB" available="15.1 GiB"
"gpu memory" id=0 library=Vulkan available="14.6 GiB" free="15.1 GiB"
    minimum="457.0 MiB" overhead="0 B"
common_param: - Vulkan0 : AMD Radeon RX 9070 XT (16304 MiB, 15437 MiB free)
```

✅ **该看到什么**：`library=Vulkan`、`type=discrete`、`15437 MiB free`。
❌ **如果 `library=ROCm`**：说明你的 Ollama 版本已支持 RDNA4 的 ROCm，
本文的 Vulkan 结论可能不再适用，速度请自行重测。

### 1.2 层卸载 —— 最关键的三行

```
load_tensors: offloaded 34/34 layers to GPU
load_tensors: offloading 32 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors:      Vulkan0 model buffer size =  4717.38 MiB
load_tensors:  Vulkan_Host model buffer size =   545.63 MiB
```

✅ **`34/34` 是成功标志。** 分母是模型总层数，两个数字必须相等。
❌ **如果是 `28/34` 之类**：有层留在 CPU 上，速度会掉到个位数 tok/s。
原因通常是显存被别的进程占了 —— 先跑主文档 §3.3 的清场脚本。

显存分配明细：

```
common_memory_breakdown_print:
  Vulkan0 (RX 9070 XT) | 16304 = 15386 + (5231 = 4717 + 306 + 208) + -4314
                         总量    可用    合计   权重  KV   其他
```

### 1.3 上下文：一个容易被忽略的余量

```
print_info: n_ctx_train           = 262144
print_info: n_ctx_orig_yarn       = 262144
OLLAMA_CONTEXT_LENGTH:8192
```

> **Qwen3.5 9B 原生支持 256K 上下文，本部署只开了 8192。**
> 不是模型的限制，是 16 GB 显存的取舍 —— 权重已占 4.7 GB，
> 上下文拉长 KV cache 会线性膨胀。
> **想要长上下文：要么降量化，要么按主文档 §3.2 换 llama.cpp 走量化 KV cache。**

其他值得注意的默认值：

```
OLLAMA_FLASH_ATTENTION:false    ← Vulkan 后端未启用
OLLAMA_NUM_PARALLEL:1           ← 单并发，16 GB 卡上合理
OLLAMA_KEEP_ALIVE:5m0s          ← 5 分钟无请求自动卸载，释放显存给游戏
tokenizer.ggml.tokens arr[str,248320]   ← 词表 248320
```

### 1.4 实测吞吐

| 指标 | 数值 |
|---|---|
| 生成 | **~24 tok/s** |
| 提示处理 | 250–340 tok/s |
| 冷启动加载 | ~十几秒 |
| 权重显存 | 4717.38 MiB |
| 主机端 | 545.63 MiB |

**24 tok/s 大致是什么体验**：比人的阅读速度快，交互上没有等待感。
作为对照，同卡走 llama.cpp ROCm 的 27B Q3_K 会明显慢于此 —— 参数量翻了 3 倍。

---

## 2. Qwen-Image 2.1 A/B 跑分

完整数据见主文档 §5.2，这里给日志侧的观察。

### 2.1 权重校验（跑分前必做）

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

> **`ok: true` 全绿再跑分。** 下载中断导致的截断文件能加载成功但出图异常，
> 这种问题不查 hash 会浪费很多时间。

### 2.2 显存监控采样（方案 B, int8_convrot）

跑分脚本每 30 秒采一次峰值分配，这是完整曲线：

```
 29 s   peak allocated  0.000 GiB   ← 还在读盘
 59 s   peak allocated  0.000 GiB
 97 s   peak allocated  5.182 GiB   ← 权重开始上卡
127 s   peak allocated  5.460 GiB
157 s   peak allocated  5.739 GiB
187 s   peak allocated  6.047 GiB
217 s   peak allocated  7.056 GiB
247 s   peak allocated 12.109 GiB   ← 采样开始，显存跳升
277 s   peak allocated 13.707 GiB
307 s   peak allocated 13.707 GiB   ← 平台期
337 s   peak allocated 13.707 GiB
FINISHED 363.88 s
```

**两个可以直接拿来对照的特征：**

1. **前 90 秒显存是 0** —— 全在读盘。看到这里别以为卡住了。
2. **247 s 处从 7 GB 跳到 12 GB** —— 采样阶段的激增。
   如果你的卡在这一步 OOM，说明前面留的余量不够。

### 2.3 A/B 最终结果

| | 冷跑 | 热跑 | 稳态显存 | 跑完剩余 |
|---|---|---|---|---|
| A · Q8_0 GGUF | 294.7 s | **40.4 s** | 14.03 GB | 2.38 GB |
| B · int8_convrot | 155.9 s | **97.4 s** | 6.02 GB | 10.49 GB |

**热跑差 2.4 倍，显存差 2.3 倍 —— 这是一组干净的反向权衡。**
1024×1024 / 20 步 / cfg 6.0 / euler。

---

## 3. MiniMax H3 文生视频 · 完整运行

唯一一次完整成功的运行，**277.87 秒 / 10 步**。

### 3.1 启动阶段

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

✅ 三个自研节点必须出现在这个列表里，否则补丁没生效。
✅ `triton` 报 ImportError 是**正常的** —— Windows ROCm 没有 triton wheel。

### 3.2 mmap 补丁生效

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

> **`mmap loading` 出现两次才对** —— 25.3 GiB 的编码器和 19.5 GiB 的扩散模型
> 都超过 16 GiB 阈值。**如果一次都没出现，说明补丁没装上，进程会直接被
> `0xC0000005` 杀掉**（详见主文档 §6.1）。

**`Native ops` / `emulated ops` 这两行是判断量化格式是否真被加速的唯一依据。**
`nvfp4` 和 `mxfp8` 在 emulated 一侧 —— 能跑，但没有硬件加速。

### 3.3 模型加载：部分加载但未降级

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

> ⭐ **`lowvram patches: 0` 是整份日志里最该 grep 的一行。**
> 模型被拆开（合计约 19 GB 卸到 CPU）是正常且必要的，
> **但 `lowvram patches` 一旦非 0，就说明落进了 ComfyUI 的降级路径，速度会断崖下跌。**
> 部分加载 ≠ 降级，这两件事经常被搞混。

同时可以看到 AOTriton 注意力后端确实启用了：

```
UserWarning: Using AOTriton backend for Efficient Attention forward...
(aten/src/ATen/native/transformers/hip/attention.hip:1452)
```

### 3.4 采样：首步慢 3.4 倍是正常的

```
  0%|          | 0/10 [00:00<?, ?it/s]
 10%|█         | 1/10 [00:13<02:00, 13.41s/it]   ← 含 kernel 首次编译
 20%|██        | 2/10 [00:19<01:10,  8.82s/it]
 30%|███       | 3/10 [00:25<00:54,  7.81s/it]
 40%|████      | 4/10 [00:32<00:43,  7.30s/it]
 50%|█████     | 5/10 [00:37<00:32,  6.53s/it]
 60%|██████    | 6/10 [00:41<00:22,  5.57s/it]
 70%|███████   | 7/10 [00:44<00:15,  5.00s/it]
 80%|████████  | 8/10 [00:48<00:08,  4.45s/it]
 90%|█████████ | 9/10 [00:51<00:04,  4.14s/it]
100%|██████████| 10/10 [00:55<00:00,  3.98s/it]  ← 收敛值
100%|██████████| 10/10 [00:55<00:00,  5.52s/it]  ← 平均值
```

> **首步 13.41 s，末步 3.98 s。** 首步包含 MIOpen kernel 选择和 AOTriton
> 注意力内核编译。**评估这张卡必须用收敛值 3.98 s/it，用平均值 5.52 s/it
> 会低估 39%。** 如果你的首步耗时远超 13 s，检查 MIOpen 缓存目录是否可写
> （主文档 §4.1 的 `MIOPEN_USER_DB_PATH`）。

### 3.5 解码阶段

```
[H3-VRAM-Gate] free VRAM 15359 -> 15357 MB
Requested to load MiniMaxH3AudioVAE
loaded completely; 13513.42 MB usable, 577.08 MB loaded, full load: True
Requested to load MiniMaxH3VideoVAE
loaded completely; 12827.35 MB usable, 4966.19 MB loaded, full load: True
Prompt executed in 277.87 seconds
```

⚠️ **诚实标注**：门控节点只释放了 2 MB（15359 → 15357），
说明采样结束时 ComfyUI 已经自己清干净了。**这条日志不能证明门控"救了"这次运行** ——
它的价值是保证时序确定，不依赖 ComfyUI 恰好做对。

### 3.6 时间去向

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 模型加载 + 文本编码 | ~215 s | **77%** |
| 采样（10 步） | 55 s | 20% |
| 双 VAE 解码 | ~8 s | 3% |
| **合计** | **277.87 s** | |

> **瓶颈是读盘，不是算力。** 日志里 `fast_disk=False` 就是慢盘标志 ——
> 本机 H3 的 53 GiB 权重在机械盘上。**换 SSD 是唯一有意义的优化方向，
> 调采样器参数改不了那 77%。**

### 3.7 失败记录：音频分支

```
Prompt executed in 148.23 seconds
!!! Exception during processing !!!
AttributeError: 'ModelSamplingAdvanced' object has no attribute 'audio_scale'
```

`model_type` 是 `FLOW_AV`（音视频联合），采样节点必须携带 `audio_scale` 属性。
**视频分支完整可用，音频分支未跑通。**
这个报错**与 RDNA4 无关，NVIDIA 上同样复现** —— 是节点选择问题。

---

## 4. 安装阶段日志

装环境时最容易卡住的几处，都有日志佐证。

### 4.1 ROCm SDK 不在 PyPI 上

```
Collecting rocm-sdk-core==7.2.1
  Downloading https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/
             rocm_sdk_core-7.2.1-py3-none-win_amd64.whl (644.8 MB)
```

> **Windows 的 ROCm SDK 来自 `repo.radeon.com`，不是 PyPI。** 单个 wheel 644.8 MB。
> 本机这次下载中途因网络读取错误失败过（日志里是 urllib3 的 `_error_catcher` 栈）。
> **准备好断点重试** —— 644 MB 一次拉完在不稳定的线路上不现实。

### 4.2 镜像加速与跳过 490 MB 无用素材

```
=== 1. 生成过滤后的依赖清单 ===
  已剔除: comfyui-workflow-templates==0.11.66
  写入: requirements-nomedia.txt

=== 2. 模板包（只装本体和 JSON，跳过 ~490MB 预览素材） ===
>>> templates 本体 (--no-deps)  [1]  mirror.nju.edu.cn   OK
>>> templates core + json      [1]  mirror.nju.edu.cn   OK
=== 3. 安装 ComfyUI 其余依赖 ===
>>> ComfyUI requirements (过滤后) [1] mirror.nju.edu.cn  OK
=== 4. 安装 ComfyUI-GGUF 依赖 ===
>>> GGUF requirements          [1]  mirror.nju.edu.cn   OK
```

**两个实用技巧：**

1. **`comfyui-workflow-templates` 带约 490 MB 预览缩略图素材。**
   用 `--no-deps` 只装本体 + JSON，省 490 MB。
   代价：模板浏览器里缩略图是空的，**不影响出图**，本地工作流文件照常加载。
2. **pip 走国内镜像**（本机用 `mirror.nju.edu.cn`），否则 torch ROCm 那几个
   大 wheel 基本拉不动。

### 4.3 安装完成的验证输出

```
=== 5. 验证 ===
torch 2.9.1+rocm7.2.1
gpu_available True
device AMD Radeon RX 9070 XT
imports_ok True
```

✅ **这四行全对才算装好。** `gpu_available False` 最常见的原因是
装了 CPU 版 torch 覆盖掉了 ROCm 版 —— 检查 `torch.__version__`
末尾有没有 `+rocm7.2.1` 后缀。

---

## 5. 效果预期表

**什么设置 → 该花多久、占多少显存。** 用于开跑前对表，不是画质评价。

### 5.1 对话

| 模型 | 上下文 | 显存 | 速度 | 体验 |
|---|---|---|---|---|
| Qwen3.5 9B Q4 | 8K | 4.7 GB | 24 tok/s | 交互无等待感 |
| 27B Q3_K | 16K（KV q4_0） | ~12.6 GB + KV | 明显慢于 9B | 参数量 3 倍的代价 |
| 12B Q6_K | 32K（KV q8_0） | ~9.4 GB + KV | 介于两者之间 | 长上下文 + 较高量化质量 |

### 5.2 文生图

| 路线 | 分辨率 | 步数 | 热跑耗时 | 稳态显存 | 能否同时干别的 |
|---|---|---|---|---|---|
| Qwen-Image 2.1 · Q8_0 | 1024² | 20 | **40 s** | 14.0 GB | ❌ 卡满 |
| Qwen-Image 2.1 · int8 | 1024² | 20 | **97 s** | 6.0 GB | ✅ 余 10.5 GB |
| + Pruna 8 步 LoRA | 1024² | 8 | 未单独计时 | 同上 | 步数降 60% |
| SDXL RealVisXL V5.0 | 1024² | — | 未计时 | ~7 GB | 默认参数即可 |

⚠️ **1024×1024 是 16 GB 上不加 offload 的天花板** ——
两条 Qwen-Image 路线的峰值分配都顶到 ~15.9 GiB，即物理上限。
想上更高分辨率必须先谈 offload 或降分辨率。

### 5.3 文生视频

| 项 | 数值 |
|---|---|
| 步数 | 10 |
| 总耗时 | **277.87 s**（约 4 分 38 秒） |
| 采样收敛速度 | 3.98 s/it |
| 权重总量 | 53 GiB（需外挂大容量盘） |
| 峰值上卡 | 13.4 GB（另有约 19 GB 卸到 CPU） |
| 系统内存要求 | 32 GB 勉强够（靠 mmap 只读映射不计 commit charge） |
| 音频分支 | ⚠️ 未跑通（§3.7） |

> **H3 在这张卡上是"能跑"而不是"好用"** —— 单条 10 步视频接近 5 分钟，
> 其中 77% 在读盘。权重放 SSD 能显著改善，但 53 GiB 的占用是硬门槛。

---

## 6. 快速自检清单

照主文档部署后，按顺序 grep 你自己的日志：

```bash
# 1. torch 是 ROCm 版
python -c "import torch;print(torch.__version__)"      # 必须含 +rocm

# 2. 认到卡
python -c "import torch;print(torch.cuda.get_device_properties(0).gcnArchName)"   # gfx1201

# 3. Ollama 全层上卡
grep "offloaded" server.log                            # 必须 N/N

# 4. 自研节点已加载
grep "custom_nodes" comfyui.log                         # 三个节点都在

# 5. H3 的 mmap 补丁生效
grep "mmap loading" comfyui.log                         # 必须出现 2 次

# 6. 没有落进降级路径
grep "lowvram patches" comfyui.log                      # 必须全是 0

# 7. 权重没被截断
# 跑主文档提到的 audit_environment.py，看 matches 是否全 true
```

任何一条不符，去主文档 §9 故障速查表按现象查。

---

[← 返回主文档](README.md)
