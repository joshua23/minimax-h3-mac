# 在 MacBook 上本地运行「满血」MiniMax-H3

> **不用云端 GPU，不用 80GB 显存。** 这个项目让一台 **24GB 内存的 Apple M4 Pro MacBook Pro**，通过 MLX 流式加载，直接运行 MiniMax-H3 原始 BF16 DiT、原始 BF16 Text Encoder，并生成带立体声音频的 1344×768 视频。

## 已经真实跑通

我们在以下机器上完成了端到端实测：

- **设备**：MacBook Pro（Mac16,8）
- **芯片**：Apple M4 Pro，14 核 CPU
- **统一内存**：24GB
- **DiT**：MiniMax-H3 上游原始 BF16，约 62GiB
- **Text Encoder**：上游原始 BF16 全精度权重
- **Turbo**：LightX2V MiniMax-H3-Turbo v1.0 4-step 768p，BF16
- **输出**：1344×768、124 帧、24 FPS、5.17 秒立体声音频
- **总耗时**：**47 分 58.7 秒**
- **实测峰值内存 footprint**：约 **15.8GB**

### 实际生成效果

**提示词：**

```text
A cinematic red panda running through a misty bamboo forest, detailed fur, natural lighting, smooth coherent motion, cinematic camera movement
```

- **[点击观看或下载 1344×768 生成视频](examples/bf16-turbo-768p/output-bf16-turbo-1344x768-5s.mp4)**
- [提示词文件](examples/bf16-turbo-768p/prompt.txt)
- 视频 SHA256：`b52a18e32f58fdfa387798e7cc1425bcb14f795797ec7168323b479e7bd49c85`

MP4 内已经包含 H.264 视频和 AAC 立体声音频，不需要额外下载 WAV。

---

## 这里的「满血」是什么意思？

本项目支持两条运行路线：

### 1. 原始 BF16 满精度路线

这是本 README 重点展示的路线：

- DiT 使用 MiniMax 官方发布的**原始 BF16 权重**；
- Text Encoder 使用上游**原始 BF16 权重**；
- Turbo LoRA 保持 BF16，不合并、不重新量化；
- VAE 使用上游原始权重；
- Attention 使用精确 dense attention，没有稀疏近似；
- 不跳过 DiT Block，不做低比特重建，不改变计算顺序。

MiniMax-H3 的条件特征读取 `hidden_states[50]`，因此 Text Encoder 会严格执行模型实际需要的前 50 层，并返回 final norm 之前的状态。后 14 层本来就不属于 H3 的条件计算路径，继续执行反而会改变模型输入语义。

### 2. 校准 INT8 路线

如果更在意磁盘占用和线性层速度，可以使用 Argus 校准 INT8 DiT：

- Text Encoder 默认仍可使用原始 BF16；
- DiT 切换为校准 INT8；
- Turbo 保持 BF16；
- 同样支持低内存流式运行。

---

## 24GB Mac 为什么能装下 62GiB BF16 模型？

关键不是把整个模型塞进内存，而是**只让当前正在计算的权重驻留**。

### Text Encoder 流式加载

1. 临时加载 BF16 Embedding，生成初始 hidden states；
2. 立即释放 Embedding；
3. 使用一个可复用 Decoder Layer 槽位；
4. 逐层加载并执行 H3 需要的 50 层；
5. 每层先 `mx.eval()` 完成计算，再释放权重；
6. 得到文本条件后，释放整个 Text Encoder，再进入 DiT 阶段。

### DiT 流式加载

原始 BF16 DiT 的 50 个主 Block 每个约 1.29GB。默认每次驻留两个：

```text
1.72GB 静态 DiT 权重
+ 2 × 1.29GB 当前 BF16 Block
+ 当前激活、Attention workspace 和 Metal 缓冲区
```

Block 仍然严格顺序执行。当前组计算并物化完成后，立即释放，再加载下一组。模型总大小主要影响磁盘占用和读取量，不再直接决定峰值统一内存。

---

# 完整部署教程：BF16 DiT + BF16 Encoder + 768p Turbo

## 1. 安装环境

```bash
git clone https://github.com/Argus-AiTeam/minimax-h3-mac.git
cd minimax-h3-mac

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

安装并登录 Hugging Face CLI：

```bash
pip install -U huggingface_hub
hf auth login
```

运行前需要阅读并接受 [MiniMax-H3 License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。

## 2. 下载官方基础资产与 BF16 Text Encoder

```bash
mkdir -p models

hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/model_index.json" \
            "FL2VA/tokenizer/**" \
            "FL2VA/processor/**" \
            "FL2VA/text_encoder/**" \
            "FL2VA/video_vae/**" \
            "FL2VA/audio_vae/**" \
  --local-dir models/MiniMax-H3
```

这里会下载 tokenizer、processor、原始 BF16 Text Encoder、Video VAE 和 Audio VAE。

## 3. 下载官方原始 BF16 DiT

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/transformer/**" \
  --local-dir models/MiniMax-H3
```

BF16 Transformer 大约占用 62GiB 磁盘空间，但运行时不需要整体常驻内存。

## 4. 下载 LightX2V 768p BF16 Turbo

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --revision e6346777701aa2b64d42ed058cdd71ae00e7cd52 \
  --local-dir models/Minimax-h3-Turbo-v1.0-4step-768p-bf16
```

固定文件信息：

- 大小：`1,383,677,808` bytes
- SHA256：`1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c`
- Rank：128
- Alpha：128（程序会自动从 safetensors metadata 读取）
- 推荐 NFE：4
- Video sigma shift：6
- Audio sigma shift：3

## 5. 先做最小冒烟测试

第一次运行建议先确认模型路径、内存和 MP4 封装全部正常：

```bash
mkdir -p out profiles

caffeinate -dimsu .venv/bin/python scripts/generate.py \
  "A red panda waves from a tiny stage" \
  --checkpoint models/MiniMax-H3/FL2VA \
  --transformer models/MiniMax-H3/FL2VA/transformer \
  --turbo-lora models/Minimax-h3-Turbo-v1.0-4step-768p-bf16/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --turbo-lora-scale 1.0 \
  --sigma-shift-video 6 \
  --sigma-shift-audio 3 \
  --steps 5 \
  --low-memory \
  --stream-blocks \
  --stream-block-group-size 2 \
  --no-block-cache \
  --resolution 64x64 \
  --duration 1 \
  --memory-limit-gb 24 \
  --require-muxed-mp4 \
  --output out/bf16-smoke.mp4
```

## 6. 正式生成 1344×768、5 秒视频

```bash
mkdir -p out profiles

caffeinate -dimsu .venv/bin/python scripts/generate.py \
  "A cinematic red panda running through a misty bamboo forest, detailed fur, natural lighting, smooth coherent motion, cinematic camera movement" \
  --checkpoint models/MiniMax-H3/FL2VA \
  --transformer models/MiniMax-H3/FL2VA/transformer \
  --turbo-lora models/Minimax-h3-Turbo-v1.0-4step-768p-bf16/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --turbo-lora-scale 1.0 \
  --sigma-shift-video 6 \
  --sigma-shift-audio 3 \
  --steps 5 \
  --low-memory \
  --stream-blocks \
  --stream-block-group-size 2 \
  --no-block-cache \
  --resolution 1344x768 \
  --duration 5 \
  --memory-limit-gb 24 \
  --require-muxed-mp4 \
  --forward-profile-json profiles/bf16-turbo-1344x768-5s.json \
  --output out/bf16-turbo-1344x768-5s.mp4
```

### 参数重点

- `--steps 5`：5 个 sigma grid points，对应 **4 次 DiT forward / 4 NFE**；
- `--sigma-shift-video 6 --sigma-shift-audio 3`：LightX2V 768p Turbo 的发布配置；
- `--stream-block-group-size 2`：每次驻留两个 BF16 DiT Block，约 2.58GB；
- `--no-block-cache`：关闭近似残差缓存，保留完整计算；
- `--memory-limit-gb 24`：针对本次 24GB M4 Pro 实测配置；
- 不需要手动传 `--turbo-lora-alpha 128`，程序会读取文件 metadata；显式传入也等价。

## 7. 后台运行与查看进度

后台运行：

```bash
nohup caffeinate -dimsu .venv/bin/python scripts/generate.py \
  "你的提示词" \
  ...其余参数... \
  > generate.log 2>&1 &

echo $! > generate.pid
```

查看日志：

```bash
tail -f generate.log
```

典型进度：

```text
text encoder: full-precision weights, streaming 50 layers
loaded 34 static tensors (1.72 GB, BF16/full-precision)
main blocks stream lazily in groups of 2
step 1/4 ...
step 2/4 ...
step 3/4 ...
step 4/4 ...
wrote ...mp4
```

验证生成文件：

```bash
ffmpeg -v error -i out/bf16-turbo-1344x768-5s.mp4 -f null -
```

没有输出错误即表示视频和音频可以完整解码。

---

# 24GB M4 Pro 实测性能

本次实测生成 1344×768、124 帧、24 FPS、5.17 秒立体声音频：

| 阶段 | 实测耗时 |
|---|---:|
| BF16 Text Encoder | 30.5 秒 |
| DiT Step 1 | 610.7 秒 |
| DiT Step 2 | 599.9 秒 |
| DiT Step 3 | 593.0 秒 |
| DiT Step 4 | 600.9 秒 |
| 四次 DiT 合计 | 2,404.5 秒（40 分 4.5 秒） |
| Video VAE 解码 | 427.4 秒（7 分 7.4 秒） |
| Audio VAE 解码 | 1.8 秒 |
| MP4 封装 | 2.1 秒 |
| **端到端总耗时** | **2,878.7 秒（47 分 58.7 秒）** |

其他数据：

- 平均每次 DiT forward：**601.1 秒**；
- 峰值内存 footprint：约 **15.8GB**；
- 最大 RSS：约 **6.9GB**，Metal/IOSurface 使用不同的系统记账口径；
- Attention：约占非重叠 profiling 时间的 **50.1%**；
- 线性投影：约占 **29.6%**；
- 模型加载：仅约占 **0.6%**；
- 输出 MP4：1344×768 H.264 + 32kHz stereo AAC，已通过完整解码验证。

这说明 768p 下的主要瓶颈是 dense attention 和 BF16 线性计算，而不是磁盘加载。增大 Block group size 只能减少少量切换开销，不会让多个 Transformer Block 并行执行。

实际耗时会随提示词长度、芯片型号、温度、SSD 和后台负载变化。

---

# 更省空间：使用校准 INT8 DiT

如果不想下载 62GiB BF16 DiT，可以改用 MLX INT8：

```bash
hf download water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --local-dir models/MiniMax-H3-MLX-Argus-Calibrated-INT8
```

运行示例：

```bash
.venv/bin/python scripts/generate.py \
  "A fox running through a misty forest" \
  --checkpoint models/MiniMax-H3/FL2VA \
  --transformer models/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --low-memory \
  --stream-blocks \
  --resolution 320x192 \
  --duration 5 \
  --steps 21 \
  --require-muxed-mp4 \
  --output out/int8-output.mp4
```

`--steps 21` 对应 20 次 denoiser evaluation。Text Encoder 默认仍使用上游 BF16 流式路径。

---

# 可选：Larry Turbo LoRA 原值打包

Larry 原始 safetensors 中的 BF16 数组可以被 MLX 使用。以下工具会验证 key、shape、rank，记录 `alpha=rank`，并拆成可按组件流式加载的目录，不改变 tensor 数值：

下载：<https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora>

```bash
python scripts/convert_turbo_lora_to_mlx.py \
  --source /path/to/minimax_h3_turbo_v4_step600_ema.safetensors \
  --config models/MiniMax-H3-MLX-Argus-Calibrated-INT8/config.json \
  --output-dir models/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --source-repo larryvrh/MiniMax-H3-Turbo-Lora \
  --source-revision 43a74557ac3f6539db8e0f2a959d03feb7a81480
```

工具会保留全部 518 个 BF16 tensor、验证 259 对 LoRA，并生成 53 个组件 shard。

---

# 模型与项目链接

- 本项目：<https://github.com/Argus-AiTeam/minimax-h3-mac>
- MiniMax-H3 官方权重：<https://huggingface.co/MiniMaxAI/MiniMax-H3>
- Argus 校准 MLX INT8 DiT：<https://huggingface.co/water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8>
- LightX2V Turbo v1.0 768p：<https://huggingface.co/lightx2v/Minimax-h3-Turbo>
- Larry Turbo LoRA：<https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora>

---

## 重要说明

- MiniMax-H3 权重受其原始 License 约束；
- 1344×768、5 秒生成在 24GB M4 Pro 上实测接近 48 分钟，不是实时推理；
- 当前开源路径使用 dense attention，MiniMax 官方稀疏 Attention 实现并未随权重公开；
- “满血 BF16”指推理使用上游原始 BF16 DiT/Text Encoder 权重和完整 H3 计算路径，不代表改变或绕过上游模型本身的架构定义。

**现在，一台 24GB 的 MacBook Pro，也可以在本地完整跑起 MiniMax-H3。**
