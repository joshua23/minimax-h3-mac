# Run Full-Precision MiniMax-H3 Locally on a MacBook

[中文](README.md) | [English](README_EN.md)

> **Most of the code adaptation, experiment execution, performance validation, and documentation in this repository were completed autonomously by [Argus](https://github.com/lbx154/Argus), building on community contributions with very little human involvement in the loop.**

> **We strongly encourage everyone to try [Argus](https://github.com/lbx154/Argus): let the agent read the codebase, modify the project, run long experiments, analyze results, and iterate continuously. Stars, trials, and contributions are welcome.**

> **This repository is continuously maintained by Argus Agent / Argus-AiTeam, with ongoing work on MiniMax-H3 inference speed, memory efficiency, stability, and generation quality on Apple Silicon Macs.**

> **No cloud GPU and no 80 GB discrete VRAM required.** With MLX weight streaming, this project runs the original MiniMax-H3 BF16 DiT and BF16 text encoder on a **24 GB Apple M4 Pro MacBook Pro**, producing 1344×768 video with stereo audio entirely on-device.

## Proven end-to-end on a real MacBook

The full pipeline was measured on:

- **Device:** MacBook Pro (`Mac16,8`)
- **Chip:** Apple M4 Pro, 14-core CPU
- **Unified memory:** 24 GB
- **DiT:** original upstream MiniMax-H3 BF16, approximately 62 GiB
- **Text encoder:** original upstream full-precision BF16 weights
- **Turbo:** LightX2V MiniMax-H3-Turbo v1.0 4-step 768p, BF16
- **Output:** 1344×768, 124 frames, 24 FPS, 5.17 seconds of stereo audio
- **End-to-end time:** **47 minutes 58.7 seconds**
- **Measured peak memory footprint:** approximately **15.8 GB**

### Actual generated result

**Prompt:**

```text
A cinematic red panda running through a misty bamboo forest, detailed fur, natural lighting, smooth coherent motion, cinematic camera movement
```

- **[Watch or download the generated 1344×768 video](examples/bf16-turbo-768p/output-bf16-turbo-1344x768-5s.mp4)**
- [Prompt text file](examples/bf16-turbo-768p/prompt.txt)
- Video SHA256: `b52a18e32f58fdfa387798e7cc1425bcb14f795797ec7168323b479e7bd49c85`

The MP4 already contains H.264 video and AAC stereo audio; no separate WAV is required.

---

## What does “full-precision” mean here?

This project supports two deployment routes.

### 1. Original BF16 route

This is the quality-oriented route highlighted in this README:

- The DiT uses the **original BF16 weights published upstream** by MiniMax;
- The text encoder uses the **original upstream BF16 weights**;
- the Turbo LoRA stays BF16 and is neither merged nor requantized;
- the VAEs use the upstream weights;
- attention uses exact dense attention, without a sparse approximation;
- no DiT blocks are skipped, no low-bit reconstruction is introduced, and block execution order is unchanged.

MiniMax-H3 consumes `hidden_states[50]` from its conditioner. The text encoder therefore executes exactly the first 50 layers required by H3 and returns the state before the final norm. The remaining 14 language-model layers are not part of H3 conditioning; executing them would change the conditioning semantics rather than improve precision.

### 2. Calibrated INT8 route

If disk footprint and linear-layer speed matter more, you can use the Argus calibrated INT8 DiT:

- the text encoder can still use the original BF16 weights by default;
- only the DiT changes to calibrated INT8;
- Turbo remains BF16;
- low-memory streamed execution remains available.

---

## How does a 24 GB Mac run a 62 GiB BF16 model?

The key is not to place the complete model in memory. Only the weights required by the current computation remain resident.

### Streamed text encoder

1. Load the BF16 embedding transiently and produce the initial hidden states;
2. release the embedding immediately;
3. keep one reusable decoder-layer slot;
4. load and execute the 50 layers required by H3 one at a time;
5. call `mx.eval()` before releasing each layer’s weights;
6. release the complete text encoder after producing the conditioning, then load the DiT stage.

### Streamed DiT

Each of the 50 original BF16 DiT blocks is approximately 1.29 GB. By default, two adjacent blocks are resident at a time:

```text
1.72 GB static DiT weights
+ 2 × 1.29 GB active BF16 blocks
+ current activations, attention workspace, and Metal buffers
```

Blocks still execute in strict sequence. Once the current group has finished and materialized, it is released before the next group is loaded. Total model size primarily affects disk space and I/O volume rather than peak unified-memory residency.

---

# Full deployment guide: BF16 DiT + BF16 encoder + 768p Turbo

## 1. Install

```bash
git clone https://github.com/Argus-AiTeam/minimax-h3-mac.git
cd minimax-h3-mac

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

Install and authenticate the Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf auth login
```

Review and accept the [MiniMax-H3 License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before downloading or running the upstream assets.

## 2. Download common assets and the BF16 text encoder

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

This downloads the tokenizer, processor, original BF16 text encoder, Video VAE, and Audio VAE.

## 3. Download the original BF16 DiT

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/transformer/**" \
  --local-dir models/MiniMax-H3
```

The BF16 transformer occupies about 62 GiB on disk, but it does not need to be resident in unified memory as a whole.

## 4. Download the LightX2V 768p BF16 Turbo adapter

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --revision e6346777701aa2b64d42ed058cdd71ae00e7cd52 \
  --local-dir models/Minimax-h3-Turbo-v1.0-4step-768p-bf16
```

Pinned artifact information:

- Size: `1,383,677,808` bytes
- SHA256: `1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c`
- Rank: 128
- Alpha: 128, read automatically from safetensors metadata
- Recommended NFE: 4
- Video sigma shift: 6
- Audio sigma shift: 3

## 5. Run a minimal smoke test first

Confirm model paths, memory behavior, and MP4 muxing before starting the production run:

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

## 6. Generate a five-second 1344×768 video

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

### Important arguments

- `--steps 5`: five sigma-grid points, corresponding to **four DiT forwards / 4 NFE**;
- `--sigma-shift-video 6 --sigma-shift-audio 3`: the published LightX2V 768p Turbo schedule;
- `--stream-block-group-size 2`: retain two BF16 DiT blocks, approximately 2.58 GB, at a time;
- `--no-block-cache`: disable approximate residual caching and preserve complete computation;
- `--memory-limit-gb 24`: the setting used for the measured 24 GB M4 Pro run;
- `--turbo-lora-alpha 128` is unnecessary because alpha is read from metadata, though passing it explicitly is equivalent.

## 7. Run in the background and monitor progress

```bash
nohup caffeinate -dimsu .venv/bin/python scripts/generate.py \
  "your prompt" \
  ...other arguments... \
  > generate.log 2>&1 &

echo $! > generate.pid
```

Watch progress:

```bash
tail -f generate.log
```

Typical output:

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

Validate the generated file:

```bash
ffmpeg -v error -i out/bf16-turbo-1344x768-5s.mp4 -f null -
```

No error output means the complete video and audio streams decoded successfully.

---

# Measured performance on a 24 GB M4 Pro

The measured run produced 1344×768 output with 124 frames at 24 FPS and 5.17 seconds of stereo audio:

| Stage | Measured time |
|---|---:|
| BF16 text encoder | 30.5 s |
| DiT step 1 | 610.7 s |
| DiT step 2 | 599.9 s |
| DiT step 3 | 593.0 s |
| DiT step 4 | 600.9 s |
| All four DiT steps | 2,404.5 s (40 min 4.5 s) |
| Video VAE decode | 427.4 s (7 min 7.4 s) |
| Audio VAE decode | 1.8 s |
| MP4 mux | 2.1 s |
| **End-to-end** | **2,878.7 s (47 min 58.7 s)** |

Additional measurements:

- Mean DiT forward time: **601.1 seconds**;
- peak memory footprint: approximately **15.8 GB**;
- maximum RSS: approximately **6.9 GB**, while Metal/IOSurface memory follows different accounting;
- attention: approximately **50.1%** of non-overlapping profiled time;
- linear projections: approximately **29.6%**;
- model loading: only approximately **0.6%**;
- output: 1344×768 H.264 with 32 kHz stereo AAC, fully decode-validated.

At 768p, exact dense attention and BF16 linear computation dominate runtime, not storage loading. Increasing block-group residency can reduce some switching overhead, but it cannot execute dependent transformer blocks in parallel.

Actual runtime varies with prompt length, chip model, thermals, storage, and background load.

---

# Smaller and faster: Argus calibrated INT8 + native MLX Turbo

In addition to the original BF16 route above, this repository supports and has measured the following complete configuration:

- **DiT:** [`water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8`](https://huggingface.co/water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8);
- **Turbo:** [`water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX`](https://huggingface.co/water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX);
- **Text encoder:** truncated to the 50 layers actually consumed by H3 and streamed layer by layer in MLX 4-bit;
- **VAEs, tokenizer, and processor:** the official MiniMax-H3 FL2VA assets.

The Turbo repository is our published **native BF16 MLX streaming directory**, not an on-the-fly conversion. It is neither merged into nor requantized with the INT8 DiT. It preserves all 518 BF16 tensors and 259 LoRA pairs from the Larry v4-step600 EMA adapter, records `alpha=rank`, and splits the weights into 53 component-level shards.

## 1. Download the Argus INT8 DiT and native MLX Turbo

```bash
hf download water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --revision 70505b09c80e298e684d798d8e0f946937dcfad4 \
  --local-dir models/MiniMax-H3-MLX-Argus-Calibrated-INT8

hf download water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --revision 9771f9c606a50671bb94ee57191461602f02d1fc \
  --local-dir models/MiniMax-H3-Turbo-v4-step600-EMA-MLX
```

To use a Hugging Face mirror, set it before downloading:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

The DiT uses group-size-32 MLX affine INT8 for 254 linear layers while retaining four sensitivity-selected projections in BF16. Its effective files occupy approximately 37.8 GB in decimal units and do not include the text encoder or VAEs.

## 2. Build the low-memory MLX 4-bit text encoder

If the text encoder has not been converted yet:

```bash
python scripts/quantize_text_encoder.py \
  --source models/MiniMax-H3/FL2VA/text_encoder \
  --output models/MiniMax-H3-MLX-TextEncoder-4bit \
  --bits 4 \
  --group-size 64 \
  --num-layers 50
```

The streaming route dequantizes only the embedding rows needed by the prompt and then loads the 50 decoder layers one at a time. It never expands the complete quantized vocabulary table or keeps the complete text encoder resident.

## 3. Production INT8 + native MLX Turbo command

```bash
mkdir -p out profiles

caffeinate -dimsu .venv/bin/python scripts/generate.py \
  "A luminous silver-white fox runs gracefully through an ancient enchanted autumn forest at blue hour, glowing fireflies spiral between moss-covered trees, golden leaves drift slowly through soft volumetric moonlight, cinematic low-angle tracking shot, shallow depth of field, exquisite detailed fur, magical realism, rich teal and amber color grading, smooth coherent natural motion, atmospheric and elegant, no text, no watermark" \
  --checkpoint models/MiniMax-H3/FL2VA \
  --transformer models/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --text-encoder models/MiniMax-H3-MLX-TextEncoder-4bit \
  --turbo-lora models/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --turbo-lora-scale 1.0 \
  --profile quality \
  --low-memory \
  --stream-blocks \
  --resolution 768x448 \
  --duration 5 \
  --steps 9 \
  --seed 42 \
  --memory-pressure-guard \
  --no-block-cache \
  --dense-dequant-profile off \
  --require-muxed-mp4 \
  --forward-profile-json profiles/int8-turbo-768x448-5s.json \
  --output out/int8-turbo-768x448-5s.mp4
```

`--steps 9` produces eight DiT forwards. The native MLX adapter records each component's rank and alpha, so **do not pass** `--turbo-lora-alpha`; the runtime keeps the BF16 LoRA update separate from the INT8 base.

## 4. Measured result on the 24 GB M4 Pro

The command above completed end to end on the same 24 GB M4 Pro MacBook Pro:

| Item | Measured result |
|---|---:|
| Resolution and frames | 768×448, 124 frames, 24 FPS |
| Media duration | 5.1667 s video; 5.152 s stereo audio |
| Text encoder | 23.4 s |
| AdaLN cache | 3.4 s |
| DiT | 8 NFE, mean 154.3 s/NFE |
| **End-to-end** | **1,378.0 s (approximately 23 minutes)** |
| Maximum RSS | **approximately 12.0 GB** |
| Output | H.264 + 32 kHz stereo AAC |

- **[Watch or download the INT8 + MLX Turbo result](examples/int8-turbo-768x448/output-int8-turbo-768x448-5s.mp4)**
- [View the six-frame contact sheet](examples/int8-turbo-768x448/contact-sheet.jpg)
- [Prompt text file](examples/int8-turbo-768x448/prompt.txt)
- Video SHA256: `9912a4f1ed3e8b909cb63ad216a38dfd96a5230ab73c21273a0579f450043037`

The output passed a complete ffmpeg video/audio decode. Its AAC stream is active, non-silent, 32 kHz stereo. This calibrated INT8 release is a measured quality candidate, not a claim of mathematically lossless quantization.

### More measured INT8 + MLX Turbo videos

- **[Nai Long vs. Ultraman Belial: watch or download the 480×864 vertical video](examples/nailong-vs-belial-480x864/nailong-vs-belial-int8-mlx-turbo-480x864-6s.mp4)**
- [Six-frame contact sheet](examples/nailong-vs-belial-480x864/contact-sheet.jpg) · [Full prompt](examples/nailong-vs-belial-480x864/prompt.txt)
- Configuration: full BF16 text encoder, Argus INT8 DiT, native MLX BF16 Turbo, 4 NFE, 480×864, 6.575-second H.264 + AAC
- SHA256: `9c17fdb2e85f37a4f7b836f17c950165378245fc01fca8051837d367fed5b4d8`

## 5. Reproducible Turbo conversion tool

Normally, download `water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX` directly. To revalidate and package the original Larry release yourself:

```bash
python scripts/convert_turbo_lora_to_mlx.py \
  --source /path/to/minimax_h3_turbo_v4_step600_ema.safetensors \
  --config models/MiniMax-H3-MLX-Argus-Calibrated-INT8/config.json \
  --output-dir models/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --source-repo larryvrh/MiniMax-H3-Turbo-Lora \
  --source-revision 43a74557ac3f6539db8e0f2a959d03feb7a81480
```

Conversion changes only the MLX packaging and index layout; it does not change the BF16 tensor values.

---

# Project and model links

- Argus Agent: <https://github.com/lbx154/Argus>
- This project: <https://github.com/Argus-AiTeam/minimax-h3-mac>
- Upstream MiniMax-H3: <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- Argus calibrated MLX INT8 DiT: <https://huggingface.co/water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8>
- Argus native MLX BF16 Turbo: <https://huggingface.co/water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX>
- LightX2V Turbo v1.0 768p: <https://huggingface.co/lightx2v/Minimax-h3-Turbo>
- Original Larry Turbo LoRA release: <https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora>

---

# Acknowledgements

This project builds on the work of the following projects and teams:

- **Argus:** autonomously completed most of the code adaptation, experiments, validation, and documentation in this repository: <https://github.com/lbx154/Argus>
- **Argus-AiTeam:** repository maintenance and continued Apple Silicon optimization: <https://github.com/Argus-AiTeam/minimax-h3-mac>
- **MiniMaxAI:** the MiniMax-H3 model, architecture, and official base weights: <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- **PipeNetwork:** the early MiniMax-H3 MLX port and community foundation that provided an important basis for this Mac localization: <https://github.com/PipeNetwork/minimax-h3-mlx>

Powered by MiniMax H3.

---

## Important notes

- MiniMax-H3 weights remain subject to the upstream license;
- generating five seconds at 1344×768 took nearly 48 minutes on the measured 24 GB M4 Pro and is not real-time inference;
- the open-source path currently uses dense attention because MiniMax’s sparse-attention implementation was not released with the weights;
- “full BF16” means inference uses the original upstream BF16 DiT/text-encoder weights and the complete H3 computation path. It does not alter or bypass the architecture defined by the upstream model.

**A 24 GB MacBook Pro can now run MiniMax-H3 locally, end to end.**
