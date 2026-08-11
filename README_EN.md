# Run Full-Precision MiniMax-H3 Locally on a MacBook

[中文](README.md) | [English](README_EN.md)

> **This project was independently completed by ArgusAgent, with reference to and support from community contributions, and with very little human involvement in the loop.**

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

# Smaller alternative: calibrated INT8 DiT

If you do not want to download the 62 GiB BF16 DiT, download the calibrated MLX INT8 model:

```bash
hf download water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --local-dir models/MiniMax-H3-MLX-Argus-Calibrated-INT8
```

Example:

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

`--steps 21` corresponds to 20 denoiser evaluations. The text encoder still defaults to the upstream BF16 streamed path.

---

# Optional: package the original Larry Turbo LoRA

The original Larry safetensors contains BF16 arrays that MLX can use. This tool validates keys, shapes, and ranks, records `alpha=rank`, and splits the adapter into streamable component files without changing tensor values.

Download: <https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora>

```bash
python scripts/convert_turbo_lora_to_mlx.py \
  --source /path/to/minimax_h3_turbo_v4_step600_ema.safetensors \
  --config models/MiniMax-H3-MLX-Argus-Calibrated-INT8/config.json \
  --output-dir models/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --source-repo larryvrh/MiniMax-H3-Turbo-Lora \
  --source-revision 43a74557ac3f6539db8e0f2a959d03feb7a81480
```

The packager preserves all 518 BF16 tensors, validates 259 LoRA pairs, and writes 53 component shards.

---

# Project and model links

- This project: <https://github.com/Argus-AiTeam/minimax-h3-mac>
- Upstream MiniMax-H3: <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- Argus calibrated MLX INT8 DiT: <https://huggingface.co/water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8>
- LightX2V Turbo v1.0 768p: <https://huggingface.co/lightx2v/Minimax-h3-Turbo>
- Larry Turbo LoRA: <https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora>

---

## Important notes

- MiniMax-H3 weights remain subject to the upstream license;
- generating five seconds at 1344×768 took nearly 48 minutes on the measured 24 GB M4 Pro and is not real-time inference;
- the open-source path currently uses dense attention because MiniMax’s sparse-attention implementation was not released with the weights;
- “full BF16” means inference uses the original upstream BF16 DiT/text-encoder weights and the complete H3 computation path. It does not alter or bypass the architecture defined by the upstream model.

**A 24 GB MacBook Pro can now run MiniMax-H3 locally, end to end.**
