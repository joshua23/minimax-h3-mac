# MiniMax-H3 on Apple Silicon with MLX

Run MiniMax-H3 video + audio generation on an Apple Silicon Mac using a streamed MLX INT8 or original BF16 DiT, a full-precision streamed text encoder, and optional BF16 Turbo adapters.

## 1. Install

```bash
git clone https://github.com/Argus-AiTeam/minimax-h3-mac.git
cd minimax-h3-mac

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

Install the Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub
hf auth login
```

You must review and accept the [MiniMax-H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before downloading or running the upstream model assets.

## 2. Download the required models

Create a model directory:

```bash
mkdir -p models
```

### A. MLX INT8 DiT — required for the INT8 route

```bash
hf download water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --local-dir models/MiniMax-H3-MLX-Argus-Calibrated-INT8
```

This repository contains the DiT/transformer only.

### B. MiniMax-H3 tokenizer, processor and VAEs — required

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/model_index.json" \
            "FL2VA/tokenizer/**" \
            "FL2VA/processor/**" \
            "FL2VA/text_encoder/**" \
            "FL2VA/video_vae/**" \
            "FL2VA/audio_vae/**" \
  --local-dir models/MiniMax-H3
```

The upstream BF16 DiT is not needed for the INT8 route. Download it separately only if you want
the full-BF16 route documented below.

### C. Full-precision text encoder — included in step B

Low-memory generation now uses the upstream `FL2VA/text_encoder` weights by default. It loads the
embedding only while producing the prompt rows, then evaluates the 50 conditioning layers through a
single reusable layer slot. Only one decoder layer is resident at a time, and the complete text
encoder is released before the DiT is loaded. No 4-bit text-encoder conversion is required.

A converted text encoder may still be selected explicitly with `--text-encoder`, but it is no longer
the default quality path. The Argus calibrated INT8 release quantizes only the DiT.

### D. BF16 Turbo adapter — optional

For the native MLX Larry adapter:

```bash
hf download water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --local-dir models/MiniMax-H3-Turbo-v4-step600-EMA-MLX
```

For the LightX2V 4-step v1.0 768p adapter used by the measured full-BF16 recipe:

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --revision e6346777701aa2b64d42ed058cdd71ae00e7cd52 \
  --local-dir models/Minimax-h3-Turbo-v1.0-4step-768p-bf16
```

The pinned file is 1,383,677,808 bytes with SHA256
`1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c`.
Both Turbo adapters stay BF16; they are streamed rather than merged into or requantized with the
base DiT.

### E. Original upstream BF16 DiT — required only for the full-BF16 recipe

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --include "FL2VA/transformer/**" \
  --local-dir models/MiniMax-H3
```

The BF16 transformer occupies about 62 GiB on disk. It does not need to fit in unified memory as a
whole: low-memory mode retains the 1.72 GB static portion and streams the 50 main blocks in bounded
groups.

## 3. Run without Turbo

Start with a small smoke test:

```bash
python scripts/generate.py "A fox running through a misty forest" \
  --checkpoint models/MiniMax-H3/FL2VA \
  --transformer models/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --low-memory --stream-blocks \
  --resolution 64x64 --duration 5 --steps 21 \
  --require-muxed-mp4 --output output.mp4
```

`--steps 21` means 20 denoiser evaluations in this runtime.

## 4. Run with the BF16 Turbo adapter

```bash
python scripts/generate.py "A fox running through a misty forest" \
  --checkpoint models/MiniMax-H3/FL2VA \
  --transformer models/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --turbo-lora models/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --turbo-lora-scale 1.0 \
  --low-memory --stream-blocks \
  --resolution 64x64 --duration 5 --steps 9 \
  --require-muxed-mp4 --output output-turbo.mp4
```

`--steps 9` means eight denoiser evaluations. Use `--steps 7` for the faster six-evaluation setting. Do not pass `--turbo-lora-alpha` for the native adapter; it records `alpha=rank` automatically.

After the smoke test works, increase `--resolution`. Large production resolutions can be very slow because the open-source H3 release uses dense attention.

## 5. Full BF16 DiT + full BF16 text encoder + LightX2V Turbo at 768p

This is the measured quality-oriented recipe. Low-memory mode recognizes that the upstream DiT has
no `quant_config.json` and streams its original BF16 arrays without quantization or numerical
conversion. The upstream BF16 text-encoder checkpoint is also used by default: its embedding is
loaded transiently, the 50 layers H3 actually reads are evaluated through one reusable layer slot,
and the encoder is released before the DiT stage.

Start with a wiring test by changing the production command below to:

```bash
--resolution 64x64 --duration 1
```

For the real 768p, five-second run:

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

`--steps 5` is four denoiser evaluations (4 NFE). The adapter metadata supplies `alpha=128`, so
`--turbo-lora-alpha` is unnecessary; explicitly passing `--turbo-lora-alpha 128` is equivalent. Its
published scheduler contract is video shift 6 and audio shift 3, which is why both overrides are
present. `--stream-block-group-size 2` retains about 2.58 GB of main-block BF16 weights at once;
blocks still execute sequentially.

To watch a redirected/background run:

```bash
tail -f path/to/generate.log
```

A successful run ends with `wrote ...mp4`, `wrote forward profile ...json`, and the measured seconds
per step/total minutes. Validate the resulting container with:

```bash
ffmpeg -v error -i out/bf16-turbo-1344x768-5s.mp4 -f null -
```

### Measured runtime on a 24 GB M4 Pro

Measured on a MacBook Pro (`Mac16,8`), Apple M4 Pro with 14 CPU cores and 24 GB unified memory. The
exact command above produced 124 video frames at 24 fps plus 5.17 seconds of stereo audio:

| Stage | Measured time |
|---|---:|
| Full BF16 text encoding | 30.5 s |
| DiT step 1 | 610.7 s |
| DiT step 2 | 599.9 s |
| DiT step 3 | 593.0 s |
| DiT step 4 | 600.9 s |
| All four DiT steps | 2,404.5 s (40 min 4.5 s) |
| Video VAE decode | 427.4 s (7 min 7.4 s) |
| Audio VAE decode | 1.8 s |
| MP4 mux | 2.1 s |
| **End-to-end** | **2,878.7 s (47 min 58.7 s)** |

Average DiT time was **601.1 seconds per step**. The measured peak memory footprint was about
**15.8 GB** (maximum RSS about 6.9 GB; Metal/IOSurface memory uses a different accounting path).
The generated MP4 was 1344x768 H.264 with stereo AAC and decoded successfully. Attention consumed
about 50.1% of profiled non-overlapping time and linear projections 29.6%; model loading was only
about 0.6%, so keeping more blocks resident is unlikely to produce a large speedup at this canvas.
Runtime varies with prompt token count, thermals, storage and background load.

### Measured example output

Prompt:

```text
A cinematic red panda running through a misty bamboo forest, detailed fur, natural lighting, smooth coherent motion, cinematic camera movement
```

- [Watch/download the generated 1344x768 MP4](examples/bf16-turbo-768p/output-bf16-turbo-1344x768-5s.mp4)
- [Prompt text file](examples/bf16-turbo-768p/prompt.txt)
- Video SHA256: `b52a18e32f58fdfa387798e7cc1425bcb14f795797ec7168323b479e7bd49c85`

The MP4 contains the generated H.264 video and AAC stereo audio; no separate WAV is required.

## 6. Validate and shard the original Larry Turbo LoRA for MLX streaming

The original Larry safetensors already contains BF16 arrays that MLX can read, and its fused H3 tensor values are not numerically converted. This packaging step validates every key/shape/rank, records `alpha=rank`, and splits the monolithic 744 MB file into per-component shards so a 24 GB Mac loads only the current block. Runtimes with native support for Larry's raw key layout may use the original file directly; this repository's low-memory path expects the indexed directory produced below.

Download `minimax_h3_turbo_v4_step600_ema.safetensors` from:

https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora

Then run:

```bash
python scripts/convert_turbo_lora_to_mlx.py \
  --source /path/to/minimax_h3_turbo_v4_step600_ema.safetensors \
  --config models/MiniMax-H3-MLX-Argus-Calibrated-INT8/config.json \
  --output-dir models/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --source-repo larryvrh/MiniMax-H3-Turbo-Lora \
  --source-revision 43a74557ac3f6539db8e0f2a959d03feb7a81480
```

The packager keeps all 518 tensors in BF16 and exactly unchanged, validates all 259 LoRA pairs, records `alpha=rank`, and creates 53 component shards for streamed loading.

## Model links

- Code: https://github.com/Argus-AiTeam/minimax-h3-mac
- MLX INT8 DiT: https://huggingface.co/water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8
- MLX BF16 Turbo: https://huggingface.co/water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX
- LightX2V Turbo v1.0 768p: https://huggingface.co/lightx2v/Minimax-h3-Turbo
- Upstream MiniMax-H3/BF16 DiT and text encoder: https://huggingface.co/MiniMaxAI/MiniMax-H3

Powered by MiniMax H3.
