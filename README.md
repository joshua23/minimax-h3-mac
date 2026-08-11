# MiniMax-H3 on Apple Silicon with MLX

Run MiniMax-H3 video + audio generation on an Apple Silicon Mac using a streamed MLX INT8 DiT and an optional BF16 Turbo adapter.

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

### A. MLX INT8 DiT — required

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

The upstream BF16 DiT is not needed because the MLX INT8 DiT replaces it.

### C. MLX text encoder — required for low-memory generation

Convert the upstream text encoder to a truncated 50-layer MLX 4-bit conditioner:

```bash
python scripts/quantize_text_encoder.py \
  --source models/MiniMax-H3/FL2VA/text_encoder \
  --output models/MiniMax-H3-MLX-TextEncoder-4bit \
  --bits 4 --group-size 64 --num-layers 50
```

This text encoder is a runtime dependency; the Argus calibrated INT8 release quantizes only the DiT.

### D. BF16 Turbo adapter — optional

```bash
hf download water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --local-dir models/MiniMax-H3-Turbo-v4-step600-EMA-MLX
```

The Turbo adapter stays BF16. It is not merged into or requantized with the INT8 DiT.

## 3. Run without Turbo

Start with a small smoke test:

```bash
python scripts/generate.py "A fox running through a misty forest" \
  --checkpoint models/MiniMax-H3/FL2VA \
  --transformer models/MiniMax-H3-MLX-Argus-Calibrated-INT8 \
  --text-encoder models/MiniMax-H3-MLX-TextEncoder-4bit \
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
  --text-encoder models/MiniMax-H3-MLX-TextEncoder-4bit \
  --turbo-lora models/MiniMax-H3-Turbo-v4-step600-EMA-MLX \
  --turbo-lora-scale 1.0 \
  --low-memory --stream-blocks \
  --resolution 64x64 --duration 5 --steps 9 \
  --require-muxed-mp4 --output output-turbo.mp4
```

`--steps 9` means eight denoiser evaluations. Use `--steps 7` for the faster six-evaluation setting. Do not pass `--turbo-lora-alpha` for the native adapter; it records `alpha=rank` automatically.

After the smoke test works, increase `--resolution`. Large production resolutions can be very slow because the open-source H3 release uses dense attention.

## 5. Convert the original Larry Turbo LoRA yourself

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

The converter keeps all 518 tensors in BF16, validates all 259 LoRA pairs, records `alpha=rank`, and creates 53 component shards for streamed loading.

## Model links

- Code: https://github.com/Argus-AiTeam/minimax-h3-mac
- MLX INT8 DiT: https://huggingface.co/water1234/MiniMax-H3-MLX-Argus-Calibrated-INT8
- MLX BF16 Turbo: https://huggingface.co/water1234/MiniMax-H3-Turbo-v4-step600-EMA-MLX
- Upstream MiniMax-H3: https://huggingface.co/MiniMaxAI/MiniMax-H3

Powered by MiniMax H3.
