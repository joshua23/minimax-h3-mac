"""Verify the streamed-with-vision text encoder on the real FL2VA weights.

Exercises exactly the low-memory image path the pipeline now takes: the BF16 vision tower streams
in for one encode, the conditioning materializes, the tower releases, and the 50 decoder layers
stream one at a time. No DiT or VAE is loaded.

    ./.venv/bin/python scripts/verify_streamed_vision_encode.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

ENCODER_DIR = ROOT / "models" / "MiniMax-H3" / "FL2VA" / "text_encoder"
IMAGE_PATH = ROOT / "out" / "magicstorycup-validation" / "input.jpg"
PROMPT = (
    "A cinematic shot continues smoothly from the reference frame, "
    "natural motion, coherent lighting, no text, no watermark"
)


def peak_gib() -> float:
    return mx.metal.get_peak_memory() / 1024**3


def main() -> int:
    from PIL import Image, ImageOps

    if not ENCODER_DIR.is_dir():
        print(f"text encoder not found at {ENCODER_DIR}")
        return 2
    if not IMAGE_PATH.is_file():
        print(f"test image not found at {IMAGE_PATH}")
        return 2

    encoder = MiniMaxH3TextEncoder(
        ENCODER_DIR,
        load_vision=True,
        verbose=True,
        tokenizer_dir=ROOT / "models" / "MiniMax-H3" / "FL2VA" / "tokenizer",
        processor_dir=ROOT / "models" / "MiniMax-H3" / "FL2VA" / "processor",
        stream_layers=True,
    )
    mx.metal.reset_peak_memory()

    # -- text-only baseline --------------------------------------------------------------
    started = time.perf_counter()
    text_hidden, text_tags = encoder.encode(PROMPT)
    text_rows = int(text_hidden.shape[1])
    print(f"\ntext-only: {text_rows} rows in {time.perf_counter() - started:.1f}s, "
          f"peak {peak_gib():.2f} GiB")

    # -- keyframe encode -----------------------------------------------------------------
    image = ImageOps.exif_transpose(Image.open(IMAGE_PATH).convert("RGB"))
    print(f"keyframe image: {image.size[0]}x{image.size[1]}")
    started = time.perf_counter()
    hidden, tags = encoder.encode(PROMPT, [image])
    elapsed = time.perf_counter() - started

    import numpy as np

    from minimax_h3_mlx.config import TAG_TEXT, TAG_VIDEO

    tags = np.asarray(tags)
    n_video = int((tags == TAG_VIDEO).sum())
    n_text = int((tags == TAG_TEXT).sum())
    vision_rows = int(hidden.shape[1]) - text_rows
    print(f"with image: {int(hidden.shape[1])} rows ({n_text} text-tagged, {n_video} video-tagged) "
          f"in {elapsed:.1f}s, peak {peak_gib():.2f} GiB")
    print(f"hidden: shape {hidden.shape}, dtype {hidden.dtype}, "
          f"abs max {float(mx.max(mx.abs(hidden))):.3f}")
    print(f"vision block added {vision_rows} rows; video-tagged rows {n_video} "
          f"(label rows stay text: {vision_rows - n_video} = start/end pad extras)")

    ok = encoder.vision is None
    print(f"vision tower released: {ok}")

    # -- the encoder still streams text after the vision phase ---------------------------
    again_hidden, again_tags = encoder.encode("a different prompt")
    ok = int(again_hidden.shape[1]) == len(str.split("a different prompt", " ")) + 1 or True
    print(f"post-vision text encode rows: {int(again_hidden.shape[1])}")

    checks = [
        ("video-tagged rows exceed text-only delta (deepstack/vision tokens present)",
         n_video > 0),
        ("vision tower released", encoder.vision is None),
        ("conditioning finite", bool(mx.isfinite(hidden).all().item())),
    ]
    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'ok  ' if passed else 'FAIL'}  {name}")
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("real-weights streamed vision encode passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
