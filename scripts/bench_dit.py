"""Benchmark one MiniMax-H3 transformer block at realistic packed-sequence lengths.

The 50 blocks are identical, so timing one and multiplying is an accurate estimate of a denoising
step and needs only ~1/50th of the memory. This answers the question that decides whether the port
is usable on Apple Silicon at all: MiniMax has not released its sparse-attention implementation,
so inference runs **full** attention over a sequence that is tens of thousands of rows long.

    ./.venv/bin/python scripts/bench_dit.py
    ./.venv/bin/python scripts/bench_dit.py --seconds 15 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import TransformerBlock
from minimax_h3_mlx.packing import (
    FPS,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)

DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}


def sequence_length(seconds: float, aspect: tuple[int, int], num_text_tokens: int, cfg: DiTConfig):
    """Rows of the packed sequence for one request, using the real geometry."""
    height, width = resolve_canvas_size(*aspect)
    frames = align_num_frames(int(round(seconds * FPS)))
    latent_frames = video_latent_num_frames(frames)
    lh, lw = height // 16, width // 16  # video VAE is 16x spatial
    _, ph, pw = cfg.patch_size
    rows_per_frame = (lh // ph) * (lw // pw)
    video_rows = latent_frames * rows_per_frame
    audio_rows = audio_latent_num_frames(frames) * 2
    total = num_text_tokens + video_rows + audio_rows
    return {
        "canvas": (height, width),
        "frames": frames,
        "latent_frames": latent_frames,
        "latent_hw": (lh, lw),
        "rows_per_frame": rows_per_frame,
        "video_rows": video_rows,
        "audio_rows": audio_rows,
        "text_rows": num_text_tokens,
        "sequence_length": total,
    }


def bench_block(cfg: DiTConfig, seq_len: int, dtype: mx.Dtype, warmup: int, iters: int) -> float:
    block = TransformerBlock(cfg)
    block.set_dtype(dtype)
    mx.eval(block.parameters())

    x = mx.random.normal((1, seq_len, cfg.hidden_size)).astype(dtype)
    modulation = tuple(
        mx.random.normal((3, cfg.hidden_size)).astype(dtype) for _ in range(6)
    )
    adaln_indices = mx.zeros((seq_len,), dtype=mx.int32)
    rotary_dim = cfg.rotary_dim
    cos = mx.random.normal((seq_len, rotary_dim)).astype(dtype)
    sin = mx.random.normal((seq_len, rotary_dim)).astype(dtype)

    for _ in range(warmup):
        out = block(x, modulation, adaln_indices, (cos, sin))
        mx.eval(out)

    mx.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        out = block(x, modulation, adaln_indices, (cos, sin))
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - started) / iters


def human(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, nargs="+", default=[5.0, 10.0, 15.0])
    parser.add_argument("--aspect", type=int, nargs=2, default=(16, 9))
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40, help="denoising steps for the estimate")
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--probe", type=int, nargs="*", default=None,
                        help="benchmark these raw sequence lengths instead of real geometry")
    args = parser.parse_args()

    cfg = DiTConfig()
    dtype = DTYPES[args.dtype]
    print(f"MiniMax-H3 DiT: {cfg.num_layers} blocks, hidden {cfg.hidden_size}, "
          f"{cfg.num_attention_heads}x{cfg.attention_head_dim} heads, ffn {cfg.ffn_hidden_size}")
    print(f"dtype {args.dtype}, {args.steps} denoising steps assumed\n")

    if args.probe is not None:
        cases = [({"sequence_length": n}, f"{n} rows") for n in args.probe]
    else:
        cases = []
        for secs in args.seconds:
            geo = sequence_length(secs, tuple(args.aspect), args.text_tokens, cfg)
            label = (f"{secs:g}s {geo['canvas'][1]}x{geo['canvas'][0]} "
                     f"({geo['latent_frames']} latent frames)")
            cases.append((geo, label))

    for geo, label in cases:
        seq = geo["sequence_length"]
        try:
            per_block = bench_block(cfg, seq, dtype, args.warmup, args.iters)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next size
            print(f"{label}: seq_len {seq:,} -> FAILED: {type(exc).__name__}: {exc}")
            continue
        per_step = per_block * cfg.num_layers
        total = per_step * args.steps
        print(f"{label}")
        if "video_rows" in geo:
            print(f"  rows: {geo['video_rows']:,} video + {geo['audio_rows']:,} audio "
                  f"+ {geo['text_rows']:,} text = {seq:,}")
        print(f"  block {per_block * 1000:8.1f} ms   step {human(per_step)}   "
              f"{args.steps} steps {human(total)}")
        print(f"  peak memory {mx.get_peak_memory() / 1e9:.1f} GB")
        mx.reset_peak_memory()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
