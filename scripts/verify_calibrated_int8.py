#!/usr/bin/env python3
"""Strictly load and execute a real-prompt full DiT forward from calibrated INT8."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_quant import build_case, forward, timestep_plan  # noqa: E402
from minimax_h3_mlx.load import load_dit  # noqa: E402
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--prompt",
        default="A red fox leaps over a mossy log in a misty forest at dawn.",
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--duration", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=707)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")

    started = time.perf_counter()
    checkpoint = Path(args.checkpoint)
    text_encoder = MiniMaxH3TextEncoder(
        args.text_encoder,
        load_vision=False,
        verbose=True,
        tokenizer_dir=checkpoint / "tokenizer",
        processor_dir=checkpoint / "processor",
    )
    embeds, tags = text_encoder.encode(args.prompt)
    embeds_host = np.array(embeds.astype(mx.bfloat16), copy=True)
    del text_encoder
    mx.clear_cache()

    transformer_path = Path(args.transformer)
    model = load_dit(transformer_path, strict=True, verbose=True)
    quantized = [
        (path, module)
        for path, module in tree_flatten(
            model.leaf_modules(),
            is_leaf=nn.Module.is_module,
        )
        if isinstance(module, nn.QuantizedLinear)
    ]
    if not quantized:
        raise RuntimeError("strictly loaded artifact contains no QuantizedLinear modules")
    wrong_dtype = [
        path for path, module in quantized if module.weight.dtype != mx.uint32
    ]
    if wrong_dtype:
        raise TypeError(f"quantized layers without packed uint32 weights: {wrong_dtype[:8]}")

    layout, video_rows, audio_rows, video_sched, audio_sched = build_case(
        len(tags),
        args.height,
        args.width,
        args.duration,
        args.steps,
        model.config.latents_dim,
        model.config.audio_latents_dim,
        model.config.patch_size,
        args.seed,
    )
    table, plan = timestep_plan(layout, video_sched, audio_sched)
    video, audio = forward(
        model,
        None,
        layout,
        video_rows,
        audio_rows,
        mx.array(embeds_host).astype(mx.bfloat16),
        table,
        plan[0],
    )
    mx.eval(video, audio)
    video_host = np.asarray(video.astype(mx.float32))
    audio_host = np.asarray(audio.astype(mx.float32))
    finite = bool(np.isfinite(video_host).all() and np.isfinite(audio_host).all())
    if not finite:
        raise RuntimeError("strictly loaded full calibrated DiT produced non-finite output")

    report = {
        "schema_version": 1,
        "artifact": str(transformer_path.resolve()),
        "quant_config_sha256": sha256(transformer_path / "quant_config.json"),
        "strict_load": True,
        "quantized_linear_count": len(quantized),
        "packed_weight_dtype": "uint32",
        "prompt": args.prompt,
        "token_count": len(tags),
        "case": {
            "height": args.height,
            "width": args.width,
            "duration_seconds": args.duration,
            "steps": args.steps,
            "seed": args.seed,
            "sequence_rows": layout.sequence_length,
        },
        "full_dit_forward": {
            "video_shape": list(video_host.shape),
            "audio_shape": list(audio_host.shape),
            "video_l2": float(np.linalg.norm(video_host.astype(np.float64))),
            "audio_l2": float(np.linalg.norm(audio_host.astype(np.float64))),
            "finite": finite,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "passed": True,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PASS: strict load, {len(quantized)} QuantizedLinear modules, "
        f"finite full DiT forward; report: {output.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
