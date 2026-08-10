#!/usr/bin/env python3
"""Verify official-weight low-memory gates before MiniMax-H3 5s generation.

This script intentionally performs intermediate checks that are stronger than
synthetic unit tests but cheaper than a full denoising ladder:

* ``prompt-release`` loads the quantized H3 text encoder, runs the real prompt
  encoder, detaches the BF16 conditioning tensor, releases the component through
  the same pipeline release helper used by low-memory generation, and proves the
  detached tensor remains materialized.
* ``streaming-components`` loads the streamed quantized DiT, touches static input
  and output projections, loads and evaluates the first and last blocks with
  their AdaLN projections, and reports Turbo LoRA as loaded only when a real
  adapter path is configured.

The command emits one JSON object to stdout and exits 0 only when the requested
gate is verified. It never downloads assets.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1
VALIDATION_FAILURE = 3
USAGE_ERROR = 64
DEFAULT_PROMPT = "a red fox leaps over a mossy log, natural motion, synchronized ambient sound"


def _repo_rel(path: str | Path) -> str:
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else ROOT / candidate
    try:
        return str(absolute.relative_to(ROOT))
    except ValueError:
        return str(absolute)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _shape(value: Any) -> list[int]:
    return [int(dim) for dim in value.shape]


def _array_info(value: mx.array) -> dict[str, Any]:
    return {
        "shape": _shape(value),
        "dtype": str(value.dtype),
        "nbytes": int(value.nbytes),
    }


def _mlx_memory_snapshot() -> dict[str, int | None]:
    def call(name: str) -> int | None:
        func = getattr(mx, name, None)
        if func is None and hasattr(mx, "metal"):
            func = getattr(mx.metal, name, None)
        if func is None:
            return None
        try:
            return int(func())
        except Exception:
            return None

    return {
        "peak_bytes": call("get_peak_memory"),
        "active_bytes": call("get_active_memory"),
        "cache_bytes": call("get_cache_memory"),
    }


def _set_mlx_limits(memory_limit_gb: float) -> None:
    try:
        mx.set_cache_limit(0)
    except Exception:
        pass
    if memory_limit_gb > 0:
        try:
            mx.set_memory_limit(int(memory_limit_gb * 1e9))
        except Exception:
            pass


def _cleanup() -> None:
    gc.collect()
    try:
        mx.synchronize()
    except Exception:
        pass
    try:
        mx.clear_cache()
    except Exception:
        pass


def _prompt_release(args: argparse.Namespace) -> dict[str, Any]:
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline, detach_bfloat16
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    _set_mlx_limits(args.memory_limit_gb)
    text_encoder_dir = Path(args.text_encoder)
    encoder = MiniMaxH3TextEncoder(text_encoder_dir, load_vision=False, verbose=args.verbose)
    pipe = MiniMaxH3Pipeline(None, encoder, None, None)

    prompt_embeds, token_tags = pipe.text_encoder.encode(args.prompt, images=None)
    detached = detach_bfloat16(prompt_embeds)
    token_tags_copy = np.array(token_tags, copy=True)

    pipe._release_component("text_encoder")
    released = pipe.text_encoder is None
    detached_sum = float(np.array(mx.sum(detached.astype(mx.float32))).item())
    mx.eval(detached)
    usable_after_release = detached.shape[0] == 1 and detached.shape[-1] == 5120 and token_tags_copy.size > 0
    result = {
        "schema_version": SCHEMA_VERSION,
        "gate": "prompt_release",
        "ok": bool(released and usable_after_release),
        "text_encoder_dir": _repo_rel(text_encoder_dir),
        "prompt_chars": len(args.prompt),
        "prompt_embedding": _array_info(detached),
        "token_tag_count": int(token_tags_copy.size),
        "unique_token_tags": sorted(int(value) for value in np.unique(token_tags_copy).tolist()),
        "detached_checksum": detached_sum,
        "component_released": released,
        "detached_usable_after_release": bool(usable_after_release),
        "mlx_memory_after_release": _mlx_memory_snapshot(),
    }
    del detached, prompt_embeds, token_tags, token_tags_copy, pipe, encoder
    _cleanup()
    return result


def _projection_checks(model: Any, timesteps: mx.array) -> dict[str, Any]:
    from minimax_h3_mlx.adaln import final_layer_modulation
    from minimax_h3_mlx.dit import param_dtype

    cfg = model.config
    video_in = mx.zeros((1, 1, cfg.video_patch_dim), dtype=mx.float32)
    audio_in = mx.zeros((1, 1, cfg.audio_latents_dim), dtype=mx.float32)
    text_in = mx.zeros((1, 1, cfg.text_dim), dtype=mx.bfloat16)
    video_projection = model.video_patch_proj(video_in.astype(param_dtype(model.video_patch_proj)))
    audio_projection = model.audio_patch_proj(audio_in.astype(param_dtype(model.audio_patch_proj)))
    condition_projection = model.condition_proj(text_in.astype(param_dtype(model.condition_proj)))

    hidden = mx.zeros((1, 3, cfg.hidden_size), dtype=mx.bfloat16)
    video_out = model.final_layer.video_out(hidden.astype(param_dtype(model.final_layer.video_out)))
    audio_out = model.final_layer.audio_out(hidden.astype(param_dtype(model.final_layer.audio_out)))
    final_shift, final_scale = final_layer_modulation(model, timesteps, dtype=mx.bfloat16)
    mx.eval(video_projection, audio_projection, condition_projection, video_out, audio_out, final_shift, final_scale)

    return {
        "video_patch_proj": _array_info(video_projection),
        "audio_patch_proj": _array_info(audio_projection),
        "condition_proj": _array_info(condition_projection),
        "final_video_out": _array_info(video_out),
        "final_audio_out": _array_info(audio_out),
        "final_adaln_shift": _array_info(final_shift),
        "final_adaln_scale": _array_info(final_scale),
    }


def _block_check(model: Any, provider: Any, index: int, timesteps: mx.array) -> dict[str, Any]:
    from minimax_h3_mlx.config import MODALITY_NUM
    from minimax_h3_mlx.dit import timestep_embedding

    cfg = provider.config
    before = int(provider.logical_bytes_loaded)
    block = provider.load_block(index, include_adaln=True)
    temb = model.time_embedder(timestep_embedding(timesteps, cfg.timestep_input_dim))
    modulation = tuple(t.astype(mx.bfloat16) for t in block.adaln_proj(temb))
    mx.eval(temb, modulation)

    sequence_length = MODALITY_NUM
    hidden = mx.zeros((1, sequence_length, cfg.hidden_size), dtype=mx.bfloat16)
    adaln_indices = mx.array(np.arange(sequence_length, dtype=np.int32))
    position_ids = mx.array(
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int32)[:sequence_length]
    )
    output = block(
        hidden,
        modulation,
        adaln_indices,
        model.rope(position_ids),
        None,
        provider.current_lora,
    )
    mx.eval(output)
    lora_loaded = provider.turbo_lora is not None and provider.current_lora is not None
    after = int(provider.logical_bytes_loaded)
    return {
        "index": int(index),
        "logical_bytes_loaded_delta": after - before,
        "adaln_tables": [_array_info(item) for item in modulation],
        "block_forward_output": _array_info(output),
        "turbo_lora_loaded": bool(lora_loaded),
    }


def _streaming_components(args: argparse.Namespace) -> dict[str, Any]:
    from minimax_h3_mlx.streaming import load_streaming_dit

    _set_mlx_limits(args.memory_limit_gb)
    transformer = Path(args.transformer)
    turbo_lora_path = Path(args.turbo_lora) if args.turbo_lora else None
    model, provider = load_streaming_dit(
        transformer,
        turbo_lora_path=turbo_lora_path,
        turbo_lora_alpha=args.turbo_lora_alpha,
        turbo_lora_scale=args.turbo_lora_scale,
        verbose=args.verbose,
    )
    cfg = provider.config
    timesteps = mx.array(np.array([0.0, 1.0], dtype=np.float32))

    projections = _projection_checks(model, timesteps)
    block_indices = [0] if cfg.num_layers == 1 else [0, cfg.num_layers - 1]
    blocks = [_block_check(model, provider, index, timesteps) for index in block_indices]

    if provider.turbo_lora is None:
        turbo_status = {
            "configured": False,
            "status": "not_configured",
            "verified": False,
        }
    else:
        turbo_status = {
            "configured": True,
            "status": "loaded",
            "verified": all(item["turbo_lora_loaded"] for item in blocks),
            "path": _repo_rel(turbo_lora_path),
            "rank": int(provider.turbo_lora.rank),
            "multiplier": float(provider.turbo_lora.multiplier),
            "refiner_lora_count": len(provider.refiner_loras or []),
        }

    block_ok = len(blocks) == len(block_indices) and all(item["block_forward_output"]["shape"][-1] == cfg.hidden_size for item in blocks)
    projections_ok = (
        projections["video_patch_proj"]["shape"][-1] == cfg.hidden_size
        and projections["audio_patch_proj"]["shape"][-1] == cfg.hidden_size
        and projections["condition_proj"]["shape"][-1] == cfg.hidden_size
        and projections["final_video_out"]["shape"][-1] == cfg.video_patch_dim
        and projections["final_audio_out"]["shape"][-1] == cfg.audio_latents_dim
    )
    turbo_ok = True if provider.turbo_lora is None else bool(turbo_status["verified"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "gate": "streaming_components",
        "ok": bool(block_ok and projections_ok and turbo_ok),
        "transformer_dir": _repo_rel(transformer),
        "block_count": int(cfg.num_layers),
        "checked_block_indices": block_indices,
        "projections": projections,
        "blocks": blocks,
        "turbo_lora": turbo_status,
        "logical_bytes_loaded_total": int(provider.logical_bytes_loaded),
        "mlx_memory_after_checks": _mlx_memory_snapshot(),
    }
    del model, provider, projections, blocks
    _cleanup()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    sub = parser.add_subparsers(dest="command", required=True)

    prompt = sub.add_parser("prompt-release", help="encode the real prompt and release text encoder")
    prompt.add_argument("--text-encoder", default="models/MiniMax-H3/FL2VA/text_encoder-mlx-4bit")
    prompt.add_argument("--prompt", default=DEFAULT_PROMPT)

    streaming = sub.add_parser("streaming-components", help="verify streamed DiT/AdaLN/projections/Turbo access")
    streaming.add_argument("--transformer", default="models/MiniMax-H3-MLX-4bit")
    streaming.add_argument("--turbo-lora", default=None)
    streaming.add_argument("--turbo-lora-alpha", type=float, default=8.0)
    streaming.add_argument("--turbo-lora-scale", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prompt-release":
            payload = _prompt_release(args)
        elif args.command == "streaming-components":
            payload = _streaming_components(args)
        else:  # argparse should prevent this.
            parser.error(f"unknown command {args.command!r}")
            return USAGE_ERROR
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "gate": getattr(args, "command", "unknown"),
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        traceback.print_exc(file=sys.stderr)
        return VALIDATION_FAILURE

    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0 if payload.get("ok") else VALIDATION_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
