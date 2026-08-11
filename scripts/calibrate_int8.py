#!/usr/bin/env python3
"""Record disjoint real MiniMax-H3 DiT activations for calibrated INT8."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_quant import build_case, forward, timestep_plan  # noqa: E402
from minimax_h3_mlx.activation_quant import (  # noqa: E402
    ActivationRecorder,
    bind_activation_paths,
)
from minimax_h3_mlx.dit import timestep_embedding  # noqa: E402
from minimax_h3_mlx.load import load_dit  # noqa: E402
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler  # noqa: E402
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder  # noqa: E402


CASES = (
    {
        "split": "calibration",
        "prompt": "A red fox leaps over a mossy log in a misty forest at dawn.",
        "seed": 101,
        "height": 256,
        "width": 256,
        "step_index": 0,
    },
    {
        "split": "calibration",
        "prompt": "Waves crash against black volcanic rocks under a grey sky.",
        "seed": 202,
        "height": 192,
        "width": 320,
        "step_index": 2,
    },
    {
        "split": "calibration",
        "prompt": "A street musician plays saxophone on a rainy neon-lit corner.",
        "seed": 303,
        "height": 320,
        "width": 192,
        "step_index": 4,
    },
    {
        "split": "holdout",
        "prompt": "Steam rises from a bowl of noodles on a wooden table.",
        "seed": 404,
        "height": 192,
        "width": 256,
        "step_index": 1,
    },
    {
        "split": "holdout",
        "prompt": "An origami robot walks across a sunlit library desk.",
        "seed": 505,
        "height": 256,
        "width": 192,
        "step_index": 3,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _is_adaln_projection(path: str) -> bool:
    return ".adaln_proj.linear" in path


def validate_cases(steps: int) -> dict[str, list[float]]:
    calibration = [case for case in CASES if case["split"] == "calibration"]
    holdout = [case for case in CASES if case["split"] == "holdout"]
    if not calibration or not holdout:
        raise ValueError("both calibration and holdout cases are required")
    calibration_prompts = {str(case["prompt"]) for case in calibration}
    holdout_prompts = {str(case["prompt"]) for case in holdout}
    calibration_seeds = {int(case["seed"]) for case in calibration}
    holdout_seeds = {int(case["seed"]) for case in holdout}
    if calibration_prompts & holdout_prompts or calibration_seeds & holdout_seeds:
        raise ValueError("calibration and holdout prompts/seeds must be disjoint")
    schedules = {}
    for name, shift in (("video", 12.0), ("audio", 3.0)):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(steps)
        schedules[name] = [float(value) for value in scheduler.timesteps.tolist()]
    schedule_length = min(len(values) for values in schedules.values())
    invalid = [
        case["step_index"]
        for case in CASES
        if not 0 <= int(case["step_index"]) < schedule_length
    ]
    if invalid:
        raise ValueError(
            f"case step indices outside the {schedule_length} model evaluations "
            f"produced by --steps={steps}: {invalid}"
        )
    split_timesteps: dict[str, set[float]] = {"calibration": set(), "holdout": set()}
    for case in CASES:
        split = str(case["split"])
        index = int(case["step_index"])
        split_timesteps[split].update(schedule[index] for schedule in schedules.values())
    overlap = split_timesteps["calibration"] & split_timesteps["holdout"]
    if overlap:
        raise ValueError(f"AdaLN calibration/holdout timesteps overlap: {sorted(overlap)}")
    return {split: sorted(values) for split, values in split_timesteps.items()}


def capture_disjoint_adaln_inputs(
    teacher,
    recorder: ActivationRecorder,
    split_timesteps: dict[str, list[float]],
) -> None:
    """Record only preregistered, split-disjoint real scheduler inputs for AdaLN."""

    recorder.set_path_predicate(_is_adaln_projection)
    for split in ("calibration", "holdout"):
        timesteps = mx.array(split_timesteps[split], dtype=mx.float32)
        temb = teacher.time_embedder(
            timestep_embedding(timesteps, teacher.config.timestep_input_dim)
        )
        mx.eval(temb)
        recorder.set_split(split)
        with recorder:
            for block in teacher.blocks:
                modulation = block.adaln_proj(temb)
                mx.eval(modulation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="pinned FL2VA source directory")
    parser.add_argument("--text-encoder", required=True, help="deployed MLX text encoder directory")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--duration", type=float, default=0.2)
    parser.add_argument("--max-rows-per-call", type=int, default=8)
    parser.add_argument("--max-rows-per-layer", type=int, default=64)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    try:
        adaln_split_timesteps = validate_cases(args.steps)
    except ValueError as exc:
        parser.error(str(exc))

    source = Path(args.checkpoint)
    text_encoder_path = Path(args.text_encoder)
    transformer = source / "transformer"
    manifest = {
        "schema_version": 1,
        "algorithm": "diagonal-hessian-activation-aware-affine-int8",
        "bits": 8,
        "group_size": 32,
        "mode": "affine",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model": "MiniMaxAI/MiniMax-H3",
            "revision": args.source_revision,
            "transformer_config_sha256": sha256(transformer / "config.json"),
        },
        "implementation": {
            "repository": "https://github.com/Argus-AiTeam/minimax-h3-mac",
            **repository_state(),
        },
        "text_encoder": {
            "path": str(text_encoder_path.resolve()),
            "quant_config_sha256": sha256(text_encoder_path / "quant_config.json"),
        },
        "sampling": {
            "steps": args.steps,
            "duration_seconds": args.duration,
            "max_rows_per_call": args.max_rows_per_call,
            "max_rows_per_layer": args.max_rows_per_layer,
            "selection": "evenly spaced rows from each real layer input",
            "adaln_selection": (
                "real video/audio scheduler timesteps at preregistered case indices, "
                "recorded separately from the full per-forward timestep table"
            ),
            "adaln_split_timesteps": adaln_split_timesteps,
        },
        "cases": [dict(case) for case in CASES],
        "split_contract": (
            "calibration and holdout prompts, seeds, and recorded AdaLN timestep inputs are disjoint"
        ),
    }
    if args.manifest_only:
        print(json.dumps(manifest, indent=2))
        return 0

    print(f"loading deployed text encoder: {text_encoder_path}", flush=True)
    encoder = MiniMaxH3TextEncoder(
        text_encoder_path,
        load_vision=False,
        verbose=True,
        tokenizer_dir=source / "tokenizer",
        processor_dir=source / "processor",
    )
    encoded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in CASES:
        prompt = str(case["prompt"])
        if prompt in encoded:
            continue
        hidden, tags = encoder.encode(prompt)
        encoded[prompt] = (
            np.array(hidden.astype(mx.float16), copy=True),
            np.array(tags, copy=True),
        )
        print(f"  encoded {prompt[:48]!r}: {len(tags)} tokens", flush=True)
    del encoder
    mx.clear_cache()

    print(f"loading bfloat16 teacher DiT: {transformer}", flush=True)
    teacher = load_dit(transformer, strict=True, verbose=True)
    bind_activation_paths(teacher)
    recorder = ActivationRecorder(args.max_rows_per_call, args.max_rows_per_layer)
    recorder.set_path_predicate(lambda path: not _is_adaln_projection(path))

    for case_index, case in enumerate(CASES):
        prompt = str(case["prompt"])
        embeds_host, tags = encoded[prompt]
        embeds = mx.array(embeds_host).astype(mx.bfloat16)
        layout, video_rows, audio_rows, video_sched, audio_sched = build_case(
            len(tags),
            int(case["height"]),
            int(case["width"]),
            args.duration,
            args.steps,
            teacher.config.latents_dim,
            teacher.config.audio_latents_dim,
            teacher.config.patch_size,
            int(case["seed"]),
        )
        table, plan = timestep_plan(layout, video_sched, audio_sched)
        selected_step = int(case["step_index"])
        recorder.set_split(str(case["split"]))
        captured = False
        for step_index, timestep in enumerate(video_sched.timesteps.tolist()):
            if step_index == selected_step:
                with recorder:
                    video_pred, audio_pred = forward(
                        teacher,
                        None,
                        layout,
                        video_rows,
                        audio_rows,
                        embeds,
                        table,
                        plan[step_index],
                    )
                captured = True
            else:
                video_pred, audio_pred = forward(
                    teacher,
                    None,
                    layout,
                    video_rows,
                    audio_rows,
                    embeds,
                    table,
                    plan[step_index],
                )
            mx.eval(video_pred, audio_pred)
            if step_index == selected_step:
                break
            video_rows = video_sched.step(
                video_pred[0].astype(mx.float32),
                float(timestep),
                video_rows,
            )
            audio_rows = audio_sched.step(
                audio_pred[0].astype(mx.float32),
                float(audio_sched.timesteps[step_index].item()),
                audio_rows,
            )
            mx.eval(video_rows, audio_rows)
        if not captured:
            raise RuntimeError(
                f"case {case_index} did not reach selected model evaluation {selected_step}"
            )
        print(
            f"case {case_index + 1}/{len(CASES)} {case['split']}: "
            f"seed={case['seed']} shape={case['width']}x{case['height']} "
            f"step={selected_step} rows={layout.sequence_length}",
            flush=True,
        )

    capture_disjoint_adaln_inputs(teacher, recorder, adaln_split_timesteps)
    dataset = recorder.dataset(manifest)
    required_paths = {
        path
        for path in dataset.paths("calibration")
        if path in dataset.paths("holdout")
    }
    if len(required_paths) < 250:
        raise RuntimeError(
            f"captured only {len(required_paths)} shared calibration/holdout layers; expected at least 250"
        )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output)
    print(
        f"wrote {output}: {len(dataset.paths('calibration'))} calibration layers, "
        f"{len(dataset.paths('holdout'))} holdout layers, sha256={sha256(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
