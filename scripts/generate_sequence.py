#!/usr/bin/env python3
"""Generate a multi-shot sequence with motion context chaining.

Shot 1 is a plain generation (text-to-video, or image-to-video with ``--first-image``); every
following shot pins the previous clip's tail frames and tail sound as conditioning, so motion and
sound continue across the cut instead of restarting. Each shot's pinned head is trimmed off its
delivery, and the trimmed shots are concatenated into one video.

Example::

    ./.venv/bin/python scripts/generate_sequence.py \\
        "A red panda waves from a tiny stage" \\
        "The red panda leaps off the stage toward the forest" \\
        "The red panda runs through a misty bamboo forest" \\
        --checkpoint models/MiniMax-H3/FL2VA \\
        --transformer models/MiniMax-H3/FL2VA/transformer \\
        --turbo-lora models/Minimax-h3-Turbo-v1.0-4step-768p-bf16/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \\
        --sigma-shift-video 6 --sigma-shift-audio 3 --steps 5 \\
        --low-memory --stream-blocks --no-block-cache --memory-limit-gb 24 \\
        --resolution 512x288 --duration 5 --output-dir out/sequence
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.motion_context import ClipLatents, CONTEXT_WINDOWS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts", nargs="+", help="one prompt per shot, in order (at least 2 for a chain)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--text-encoder", default=None)
    parser.add_argument("--first-image", default=None, help="optional first-frame image for shot 1")
    parser.add_argument("--chain-from", default=None, metavar="NPZ",
                        help="start the chain from a previous clip's saved latents: the prompts then "
                             "continue that clip, and no shot-1 regeneration happens")
    parser.add_argument("--turbo-lora", default=None)
    parser.add_argument("--turbo-lora-scale", type=float, default=1.0)
    parser.add_argument("--sigma-shift-video", type=float, default=None)
    parser.add_argument("--sigma-shift-audio", type=float, default=None)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", default="512x288", help="WIDTHxHEIGHT, multiples of 32")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--context-frames", type=int, default=22, choices=list(CONTEXT_WINDOWS),
                        help="previous clip frames pinned as motion context")
    parser.add_argument("--context-audio-frames", type=int, default=24,
                        help="previous clip tail sound pinned across the join (0 follows the video window)")
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--stream-block-group-size", type=int, default=2)
    parser.add_argument("--output-dir", default="out/sequence")
    parser.add_argument("--save-latents", action="store_true", help="save each shot's latents as .npz")
    parser.add_argument("--no-concat", action="store_true", help="skip the final concat")
    parser.add_argument("--ffmpeg", default=None)
    return parser


def main() -> int:
    from minimax_h3_mlx.media import save_mp4
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    args = build_parser().parse_args()
    if len(args.prompts) < 2 and not args.chain_from:
        print("A sequence needs at least two shots (or --chain-from to continue one).", file=sys.stderr)
        return 2
    width, height = (int(part) for part in args.resolution.lower().split("x"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_image = None
    if args.first_image:
        from PIL import Image, ImageOps

        first_image = [ImageOps.exif_transpose(Image.open(args.first_image).convert("RGB"))]

    pipe = MiniMaxH3Pipeline.from_pretrained(
        args.checkpoint,
        transformer_dir=args.transformer,
        text_encoder_dir=args.text_encoder,
        load_vision=first_image is not None,
        stream_blocks=True,
        low_memory=True,
        turbo_lora_path=args.turbo_lora,
        turbo_lora_scale=args.turbo_lora_scale,
        sigma_shift_video=args.sigma_shift_video,
        sigma_shift_audio=args.sigma_shift_audio,
        memory_limit_gb=args.memory_limit_gb,
        stream_block_group_size=args.stream_block_group_size,
        memory_pressure_guard=False,
        verbose=False,
    )

    context: ClipLatents | None = None
    if args.chain_from:
        context = ClipLatents.load(args.chain_from)
        print(f"chaining from {args.chain_from} ({context.latent_frames} latent steps, "
              f"{context.frames} frames)", flush=True)
    shot_paths: list[Path] = []
    for index, prompt in enumerate(args.prompts, start=1):
        shot_number = index + (1 if args.chain_from else 0)
        result = pipe(
            prompt,
            duration_seconds=args.duration,
            num_inference_steps=args.steps,
            seed=args.seed + index - 1,
            images=first_image if index == 1 and context is None else None,
            context=context,
            context_video_frames=args.context_frames,
            context_audio_frames=args.context_audio_frames,
            return_latents=True,
            height=height,
            width=width,
        )
        context = ClipLatents(video=result.video_latents, audio=result.audio_latents)
        if args.save_latents:
            context.save(output_dir / f"shot-{shot_number:02d}-latents.npz")
        shot_path = output_dir / f"shot-{shot_number:02d}.mp4"
        save_mp4(shot_path, result.video, result.fps, result.audio, result.sample_rate, ffmpeg=args.ffmpeg)
        shot_paths.append(shot_path)
        delivered = result.video.shape[0]
        print(f"shot {shot_number}: {delivered} frames, "
              f"{result.audio.shape[-1] / result.sample_rate:.2f}s audio -> {shot_path}", flush=True)

    if args.no_concat:
        return 0

    concat_path = output_dir / "sequence.mp4"
    list_path = output_dir / "concat.txt"
    with list_path.open("w") as handle:
        for shot_path in shot_paths:
            handle.write(f"file '{shot_path.resolve()}'\n")
    ffmpeg = args.ffmpeg or "ffmpeg"
    process = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(concat_path)],
        capture_output=True,
    )
    if process.returncode != 0:
        print(f"concat failed: {process.stderr.decode()[:400]}", file=sys.stderr)
        return 1
    print(f"sequence: {concat_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
