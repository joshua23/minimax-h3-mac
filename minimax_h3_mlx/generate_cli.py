"""Generate a MiniMax-H3 clip on Apple Silicon.

Release-facing examples::

    ./.venv/bin/python scripts/generate.py "a red fox leaps over a mossy log" \
        --checkpoint models/MiniMax-H3/FL2VA \
        --transformer models/MiniMax-H3-MLX-4bit \
        --text-encoder models/MiniMax-H3/FL2VA/text_encoder \
        --profile balanced --resolution 960x544 --output fox.mp4

Read the performance section of the README first: a step at the released 768-pixel canvas is ~8.8
minutes for a 5 s clip and ~1 hour for 15 s on an M3 Ultra, because MiniMax has not released its
sparse-attention implementation. `--height/--width` shrink the canvas for wiring checks, but H3 was
trained for a 768 short edge and anything else is off-distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

CHECKPOINT_ENV_VAR = "MINIMAX_H3_CHECKPOINT"
TRANSFORMER_ENV_VAR = "MINIMAX_H3_TRANSFORMER"
TEXT_ENCODER_ENV_VAR = "MINIMAX_H3_TEXT_ENCODER"
TURBO_LORA_ENV_VAR = "MINIMAX_H3_TURBO_LORA"
CANVAS_MULTIPLE = 32
DENSE_DEQUANT_PROFILE_CHOICES = (
    "off",
    "qkv-only-tiled",
    "ffn-fc2-tiled",
    "qkv-fc2-out-resident",
    "qkv-fc2-out-tiled",
)


@dataclass(frozen=True)
class ProfilePreset:
    """Machine-readable release CLI defaults for one invocation profile.

    These presets are invocation contracts only. They do not certify quality or performance; the
    release reports must still attach measured wall time, memory, and quality evidence for each
    profile before promoting any recommendation.
    """

    steps_sigma_points: int
    low_memory: bool
    stream_blocks: bool
    block_cache: bool
    quantization: str
    precision: str
    turbo: str
    cache_strategy: str
    description: str

    @property
    def denoiser_evaluations(self) -> int:
        return self.steps_sigma_points - 1

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["denoiser_evaluations"] = self.denoiser_evaluations
        payload["steps_semantics"] = "sigma grid points; denoiser evaluations = steps_sigma_points - 1"
        return payload


PROFILE_PRESETS: dict[str, ProfilePreset] = {
    "balanced": ProfilePreset(
        steps_sigma_points=5,
        low_memory=True,
        stream_blocks=True,
        block_cache=False,
        quantization="caller-supplied checkpoint/transformer/text-encoder paths; no bundled weights",
        precision="MLX BF16 activations with the precision encoded by the supplied model assets",
        turbo="uses --turbo-lora only when the caller supplies a Turbo LoRA path",
        cache_strategy="low-memory staged load with streamed DiT blocks; residual block cache off",
        description="default portable M4 24GB invocation target; measured release data still required",
    ),
    "quality": ProfilePreset(
        steps_sigma_points=16,
        low_memory=True,
        stream_blocks=True,
        block_cache=False,
        quantization="caller-supplied highest-available precision assets; no bundled weights",
        precision="MLX BF16 activations with the precision encoded by the supplied model assets",
        turbo="uses --turbo-lora only when the caller supplies a Turbo LoRA path",
        cache_strategy="low-memory staged load with streamed DiT blocks; approximate residual cache off",
        description="quality-oriented parser preset; not a measured quality claim by itself",
    ),
    "speed": ProfilePreset(
        steps_sigma_points=5,
        low_memory=True,
        stream_blocks=True,
        block_cache=True,
        quantization="caller-supplied checkpoint/transformer/text-encoder paths; no bundled weights",
        precision="MLX BF16 activations with the precision encoded by the supplied model assets",
        turbo="uses --turbo-lora only when the caller supplies a Turbo LoRA path",
        cache_strategy="low-memory staged load with streamed DiT blocks and residual block cache on",
        description="speed-oriented parser preset; approximate cache effects require measured validation",
    ),
}


def profile_contracts() -> dict[str, dict[str, object]]:
    """Return JSON-serializable profile contracts for release tooling/tests."""

    return {name: preset.to_dict() for name, preset in PROFILE_PRESETS.items()}


def generation_schedule_contract(sigma_points: int) -> dict[str, int | str]:
    """Document the `--steps` mapping without importing the heavy pipeline."""

    if sigma_points < 2:
        raise ValueError(f"--steps must request at least 2 sigma grid points, got {sigma_points}.")
    return {
        "cli_steps_argument": sigma_points,
        "sigma_points": sigma_points,
        "denoiser_evaluations": sigma_points - 1,
        "nfe": sigma_points - 1,
        "steps_semantics": "sigma grid points including terminal zero; pipeline forwards this as num_inference_steps",
    }


def _parse_resolution(value: str) -> tuple[int, int]:
    token = value.strip().lower().replace("×", "x")
    if "x" not in token:
        raise argparse.ArgumentTypeError("resolution must look like WIDTHxHEIGHT, e.g. 960x544")
    width_s, height_s = token.split("x", 1)
    try:
        width, height = int(width_s), int(height_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"resolution dimensions must be integers, got {value!r}") from exc
    _validate_canvas_dimension("width", width)
    _validate_canvas_dimension("height", height)
    return width, height


def _validate_canvas_dimension(name: str, value: int) -> None:
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive, got {value}")
    if value % CANVAS_MULTIPLE:
        raise argparse.ArgumentTypeError(
            f"{name} must be a multiple of {CANVAS_MULTIPLE}, got {value}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("-o", "--output", default="out.mp4")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PRESETS),
        default="balanced",
        help="release invocation preset; explicit low-level flags override preset defaults",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help=f"upstream release directory supplying VAEs/text encoder; required unless {CHECKPOINT_ENV_VAR} is set",
    )
    parser.add_argument(
        "-t",
        "--transformer",
        default=None,
        help=f"quantized transformer directory; may also be set with {TRANSFORMER_ENV_VAR}",
    )
    parser.add_argument("-d", "--duration", type=float, default=5.0, help="seconds, 5 to 15")
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=None,
        help="sigma grid points; denoiser forwards are steps - 1 (e.g. --steps 5 -> 4 NFE)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aspect", type=int, nargs=2, default=(16, 9))
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=None,
        metavar="WIDTHxHEIGHT",
        help="canvas override as WIDTHxHEIGHT; both dimensions must be positive multiples of 32",
    )
    parser.add_argument("--height", type=int, default=None, help="legacy canvas override, multiple of 32")
    parser.add_argument("--width", type=int, default=None, help="legacy canvas override, multiple of 32")
    parser.add_argument("--image", action="append", default=None, help="keyframe image (repeatable)")
    parser.add_argument(
        "--anchor",
        action="append",
        default=None,
        choices=["first", "last"],
        help="anchor for each --image, in order",
    )
    parser.add_argument(
        "--keep-adaln",
        action="store_true",
        help="keep the 13B adaln_proj resident instead of caching and dropping it",
    )
    parser.add_argument(
        "--stream-blocks",
        dest="stream_blocks",
        action="store_true",
        default=None,
        help="load one quantized DiT block at a time from safetensors",
    )
    parser.add_argument(
        "--no-stream-blocks",
        dest="stream_blocks",
        action="store_false",
        help="override the selected profile and keep DiT blocks resident when possible",
    )
    parser.add_argument(
        "--low-memory",
        dest="low_memory",
        action="store_true",
        default=None,
        help="stage text encoder, streamed DiT, video VAE and audio VAE",
    )
    parser.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="override the selected profile and use the legacy resident-load path",
    )
    parser.add_argument(
        "--text-encoder",
        default=None,
        help=f"H3 text encoder directory; --low-memory defaults to upstream full-precision weights and streams them layer-by-layer; may also be set with {TEXT_ENCODER_ENV_VAR}",
    )
    parser.add_argument(
        "--turbo-lora",
        default=None,
        help=f"MiniMax-H3 Turbo adapter file or native MLX adapter directory; may also be set with {TURBO_LORA_ENV_VAR}; use --steps 5 for 4 NFE",
    )
    parser.add_argument(
        "--turbo-lora-alpha",
        type=float,
        default=None,
        help="legacy PEFT training alpha; omit for native MLX adapters, which record alpha=rank",
    )
    parser.add_argument("--turbo-lora-scale", type=float, default=1.0, help="runtime multiplier for --turbo-lora")
    parser.add_argument(
        "--sigma-shift-video",
        "--video-shift",
        type=float,
        default=None,
        help="video scheduler shift override required by some Turbo adapters",
    )
    parser.add_argument(
        "--sigma-shift-audio",
        "--audio-shift",
        type=float,
        default=None,
        help="audio scheduler shift override required by some Turbo adapters",
    )
    parser.add_argument("--memory-limit-gb", type=float, default=16.0, help="MLX allocation guideline for --low-memory")
    parser.add_argument(
        "--memory-pressure-guard",
        action="store_true",
        help="experimental disabled-by-default guard: zero MLX cache, set memory/wired limits, and drain caches at stage boundaries",
    )
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable used for MP4 muxing; may be repo-local or an absolute path")
    parser.add_argument("--require-muxed-mp4", action="store_true", help="fail instead of writing frames+WAV fallback if MP4 muxing fails")
    parser.add_argument(
        "--block-cache",
        dest="block_cache",
        action="store_true",
        default=None,
        help="reuse trailing-block residuals on eligible denoising steps",
    )
    parser.add_argument(
        "--stream-block-group-size",
        type=int,
        default=1,
        help=(
            "opt-in streamed DiT residency group size; 1 preserves the default one-block-at-a-time loader"
        ),
    )
    parser.add_argument(
        "--no-block-cache",
        dest="block_cache",
        action="store_false",
        help="override the selected profile and disable residual block cache",
    )
    parser.add_argument("--block-cache-threshold", type=float, default=0.12, help="maximum adjacent sigma delta for block-cache reuse")
    parser.add_argument("--block-cache-depth", type=float, default=0.75, help="fraction of trailing transformer blocks served from cache")
    parser.add_argument("--block-cache-max-consecutive", type=int, default=2, help="maximum cached steps before a forced full refresh")
    parser.add_argument(
        "--cache-text-conditioning",
        action="store_true",
        help="experimental disabled-by-default candidate: precompute refined text rows once per run",
    )
    parser.add_argument(
        "--forward-profile-json",
        default=None,
        help=(
            "disabled-by-default profiling output path; when set, generation records synchronized "
            "load/DiT component/VAE/media timings as JSON"
        ),
    )
    parser.add_argument(
        "--dense-dequant-profile",
        choices=DENSE_DEQUANT_PROFILE_CHOICES,
        default="off",
        help=(
            "disabled-by-default dense-dequant profile: qkv-only-tiled enables only transient "
            "tiled attention qkv_proj; ffn-fc2-tiled enables only transient tiled mlp.fc2 "
            "(tile size defaults to 1024); qkv-fc2-out-* are rejected-for-default provenance "
            "profiles that also choose one attention out_proj path"
        ),
    )
    parser.add_argument(
        "--attention-qkv-tile-size",
        type=int,
        default=2048,
        help="output rows per transient dense-dequant tile for attention qkv_proj when --dense-dequant-profile is enabled",
    )
    parser.add_argument(
        "--ffn-fc2-tile-size",
        type=int,
        default=1024,
        help="output rows per transient dense-dequant tile for mlp.fc2 when --dense-dequant-profile is enabled",
    )
    parser.add_argument(
        "--attention-out-tile-size",
        type=int,
        default=2048,
        help="output rows per transient dense-dequant tile for attention out_proj in the tiled-out profile",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None, *, env: Mapping[str, str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    env_map = os.environ if env is None else env

    if args.checkpoint is None:
        args.checkpoint = env_map.get(CHECKPOINT_ENV_VAR)
    if args.transformer is None:
        args.transformer = env_map.get(TRANSFORMER_ENV_VAR)
    if args.text_encoder is None:
        args.text_encoder = env_map.get(TEXT_ENCODER_ENV_VAR)
    if args.turbo_lora is None:
        args.turbo_lora = env_map.get(TURBO_LORA_ENV_VAR)

    preset = PROFILE_PRESETS[args.profile]
    if args.steps is None:
        args.steps = preset.steps_sigma_points
    if args.low_memory is None:
        args.low_memory = preset.low_memory
    if args.stream_blocks is None:
        args.stream_blocks = preset.stream_blocks
    if args.block_cache is None:
        args.block_cache = preset.block_cache

    if args.checkpoint is None or not str(args.checkpoint).strip():
        parser.error(f"--checkpoint is required (or set {CHECKPOINT_ENV_VAR}); no local model path is assumed")
    try:
        generation_schedule_contract(int(args.steps))
    except ValueError as exc:
        parser.error(str(exc))
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.aspect[0] <= 0 or args.aspect[1] <= 0:
        parser.error("--aspect values must be positive")
    if args.resolution is not None:
        if args.width is not None or args.height is not None:
            parser.error("--resolution cannot be combined with --width or --height; use one canvas override surface")
        args.width, args.height = args.resolution
    if args.image and len(args.anchor or ()) != len(args.image):
        parser.error(f"--anchor must be given once per --image ({len(args.image)} images, {len(args.anchor or ())} anchors)")
    for name in ("width", "height"):
        value = getattr(args, name)
        if value is not None:
            try:
                _validate_canvas_dimension(name, int(value))
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
    if args.memory_limit_gb <= 0:
        parser.error("--memory-limit-gb must be positive")
    for name in ("sigma_shift_video", "sigma_shift_audio"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.block_cache_threshold < 0:
        parser.error("--block-cache-threshold must be non-negative")
    if not 0.0 <= args.block_cache_depth < 1.0:
        parser.error("--block-cache-depth must satisfy 0 <= depth < 1")
    if args.block_cache_max_consecutive < 0:
        parser.error("--block-cache-max-consecutive must be non-negative")
    if args.stream_block_group_size <= 0:
        parser.error("--stream-block-group-size must be positive")
    for name in ("attention_qkv_tile_size", "ffn_fc2_tile_size", "attention_out_tile_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    from minimax_h3_mlx.block_cache import BlockCacheConfig
    from minimax_h3_mlx.forward_profile import ForwardPassProfiler, active_profiler, profiled_block
    from minimax_h3_mlx.media import save_frames, save_mp4, save_wav
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    images = None
    if args.image:
        from PIL import Image, ImageOps

        images = [ImageOps.exif_transpose(Image.open(p).convert("RGB")) for p in args.image]
    anchors = tuple(args.anchor or ())

    profiler = ForwardPassProfiler() if args.forward_profile_json else None
    profile_write_error: str | None = None
    with active_profiler(profiler):
        pipe = MiniMaxH3Pipeline.from_pretrained(
            args.checkpoint,
            transformer_dir=args.transformer,
            load_vision=bool(images),
            stream_blocks=args.stream_blocks or args.low_memory,
            low_memory=args.low_memory,
            text_encoder_dir=args.text_encoder,
            turbo_lora_path=args.turbo_lora,
            turbo_lora_alpha=args.turbo_lora_alpha,
            turbo_lora_scale=args.turbo_lora_scale,
            sigma_shift_video=args.sigma_shift_video,
            sigma_shift_audio=args.sigma_shift_audio,
            memory_limit_gb=args.memory_limit_gb,
            stream_block_group_size=args.stream_block_group_size,
            dense_dequant_profile=args.dense_dequant_profile,
            dense_dequant_attention_qkv_tile_size=args.attention_qkv_tile_size,
            dense_dequant_ffn_fc2_tile_size=args.ffn_fc2_tile_size,
            dense_dequant_attention_out_tile_size=args.attention_out_tile_size,
            memory_pressure_guard=args.memory_pressure_guard,
        )
        result = pipe(
            args.prompt,
            duration_seconds=args.duration,
            aspect=tuple(args.aspect),
            num_inference_steps=args.steps,
            seed=args.seed,
            images=images,
            keyframe_anchors=anchors,
            height=args.height,
            width=args.width,
            drop_adaln=not args.keep_adaln,
            block_cache_config=(
                BlockCacheConfig(
                    sigma_threshold=args.block_cache_threshold,
                    max_consecutive=args.block_cache_max_consecutive,
                    cache_depth=args.block_cache_depth,
                )
                if args.block_cache
                else None
            ),
            cache_text_conditioning=args.cache_text_conditioning,
        )

        output = Path(args.output)
        try:
            with profiled_block("media.save_mp4_mux", "media_mux", synchronize=False, metadata={"output": str(output)}):
                save_mp4(output, result.video, result.fps, result.audio, result.sample_rate, ffmpeg=args.ffmpeg)
            print(f"\nwrote {output} ({result.video.shape[0]} frames, "
                  f"{result.audio.shape[-1] / result.sample_rate:.2f}s audio)")
        except RuntimeError as exc:
            if args.require_muxed_mp4 or args.ffmpeg:
                print(f"\nMP4 muxing failed ({exc}); not writing frames+WAV fallback", file=sys.stderr)
                return 70
            print(f"\nffmpeg unavailable ({exc}); writing frames and wav instead")
            with profiled_block("media.save_frames_wav_fallback", "media_mux", synchronize=False, metadata={"output": str(output)}):
                save_frames(output.with_suffix(""), result.video)
                save_wav(output.with_suffix(".wav"), result.audio, result.sample_rate)

    if profiler is not None:
        profile_path = Path(args.forward_profile_json)
        try:
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(json.dumps(profiler.to_dict(include_events=True), indent=2, sort_keys=True) + "\n")
            print(f"wrote forward profile {profile_path}")
        except OSError as exc:
            profile_write_error = f"{type(exc).__name__}: {exc}"
            print(f"failed to write forward profile {profile_path}: {profile_write_error}", file=sys.stderr)

    print(f"{result.seconds_per_step:.1f}s per step, {result.total_seconds / 60:.1f} min total")
    if result.block_cache_stats is not None:
        stats = result.block_cache_stats
        print(
            "block cache: "
            f"full={stats['full_steps']} cached={stats['cache_steps']} "
            f"skipped={stats['skipped_blocks']} blocks "
            f"({100.0 * stats['saved_fraction']:.1f}% of block executions)"
        )
    if profile_write_error is not None:
        return 74
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
