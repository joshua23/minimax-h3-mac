"""The MiniMax-H3 text/keyframe -> video+audio pipeline in MLX.

One packed sequence carries text, keyframe conditioning, audio and video rows at once, and a single
transformer forward per step predicts the velocity of every row — video and audio are denoised
*jointly*, on two schedules with different sigma shifts (12.0 and 3.0). The checkpoint is
CFG-distilled, so there is no unconditional pass and no guidance scale.

Conditioning rows are re-imposed by construction rather than by masking: only the generated rows are
ever written back, so keyframe anchors survive the whole loop untouched.

The AdaLN modulation cache is built once over the union of every timestep the run will present, and
the 13B of `adaln_proj` is then dropped — see :mod:`minimax_h3_mlx.adaln`.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .adaln import ModulationCache, drop_adaln_weights
from .block_cache import BlockCacheConfig, BlockResidualCache
from .config import DiTConfig, PipelineConfig
from .dit import (
    DENSE_DEQUANT_PROFILE_OFF,
    apply_dense_dequant_profile_to_block,
    normalize_dense_dequant_profile,
)
from .forward_profile import profiled_call
from .packing import (
    AUDIO_CHANNELS,
    FPS,
    KEYFRAME_ENCODE_SEED,
    KEYFRAME_NOISE_AUG,
    PIXEL_MEAN,
    PIXEL_STD,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .scheduler import MiniMaxH3Scheduler


def _call_mlx_memory_control(name: str, *args) -> bool:
    """Call an optional MLX allocator/memory-control hook if this MLX build exposes it."""

    for owner in (mx, getattr(mx, "metal", None)):
        if owner is None:
            continue
        func = getattr(owner, name, None)
        if func is None:
            continue
        try:
            func(*args)
            return True
        except Exception:
            continue
    return False


def _drain_mlx_cache() -> None:
    """Synchronize, collect Python references, and return cached Metal buffers to MLX/Metal."""

    gc.collect()
    mx.synchronize()
    _call_mlx_memory_control("clear_cache")


def _apply_mlx_pressure_limits(memory_limit_gb: float) -> None:
    """Apply conservative MLX allocator limits without changing model math."""

    limit_bytes = int(float(memory_limit_gb) * 1e9)
    _call_mlx_memory_control("set_cache_limit", 0)
    _call_mlx_memory_control("set_memory_limit", limit_bytes)
    _call_mlx_memory_control("set_wired_limit", limit_bytes)


@dataclass
class GenerationResult:
    video: np.ndarray  # (frames, height, width, 3) uint8
    audio: np.ndarray  # (2, samples) float32, in [-1, 1]
    sample_rate: int
    fps: int = FPS
    seconds_per_step: float = 0.0
    total_seconds: float = 0.0
    block_cache_stats: dict[str, int | float] | None = None


def detach_bfloat16(array: mx.array) -> mx.array:
    """Materialize a small BF16 result independently of its phase-owned model graph."""
    host = np.array(array.astype(mx.float16), copy=True)
    detached = mx.array(host).astype(mx.bfloat16)
    mx.eval(detached)
    return detached


class MiniMaxH3Pipeline:
    """Joint video + audio generation."""

    def __init__(
        self,
        dit,
        text_encoder,
        video_vae,
        audio_vae,
        config: PipelineConfig | None = None,
    ):
        self.dit = dit
        self.text_encoder = text_encoder
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.config = config or PipelineConfig()
        self._cache: ModulationCache | None = None
        self._cache_timesteps: tuple[float, ...] | None = None
        self._block_provider = None
        self._low_memory = False
        self._checkpoint_root: Path | None = None
        self._dit_path: Path | None = None
        self._text_encoder_path: Path | None = None
        self._turbo_lora_path: Path | None = None
        self._turbo_lora_alpha: float | None = None
        self._turbo_lora_scale = 1.0
        self._block_load_mode = "mlx"
        self._stream_block_group_size = 1
        self._dense_dequant_profile = DENSE_DEQUANT_PROFILE_OFF
        self._dense_dequant_attention_qkv_tile_size = 2048
        self._dense_dequant_ffn_fc2_tile_size = 1024
        self._dense_dequant_attention_out_tile_size = 2048
        self._dit_config = getattr(dit, "config", None)
        self._video_config = getattr(video_vae, "config", None)
        self._audio_config = getattr(audio_vae, "config", None)
        self._memory_pressure_guard = False
        self._memory_limit_gb = 16.0

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        transformer_dir: str | Path | None = None,
        dtype: mx.Dtype = mx.bfloat16,
        load_vision: bool = False,
        stream_blocks: bool = False,
        low_memory: bool = False,
        text_encoder_dir: str | Path | None = None,
        turbo_lora_path: str | Path | None = None,
        turbo_lora_alpha: float | None = None,
        turbo_lora_scale: float = 1.0,
        sigma_shift_video: float | None = None,
        sigma_shift_audio: float | None = None,
        memory_limit_gb: float = 16.0,
        block_load_mode: str = "mlx",
        stream_block_group_size: int = 1,
        dense_dequant_profile: str | None = None,
        dense_dequant_attention_qkv_tile_size: int = 2048,
        dense_dequant_ffn_fc2_tile_size: int = 1024,
        dense_dequant_attention_out_tile_size: int = 2048,
        memory_pressure_guard: bool = False,
        verbose: bool = True,
    ) -> "MiniMaxH3Pipeline":
        """Load a released ``FL2VA/`` (or ``Ref2VA/``) directory.

        Args:
            checkpoint_dir: the upstream release, which supplies the VAEs and the text encoder.
            transformer_dir: load the DiT from here instead of ``<checkpoint_dir>/transformer``.
                This is how a published quant is used: the quantized repository holds only the
                transformer, and everything else still comes from upstream. ``load_dit`` picks up
                the recorded recipe from its ``quant_config.json`` automatically.
        """
        from .load import (
            load_audio_vae,
            load_dit,
            load_video_vae,
            read_audio_vae_config,
            read_video_vae_config,
        )
        from .streaming import BLOCK_LOAD_MODES, load_streaming_dit
        from .text_encoder import MiniMaxH3TextEncoder

        root = Path(checkpoint_dir)
        dit_path = Path(transformer_dir) if transformer_dir else root / "transformer"
        if block_load_mode not in BLOCK_LOAD_MODES:
            allowed = ", ".join(BLOCK_LOAD_MODES)
            raise ValueError(f"unknown block load mode {block_load_mode!r}; expected one of: {allowed}")
        stream_block_group_size = int(stream_block_group_size)
        if stream_block_group_size <= 0:
            raise ValueError(f"stream_block_group_size must be positive, got {stream_block_group_size}")
        config = PipelineConfig.from_model_index(root / "model_index.json").with_sigma_shift_overrides(
            video=sigma_shift_video,
            audio=sigma_shift_audio,
        )

        if memory_pressure_guard:
            _apply_mlx_pressure_limits(memory_limit_gb)
            _drain_mlx_cache()

        if low_memory:
            mx.set_cache_limit(0)
            mx.set_memory_limit(int(memory_limit_gb * 1e9))
            if memory_pressure_guard:
                _apply_mlx_pressure_limits(memory_limit_gb)
            # Keep conditioning quality at the upstream precision by default. The encoder is
            # instantiated only for the conditioning phase and streams one required decoder layer
            # at a time, so the original BF16 checkpoint does not become resident as a whole.
            text_path = Path(text_encoder_dir) if text_encoder_dir else root / "text_encoder"
            if not (text_path / "config.json").is_file():
                raise FileNotFoundError(f"text encoder checkpoint not found at {text_path}")
            if not (text_path / "model.safetensors.index.json").is_file():
                raise FileNotFoundError(f"indexed text encoder weights not found at {text_path}")
            if not (dit_path / "config.json").is_file():
                raise FileNotFoundError(f"transformer config not found at {dit_path}")
            if not (dit_path / "model.safetensors.index.json").is_file():
                raise FileNotFoundError(f"indexed transformer weights not found at {dit_path}")
            pipeline = cls(None, None, None, None, config)
            pipeline._memory_pressure_guard = bool(memory_pressure_guard)
            pipeline._memory_limit_gb = float(memory_limit_gb)
            pipeline._low_memory = True
            pipeline._checkpoint_root = root
            pipeline._dit_path = dit_path
            pipeline._text_encoder_path = text_path
            pipeline._turbo_lora_path = Path(turbo_lora_path) if turbo_lora_path else None
            pipeline._turbo_lora_alpha = turbo_lora_alpha
            pipeline._turbo_lora_scale = turbo_lora_scale
            pipeline._block_load_mode = block_load_mode
            pipeline._stream_block_group_size = stream_block_group_size
            pipeline.set_dense_dequant_profile(
                dense_dequant_profile,
                attention_qkv_tile_size=dense_dequant_attention_qkv_tile_size,
                ffn_fc2_tile_size=dense_dequant_ffn_fc2_tile_size,
                attention_out_tile_size=dense_dequant_attention_out_tile_size,
            )
            pipeline._dit_config = DiTConfig.from_json(dit_path / "config.json")
            pipeline._video_config = read_video_vae_config(root / "video_vae")
            pipeline._audio_config = read_audio_vae_config(root / "audio_vae")
            return pipeline

        def step(label, fn):
            started = time.perf_counter()
            out = profiled_call(f"load.{label}", "load_overhead", fn, eval_output=False)
            if verbose:
                print(f"  {label}: {time.perf_counter() - started:.1f}s")
            return out

        if verbose:
            print(f"loading MiniMax-H3 from {root}")
        text_path = Path(text_encoder_dir) if text_encoder_dir else root / "text_encoder"
        text_encoder = step(
            "text encoder",
            lambda: MiniMaxH3TextEncoder(
                text_path,
                dtype=dtype,
                load_vision=load_vision,
                tokenizer_dir=root / "tokenizer",
                processor_dir=root / "processor",
            ),
        )
        if stream_blocks or turbo_lora_path is not None:
            dit, block_provider = step(
                f"streaming transformer ({dit_path.name})",
                lambda: load_streaming_dit(
                    dit_path,
                    turbo_lora_path=turbo_lora_path,
                    turbo_lora_alpha=turbo_lora_alpha,
                    turbo_lora_scale=turbo_lora_scale,
                    block_load_mode=block_load_mode,
                    stream_block_group_size=stream_block_group_size,
                    verbose=verbose,
                ),
            )
        else:
            dit = step(f"transformer ({dit_path.name})", lambda: load_dit(dit_path))
            block_provider = None
        video_vae = step("video vae", lambda: load_video_vae(root / "video_vae"))
        audio_vae = step("audio vae", lambda: load_audio_vae(root / "audio_vae"))
        pipeline = cls(dit, text_encoder, video_vae, audio_vae, config)
        pipeline._memory_pressure_guard = bool(memory_pressure_guard)
        pipeline._memory_limit_gb = float(memory_limit_gb)
        pipeline._block_provider = block_provider
        pipeline._stream_block_group_size = stream_block_group_size
        pipeline.set_dense_dequant_profile(
            dense_dequant_profile,
            attention_qkv_tile_size=dense_dequant_attention_qkv_tile_size,
            ffn_fc2_tile_size=dense_dequant_ffn_fc2_tile_size,
            attention_out_tile_size=dense_dequant_attention_out_tile_size,
        )
        return pipeline

    def set_dense_dequant_profile(
        self,
        profile: str | None,
        *,
        attention_qkv_tile_size: int = 2048,
        ffn_fc2_tile_size: int = 1024,
        attention_out_tile_size: int = 2048,
    ) -> None:
        """Configure an opt-in/provenance dense-dequant profile for main DiT blocks."""

        selected = normalize_dense_dequant_profile(profile)
        self._dense_dequant_profile = selected
        self._dense_dequant_attention_qkv_tile_size = int(attention_qkv_tile_size)
        self._dense_dequant_ffn_fc2_tile_size = int(ffn_fc2_tile_size)
        self._dense_dequant_attention_out_tile_size = int(attention_out_tile_size)
        if self._block_provider is not None:
            set_profile = getattr(self._block_provider, "set_dense_dequant_profile", None)
            if set_profile is not None:
                set_profile(
                    selected,
                    attention_qkv_tile_size=self._dense_dequant_attention_qkv_tile_size,
                    ffn_fc2_tile_size=self._dense_dequant_ffn_fc2_tile_size,
                    attention_out_tile_size=self._dense_dequant_attention_out_tile_size,
                )
        if self.dit is not None and getattr(self.dit, "blocks", None):
            for block in self.dit.blocks:
                apply_dense_dequant_profile_to_block(
                    block,
                    selected,
                    attention_qkv_tile_size=self._dense_dequant_attention_qkv_tile_size,
                    ffn_fc2_tile_size=self._dense_dequant_ffn_fc2_tile_size,
                    attention_out_tile_size=self._dense_dequant_attention_out_tile_size,
                )

    def _memory_guard_boundary(self) -> None:
        """Reassert optional allocator limits and drain reusable MLX/Metal buffers."""
        if self._memory_pressure_guard:
            _apply_mlx_pressure_limits(self._memory_limit_gb)
        _drain_mlx_cache()

    def _release_component(self, name: str) -> None:
        """Drop one phase-owned component and return cached Metal buffers."""
        setattr(self, name, None)
        self._memory_guard_boundary()

    # -- schedule -----------------------------------------------------------------------------

    def _build_schedules(self, num_inference_steps: int):
        video = MiniMaxH3Scheduler(shift=self.config.sigma_shift_video)
        audio = MiniMaxH3Scheduler(shift=self.config.sigma_shift_audio)
        video.set_timesteps(num_inference_steps)
        audio.set_timesteps(num_inference_steps)
        return video, audio

    def _row_timestep_plan(self, layout, video_timesteps, audio_timesteps):
        """Per-step ``(timestep_indices,)`` against one global timestep table.

        The transformer is handed the same table at every step, so a single
        :class:`ModulationCache` covers the whole run. Conditioning video rows sit at
        ``max(t, 0.999)`` and reference audio rows at ``1.0``, matching the reference.
        """
        per_step = []
        for t, at in zip(video_timesteps.tolist(), audio_timesteps.tolist()):
            distinct, inverse = build_row_timesteps(
                layout, float(t), float(at), max(float(t), KEYFRAME_NOISE_AUG), 1.0
            )
            per_step.append((np.array(distinct), np.array(inverse)))

        table = sorted({float(v) for distinct, _ in per_step for v in distinct})
        lookup = {v: i for i, v in enumerate(table)}
        plan = []
        for distinct, inverse in per_step:
            remap = np.array([lookup[float(v)] for v in distinct], dtype=np.int32)
            plan.append(mx.array(remap[inverse].astype(np.int32)))
        return mx.array(np.array(table, dtype=np.float32)), plan

    def _ensure_cache(self, timesteps: mx.array, drop_adaln: bool, verbose: bool):
        key = tuple(round(float(v), 9) for v in timesteps.tolist())
        if self._cache is not None and self._cache_timesteps == key:
            return
        started = time.perf_counter()
        if self._block_provider is None:
            self._cache = ModulationCache.build(self.dit, timesteps, dtype=mx.bfloat16)
        else:
            self._cache = ModulationCache.build_streaming(
                self.dit,
                self._block_provider,
                timesteps,
                dtype=mx.bfloat16,
            )
        self._cache_timesteps = key
        if verbose:
            print(f"  adaln cache: {len(key)} timesteps, {self._cache.nbytes() / 1e6:.0f} MB "
                  f"in {time.perf_counter() - started:.1f}s")
        if drop_adaln and self._block_provider is None:
            freed = drop_adaln_weights(self.dit)
            mx.eval(self.dit.parameters())
            if verbose:
                print(f"  dropped adaln projections, freeing {freed / 1e9:.1f} GB")

    # -- keyframe conditioning ----------------------------------------------------------------

    def _encode_keyframes(self, images: list, height: int, width: int) -> mx.array:
        """Encode ``fl2va`` keyframes into packed conditioning rows.

        Keyframes are single frames, so they go through the video VAE's **spatial** encoder only —
        none of its 17-frame temporal chunking applies. Two details of the reference are load-bearing
        and easy to miss:

        * the posterior is **sampled**, not taken at its mode, under a generator seeded with 42
          independently of the request seed;
        * the sampled latent is **rounded through float16** before normalization, which is about 11
          bits of every conditioning latent — the released model's conditioning cannot be reproduced
          without it.

        MLX's RNG differs from torch's, so the seed-42 draw is not bit-identical to the reference's;
        the distribution and every other step are.
        """
        from .packing import KEYFRAME_ENCODE_SEED, prepare_keyframe_image

        cfg = self._video_config
        latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)

        mx.random.seed(KEYFRAME_ENCODE_SEED)
        rows = []
        for index, image in enumerate(images):
            prepared = prepare_keyframe_image(image, height, width, stretch=index == 0)
            pixels = np.asarray(prepared, dtype=np.float32).transpose(2, 0, 1)[None, :, None]
            pixels = (pixels / 255.0 - pixel_mean) / pixel_std

            # (1, 3, 1, H, W) -> channels-last for the spatial encoder.
            moments = self.video_vae._encode_clip(mx.array(pixels).transpose(0, 2, 3, 4, 1))
            channels = cfg.latent_channels
            mean, logvar = moments[..., :channels], moments[..., channels:]
            logvar = mx.clip(logvar, -30.0, 20.0)
            std = mx.exp(0.5 * logvar)
            latent = mean + std * mx.random.normal(mean.shape)
            # -> (1, C, 1, H', W'), then the float16 round trip the reference relies on.
            latent = latent.transpose(0, 4, 1, 2, 3).astype(mx.float16).astype(mx.float32)
            normalized = (latent - latents_mean) / latents_std
            rows.append(patchify_video_latents(normalized, self._dit_config.patch_size))
        return mx.concatenate(rows)

    # -- generation ---------------------------------------------------------------------------

    # -- ref2va conditioning ------------------------------------------------------------------

    def _encode_ref2va_text(self, prompt: str, references: list):
        """Build MiniMax-H3's ref2va presentation and encode it through the conditioner."""
        from .config import TAG_TEXT, TAG_VIDEO
        from .ref2va import build_ref2va_presentation, preprocess_reference_video, sample_reference_video_frames

        encoder = self.text_encoder
        tokenizer = encoder.tokenizer
        merge = encoder.processor.image_processor.merge_size**2

        image_token_counts: list[int] = []
        image_pixels = image_grids = None
        images = [reference.image for reference in references if reference.kind == "image"]
        if images:
            vision = encoder.processor.image_processor(images=images, return_tensors="np")
            image_pixels = np.asarray(vision["pixel_values"])
            image_grids = np.asarray(vision["image_grid_thw"])
            image_token_counts = [int(grid.prod()) // merge for grid in image_grids]

        video_block_token_counts: list[int] = []
        video_pixels_list, video_grids_list = [], []
        for reference in (r for r in references if r.kind == "video"):
            sampled, block_timestamps = sample_reference_video_frames(reference.frames)
            reference.block_timestamps = block_timestamps
            pixels, grid = preprocess_reference_video(np.stack(sampled))
            if int(grid[0][0]) != len(block_timestamps):
                raise ValueError(
                    f"The video reference merged into {int(grid[0][0])} vision blocks, but "
                    f"{len(block_timestamps)} of them were labelled."
                )
            video_pixels_list.append(pixels)
            video_grids_list.append(grid)
            video_block_token_counts.append(int(grid[0][1]) * int(grid[0][2]) // merge)
        video_pixels = np.concatenate(video_pixels_list) if video_pixels_list else None
        video_grids = np.concatenate(video_grids_list) if video_grids_list else None

        token_ids, token_tags = build_ref2va_presentation(
            tokenizer,
            prompt,
            references,
            image_token_counts,
            video_block_token_counts,
            TAG_TEXT,
            TAG_VIDEO,
            encoder.vision_start_token_id,
            encoder.vision_end_token_id,
            tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        )
        input_ids = mx.array(np.array([token_ids], dtype=np.int32))
        return encoder.encode_presentation(
            input_ids,
            np.array(token_tags, dtype=np.int64),
            image_pixels,
            image_grids,
            video_pixels,
            video_grids,
        )

    def _encode_reference_video_frames(self, frames: np.ndarray):
        """Normalize and encode one temporal chunk at a time, keeping host pixels bounded."""
        cfg = self._video_config
        mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 1, 1, 3)
        std = np.array(PIXEL_STD, np.float32).reshape(1, 1, 1, 3)
        chunks = []
        for start in range(0, len(frames), cfg.clip_length):
            pixels = frames[start:start + cfg.clip_length].astype(np.float32)
            if len(pixels) < cfg.clip_length:
                pixels = np.concatenate([pixels, np.repeat(pixels[-1:], cfg.clip_length - len(pixels), axis=0)])
            pixels /= 255.0
            pixels -= mean
            pixels /= std
            x = mx.array(pixels[None])
            encoded = self.video_vae._encode_clip(x)
            mx.eval(encoded)
            chunks.append(encoded)
            del pixels, x
        moments = mx.concatenate(chunks, axis=1)
        if cfg.token_drop > 0:
            moments = moments[:, :-cfg.token_drop]
        return moments.transpose(0, 4, 1, 2, 3)

    def _encode_reference_media(self, references: list):
        """Encode the references through the VAEs and resolve their latent geometry.

        Image and video references take the recipe the fl2va keyframes use — posterior **sampled**
        under a fresh seed-42 draw per reference, the sample rounded through float16 before
        normalization — except a video goes through the 17-frames-per-chunk temporal encoding.
        Reference soundtracks take the posterior **mean** and are never sampled.
        """
        from .ref2va import trim_reference_num_frames

        cfg = self._video_config
        latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
        audio_cfg = self._audio_config
        audio_mean = mx.array(np.array(audio_cfg.latents_mean, np.float32)).reshape(1, -1, 1)
        audio_std = mx.array(np.array(audio_cfg.latents_std, np.float32)).reshape(1, -1, 1)

        video_rows, audio_rows = [], []
        for reference in references:
            if reference.kind != "audio":
                if reference.kind == "image":
                    pixels = np.asarray(reference.image, dtype=np.float32).transpose(2, 0, 1)[None, :, None]
                channels = cfg.latent_channels
                if reference.kind == "image":
                    pixels = (pixels / 255.0 - pixel_mean) / pixel_std
                    x = mx.array(pixels)
                    # _encode_clip returns channels-last moments (B, F, H, W, 2C).
                    moments = self.video_vae._encode_clip(x.transpose(0, 2, 3, 4, 1))
                    mean, logvar = moments[..., :channels], moments[..., channels:]
                    mx.random.seed(KEYFRAME_ENCODE_SEED)
                    latent = mean + mx.exp(0.5 * mx.clip(logvar, -30.0, 20.0)) * mx.random.normal(mean.shape)
                    latent = latent.transpose(0, 4, 1, 2, 3)
                else:
                    # encode() returns channels-first moments (B, 2C, F, H, W).
                    frames = reference.frames[: trim_reference_num_frames(reference.frames.shape[0])]
                    moments = self._encode_reference_video_frames(frames)
                    mean, logvar = moments[:, :channels], moments[:, channels:]
                    mx.random.seed(KEYFRAME_ENCODE_SEED)
                    latent = mean + mx.exp(0.5 * mx.clip(logvar, -30.0, 20.0)) * mx.random.normal(mean.shape)
                # A fresh seed-42 draw per reference, as the reference implementation's per-call
                # generator does; the sample rounds through float16 like every conditioning latent.
                latent = latent.astype(mx.float16).astype(mx.float32)
                normalized = (latent - latents_mean) / latents_std
                reference.num_latent_frames = int(latent.shape[2])
                reference.latent_height = int(latent.shape[3])
                reference.latent_width = int(latent.shape[4])
                rows = patchify_video_latents(normalized, self._dit_config.patch_size)
                mx.eval(rows)
                video_rows.append(rows)

            if reference.has_audio:
                waveform = mx.array(np.asarray(reference.waveform, np.float32))[:, None, :]
                mean, _ = self.audio_vae.encode(waveform)
                # Channel-major rows: the two stereo channels are two batch items of the mono VAE.
                latents = (mean - audio_mean) / audio_std
                reference.num_audio_latents = int(latents.shape[2])
                audio_rows.append(
                    latents.transpose(0, 2, 1).reshape(-1, self._audio_config.latent_channels)
                )
                mx.eval(audio_rows[-1])
        return (
            mx.concatenate(video_rows) if video_rows else None,
            mx.concatenate(audio_rows) if audio_rows else None,
        )

    def __call__(
        self,
        prompt: str,
        duration_seconds: float = 5.0,
        aspect: tuple[int, int] = (16, 9),
        num_inference_steps: int = 16,
        seed: int = 0,
        images: list | None = None,
        keyframe_anchors: tuple[str, ...] = (),
        references: list | None = None,
        height: int | None = None,
        width: int | None = None,
        drop_adaln: bool = True,
        block_cache_config: BlockCacheConfig | None = None,
        cache_text_conditioning: bool = False,
        verbose: bool = True,
    ) -> GenerationResult:
        """Generate a clip.

        Args:
            duration_seconds: 5 to 15; snapped up to the ``17n + 5`` frame grid the VAE encodes.
            num_inference_steps: the weights are CFG-distilled, so each step is one forward.
            keyframe_anchors: ``"first"`` / ``"last"`` per conditioning keyframe, in packed order.
            references: ``ref2va`` omni-references (:class:`ref2va.Reference`), in the order the
                model should read them. Needs the Ref2VA transformer as ``--transformer``.
            height, width: override the canvas ``aspect`` would resolve to. Both must be multiples
                of 32. H3 was released for a 768-pixel short edge only, so anything else is
                off-distribution — useful for exercising the pipeline, not for quality.
            cache_text_conditioning: disabled-by-default candidate that precomputes the refined
                text stream once and reuses it for every denoising DiT call.
        """
        run_started = time.perf_counter()
        if references is not None and (images or keyframe_anchors):
            raise ValueError("Keyframe images and Ref2VA references cannot be combined in one request.")

        # Geometry. References never bind the generated canvas: the aspect (or explicit size) does.
        if height is None or width is None:
            height, width = resolve_canvas_size(*aspect)
        elif height % 32 or width % 32:
            raise ValueError(f"`height` and `width` must be multiples of 32, got {height}x{width}.")
        num_frames = align_num_frames(int(round(duration_seconds * FPS)))
        num_latent_frames = video_latent_num_frames(num_frames)
        ratio = self._video_config.spatial_compression_ratio
        latent_height, latent_width = height // ratio, width // ratio
        num_audio_latents = audio_latent_num_frames(num_frames)
        patch_size = self._dit_config.patch_size

        prepared_references = None
        if references is not None:
            from .ref2va import check_references, prepare_references

            check_references(references)
            prepared_references, _ = prepare_references(
                references, num_frames, self._audio_config.sampling_rate
            )

        if self._low_memory:
            from .text_encoder import MiniMaxH3TextEncoder

            self.text_encoder = profiled_call(
                "load.text_encoder_low_memory",
                "load_overhead",
                lambda: MiniMaxH3TextEncoder(
                    self._text_encoder_path,
                    load_vision=bool(images or prepared_references is not None),
                    verbose=verbose,
                    tokenizer_dir=self._checkpoint_root / "tokenizer",
                    processor_dir=self._checkpoint_root / "processor",
                    stream_layers=True,
                ),
                eval_output=False,
            )

        # 1. Text conditioning. Keyframe and reference vision blocks come back tagged as *video* rows.
        if prepared_references is not None:
            prompt_embeds, text_token_tags = profiled_call(
                "pipeline.text_encoder_encode",
                "text_conditioning",
                lambda: self._encode_ref2va_text(prompt, prepared_references),
                metadata={"has_references": True},
            )
        else:
            prompt_embeds, text_token_tags = profiled_call(
                "pipeline.text_encoder_encode",
                "text_conditioning",
                lambda: self.text_encoder.encode(prompt, images),
                metadata={"has_images": bool(images)},
            )
        if self._low_memory:
            prompt_embeds = detach_bfloat16(prompt_embeds)
            text_token_tags = np.array(text_token_tags, copy=True)
            self._release_component("text_encoder")

        # 2. Conditioning (VAE phase) — run *before* the streaming DiT loads, so the VAE and the
        #    DiT never co-reside in low-memory mode.
        condition_rows = None
        ref_video_rows = None
        ref_audio_rows = None
        needs_video_vae = bool(
            images or (prepared_references is not None and any(r.kind != "audio" for r in prepared_references))
        )
        needs_audio_vae = prepared_references is not None and any(r.has_audio for r in prepared_references)
        if self._low_memory and (needs_video_vae or needs_audio_vae):
            from .load import load_audio_vae, load_video_vae

            if needs_video_vae:
                self.video_vae = profiled_call(
                    "load.video_vae_conditioning",
                    "load_overhead",
                    lambda: load_video_vae(self._checkpoint_root / "video_vae", encode_only=True),
                    eval_output=False,
                )
            if needs_audio_vae:
                self.audio_vae = profiled_call(
                    "load.audio_vae_conditioning",
                    "load_overhead",
                    lambda: load_audio_vae(self._checkpoint_root / "audio_vae"),
                    eval_output=False,
                )
        if images:
            condition_rows = profiled_call(
                "pipeline.encode_keyframes",
                "conditioning_encode",
                lambda: self._encode_keyframes(images, height, width),
            )
        if prepared_references is not None:
            ref_video_rows, ref_audio_rows = profiled_call(
                "pipeline.encode_references",
                "conditioning_encode",
                lambda: self._encode_reference_media(prepared_references),
            )
        if self._low_memory and (needs_video_vae or needs_audio_vae):
            if needs_video_vae:
                self._release_component("video_vae")
            if needs_audio_vae:
                self._release_component("audio_vae")

        if self._low_memory:
            from .streaming import load_streaming_dit

            self.dit, self._block_provider = profiled_call(
                "load.streaming_transformer_low_memory",
                "load_overhead",
                lambda: load_streaming_dit(
                    self._dit_path,
                    turbo_lora_path=self._turbo_lora_path,
                    turbo_lora_alpha=self._turbo_lora_alpha,
                    turbo_lora_scale=self._turbo_lora_scale,
                    block_load_mode=self._block_load_mode,
                    stream_block_group_size=self._stream_block_group_size,
                    verbose=verbose,
                ),
                eval_output=False,
            )
            self.set_dense_dequant_profile(
                self._dense_dequant_profile,
                attention_qkv_tile_size=self._dense_dequant_attention_qkv_tile_size,
                ffn_fc2_tile_size=self._dense_dequant_ffn_fc2_tile_size,
                attention_out_tile_size=self._dense_dequant_attention_out_tile_size,
            )
            if self._memory_pressure_guard:
                self._memory_guard_boundary()

        if prepared_references is not None:
            layout = build_ref2va_packed_sequence(
                text_token_tags,
                prepared_references,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                patch_size,
            )
        else:
            layout = build_packed_sequence(
                text_token_tags,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                patch_size,
                keyframe_anchors,
            )
        if verbose:
            print(f"canvas {width}x{height}, {num_frames} frames ({num_latent_frames} latent), "
                  f"{num_audio_latents} audio latents")
            print(f"packed sequence: {layout.sequence_length:,} rows "
                  f"({len(text_token_tags):,} text, {layout.num_condition_video_rows:,} condition)")

        # 4. Initial noise. Draw order matches the reference — the conditioning noise comes off the
        #    request generator first, then video, then audio — so a seed reproduces the same run.
        mx.random.seed(seed)
        if condition_rows is not None:
            condition_noise = mx.random.normal(condition_rows.shape).astype(mx.float32)
            # Anchors are not fully clean: they are noised to t = 0.999 and held there every step.
            condition_rows = MiniMaxH3Scheduler(shift=self.config.sigma_shift_video).scale_noise(
                condition_rows, KEYFRAME_NOISE_AUG, condition_noise
            )
        if ref_video_rows is not None:
            # Reference visuals ride at the keyframe conditioning level; reference soundtracks stay
            # clean (t = 1.0) and are concatenated without any noise draw at all.
            ref_noise = mx.random.normal(ref_video_rows.shape).astype(mx.float32)
            ref_video_rows = MiniMaxH3Scheduler(shift=self.config.sigma_shift_video).scale_noise(
                ref_video_rows, KEYFRAME_NOISE_AUG, ref_noise
            )

        latents = mx.random.normal(
            (1, self._video_config.latent_channels, num_latent_frames, latent_height, latent_width)
        ).astype(mx.float32)
        video_rows = patchify_video_latents(latents, patch_size)
        audio_rows = mx.random.normal(
            (num_audio_latents * AUDIO_CHANNELS, self._audio_config.latent_channels)
        ).astype(mx.float32)
        if condition_rows is not None:
            video_rows = mx.concatenate([condition_rows, video_rows])
        if ref_video_rows is not None:
            video_rows = mx.concatenate([ref_video_rows, video_rows])
        if ref_audio_rows is not None:
            audio_rows = mx.concatenate([ref_audio_rows, audio_rows])

        # 5. Two schedules over one shared forward.
        video_sched, audio_sched = self._build_schedules(num_inference_steps)
        timestep_table, plan = self._row_timestep_plan(layout, video_sched.timesteps, audio_sched.timesteps)
        profiled_call(
            "pipeline.adaln_cache_build",
            "load_overhead",
            lambda: self._ensure_cache(timestep_table, drop_adaln, verbose),
            eval_output=False,
            metadata={"timestep_count": timestep_table.shape[0], "drop_adaln": drop_adaln},
        )
        if self._memory_pressure_guard:
            self._memory_guard_boundary()

        n_cond_v = layout.num_condition_video_rows
        n_cond_a = layout.num_condition_audio_rows
        embeds = prompt_embeds.astype(mx.bfloat16)
        refined_text = None
        embeds_for_dit = embeds
        if cache_text_conditioning:
            started = time.perf_counter()
            refined_text = self.dit.precompute_text_conditioning(
                embeds,
                block_provider=self._block_provider,
            )
            mx.eval(refined_text)
            # The opt-in cached path no longer needs prompt embeddings during denoising.
            embeds_for_dit = None
            embeds = None
            prompt_embeds = None
            self._memory_guard_boundary()
            if verbose:
                print(
                    f"  text conditioning cache: {refined_text.nbytes / 1e6:.1f} MB "
                    f"in {time.perf_counter() - started:.1f}s"
                )
        block_cache = (
            BlockResidualCache(block_cache_config)
            if block_cache_config is not None
            else None
        )

        # 6. Denoise. One forward per step; only generated rows are written back, so the
        #    conditioning anchors survive without any masking.
        step_times = []
        for i, t in enumerate(video_sched.timesteps.tolist()):
            started = time.perf_counter()
            video_pred, audio_pred = profiled_call(
                "pipeline.dit_forward_step",
                "dit_forward_total",
                lambda i=i: self.dit(
                    video_rows[None].astype(mx.bfloat16),
                    audio_rows[None].astype(mx.bfloat16),
                    embeds_for_dit,
                    timestep_table,
                    plan[i],
                    layout.token_tags,
                    layout.position_ids,
                    layout.video_indices,
                    layout.audio_indices,
                    layout.text_indices,
                    modulation_cache=self._cache,
                    block_cache=block_cache,
                    block_cache_sigma=float(video_sched.sigmas[i].item()),
                    block_cache_step=i,
                    block_cache_total_steps=len(video_sched.timesteps),
                    block_provider=self._block_provider,
                    refined_text=refined_text,
                ),
                metadata={"step_index": i, "sigma": float(video_sched.sigmas[i].item())},
            )
            # Rebind rather than assign into a slice: the stepped result is a lazy graph reading the
            # very rows it would overwrite, and with conditioning rows present the two halves must
            # stay distinct. Concatenating is unambiguous and costs nothing next to the forward.
            stepped_video = video_sched.step(
                video_pred[0, n_cond_v:].astype(mx.float32), float(t), video_rows[n_cond_v:]
            )
            stepped_audio = audio_sched.step(
                audio_pred[0, n_cond_a:].astype(mx.float32),
                float(audio_sched.timesteps[i].item()),
                audio_rows[n_cond_a:],
            )
            video_rows = (
                mx.concatenate([video_rows[:n_cond_v], stepped_video]) if n_cond_v else stepped_video
            )
            audio_rows = (
                mx.concatenate([audio_rows[:n_cond_a], stepped_audio]) if n_cond_a else stepped_audio
            )
            mx.eval(video_rows, audio_rows)
            if self._memory_pressure_guard:
                video_pred = None
                audio_pred = None
                stepped_video = None
                stepped_audio = None
                self._memory_guard_boundary()
            step_times.append(time.perf_counter() - started)
            if verbose:
                done = i + 1
                mean = sum(step_times) / len(step_times)
                eta = mean * (len(video_sched.timesteps) - done)
                print(f"  step {done}/{len(video_sched.timesteps)}  "
                      f"{step_times[-1]:.1f}s  eta {eta / 60:.1f} min", flush=True)

        # 7. Decode both modalities.
        if self._low_memory:
            video_rows = mx.array(np.array(video_rows[n_cond_v:]), dtype=mx.float32)
            audio_rows = mx.array(np.array(audio_rows[n_cond_a:]), dtype=mx.float32)
            self._cache = None
            self._cache_timesteps = None
            self._block_provider = None
            self._release_component("dit")

            from .load import load_audio_vae, load_video_vae

            self.video_vae = profiled_call(
                "load.video_vae_low_memory",
                "load_overhead",
                lambda: load_video_vae(self._checkpoint_root / "video_vae"),
                eval_output=False,
            )
            video = profiled_call(
                "pipeline.video_vae_decode",
                "vae_decode",
                lambda: self._decode_video(video_rows, num_latent_frames, latent_height, latent_width),
                eval_output=False,
            )
            self._release_component("video_vae")

            self.audio_vae = profiled_call(
                "load.audio_vae_low_memory",
                "load_overhead",
                lambda: load_audio_vae(self._checkpoint_root / "audio_vae"),
                eval_output=False,
            )
            audio = profiled_call(
                "pipeline.audio_vae_decode",
                "vae_decode",
                lambda: self._decode_audio(audio_rows, num_audio_latents),
                eval_output=False,
            )
            self._release_component("audio_vae")
        else:
            video = profiled_call(
                "pipeline.video_vae_decode",
                "vae_decode",
                lambda: self._decode_video(video_rows[n_cond_v:], num_latent_frames, latent_height, latent_width),
                eval_output=False,
            )
            audio = profiled_call(
                "pipeline.audio_vae_decode",
                "vae_decode",
                lambda: self._decode_audio(audio_rows[n_cond_a:], num_audio_latents),
                eval_output=False,
            )
        total = time.perf_counter() - run_started
        return GenerationResult(
            video=video,
            audio=audio,
            sample_rate=self._audio_config.sampling_rate,
            seconds_per_step=sum(step_times) / max(len(step_times), 1),
            total_seconds=total,
            block_cache_stats=block_cache.stats() if block_cache is not None else None,
        )

    # -- decoding -----------------------------------------------------------------------------

    def _decode_video(self, rows, num_latent_frames, latent_height, latent_width) -> np.ndarray:
        cfg = self._video_config
        latents = unpatchify_video_tokens(
            rows, num_latent_frames, latent_height, latent_width, cfg.latent_channels, self._dit_config.patch_size
        )
        mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        latents = latents * std + mean

        frames = np.array(self.video_vae.decode(latents.astype(mx.float32)))
        # The VAE decodes into ImageNet-normalized RGB over a [0, 1] base range.
        pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
        frames = frames * pixel_std + pixel_mean
        frames = np.clip(frames, 0.0, 1.0)[0].transpose(1, 2, 3, 0)  # -> (F, H, W, 3)
        return (frames * 255.0 + 0.5).astype(np.uint8)

    def _decode_audio(self, rows, num_audio_latents) -> np.ndarray:
        cfg = self._audio_config
        latents = unpack_audio_tokens(rows, num_audio_latents)
        mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1)
        std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1)
        latents = latents * std + mean
        waveform = np.array(self.audio_vae.decode(latents.astype(mx.float32)))
        return waveform[:, 0, :].astype(np.float32)  # (2, samples), one row per stereo channel
