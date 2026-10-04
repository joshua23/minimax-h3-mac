"""MiniMax-H3's Qwen3-VL-32B conditioner, in MLX.

H3 does not use Qwen3-VL as a language model. It reads the **unnormalized** hidden state after the
50th of its 64 decoder layers (``hidden_states[50]``, where ``hidden_states[0]`` is the embedding
output) and feeds that straight into the DiT's ``condition_proj``. The language-model head, the
final norm and the last 14 decoder layers are never evaluated.

That is worth exploiting: the port loads **only the 50 layers it reads**, skipping ``lm_head``
(151936 x 5120) and layers 50-63 entirely. For a text-only request the vision tower is skipped too.

The transformer stack itself is mlx-vlm's ``qwen3_vl`` implementation — it already has the
interleaved M-RoPE, the ``mrope_section`` split and the deepstack visual merge — so this module only
supplies H3's request presentation, the truncated forward, and a loader that reads a subset.

**Request presentation** (from the reference; no chat template and no special tokens anywhere):
each keyframe contributes a ``"<Picture i>: "`` label followed by a vision block
(``<|vision_start|>``, one ``<|image_pad|>`` per merged patch, ``<|vision_end|>``), then the prompt
verbatim. The rows of a vision block are tagged **video**, not text — that tag is what the DiT's
AdaLN modulation keys off.
"""

from __future__ import annotations

import gc
import glob
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import TAG_TEXT, TAG_VIDEO
from .packing import TEXT_ENCODER_LAYER


def _deepstack_process(hidden_states: mx.array, visual_pos_masks: mx.array, visual_embeds: mx.array) -> mx.array:
    """Add one deepstack visual embed level at the vision-token rows of ``hidden_states``.

    Mirrors mlx_vlm's ``Qwen3VLModel._deepstack_process`` (which reads no instance state): the
    streamed loop cannot call it through ``self.language`` because that module is never built.
    """
    batch_size = hidden_states.shape[0]
    updated_batches = []
    offset = 0
    for b in range(batch_size):
        batch_mask = visual_pos_masks[b]
        batch_hidden = hidden_states[b]
        batch_indices = mx.array(np.where(batch_mask)[0], dtype=mx.uint32)
        n_visual = len(batch_indices)
        if n_visual == 0:
            updated_batches.append(batch_hidden)
            continue
        sample_embeds = visual_embeds[offset : offset + n_visual]
        offset += n_visual
        if sample_embeds.shape[0] != n_visual:
            updated_batches.append(batch_hidden)
            continue
        batch_result = mx.array(batch_hidden)  # avoid modifying in-place
        batch_result = batch_result.at[batch_indices].add(sample_embeds)
        updated_batches.append(batch_result)
    return mx.stack(updated_batches, axis=0)


def _splice_vision_rows(inputs_embeds: mx.array, image_mask: mx.array, hidden: mx.array) -> mx.array:
    """Replace the embedding rows at image-token positions with the vision tower's rows.

    Follows mlx_vlm's ``masked_scatter``: a plain ``mx.where`` cannot broadcast the (L,) token mask
    against the (num_image_tokens, D) update rows, and this masked scatter is exactly what the
    torch reference's ``inputs_embeds[image_mask] = hidden`` computes.
    """
    n_mask = int(mx.sum(image_mask))
    if n_mask != hidden.shape[0]:
        raise ValueError(
            f"Image features and image tokens do not match: tokens {n_mask}, features {hidden.shape[0]}"
        )
    shape = inputs_embeds.shape
    flat = mx.flatten(inputs_embeds)
    mask_flat = mx.flatten(mx.broadcast_to(image_mask[..., None], shape))
    positions = mx.array(np.where(mask_flat)[0], mx.uint32)
    flat[positions] = mx.flatten(hidden)
    return mx.reshape(flat, shape)


class MiniMaxH3TextEncoder:
    """Qwen3-VL-32B truncated to the layers MiniMax-H3 actually conditions on."""

    def __init__(
        self,
        model_dir: str | Path,
        num_layers: int = TEXT_ENCODER_LAYER,
        dtype: mx.Dtype = mx.bfloat16,
        load_vision: bool = True,
        verbose: bool = False,
        tokenizer_dir: str | Path | None = None,
        processor_dir: str | Path | None = None,
        stream_layers: bool = False,
    ):
        from mlx_vlm.models.qwen3_vl.config import ModelConfig, TextConfig, VisionConfig
        from mlx_vlm.models.qwen3_vl.language import Qwen3VLDecoderLayer, Qwen3VLModel
        from mlx_vlm.models.qwen3_vl.vision import VisionModel

        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as fh:
            raw = json.load(fh)

        full_layers = raw["text_config"]["num_hidden_layers"]
        if full_layers <= num_layers:
            raise ValueError(
                f"MiniMax-H3 conditions on hidden_states[{num_layers}] of its Qwen3-VL conditioner, "
                f"which needs more than {num_layers} decoder layers, but the checkpoint has "
                f"{full_layers}. The last hidden state of a stack truncated to exactly {num_layers} "
                "layers is post-norm and is not the conditioning MiniMax-H3 expects."
            )

        # Streamed mode supports images: the BF16 vision tower (~1.2 GB) streams in for the one
        # encode call, its outputs are materialized, and it is released before any decoder layer
        # becomes resident — so peak residency is the tower *or* the embedding table + layer slot,
        # never both towers.
        self.num_layers = num_layers
        self.full_layers = full_layers
        self.dtype = dtype
        self.stream_layers = bool(stream_layers)

        text_raw = dict(raw["text_config"])
        # Resident mode builds the 50 evaluated layers. Streamed mode builds only one reusable
        # decoder-layer slot; its original BF16 weights are replaced before every layer forward.
        text_raw["num_hidden_layers"] = 1 if self.stream_layers else num_layers
        self.text_config = TextConfig.from_dict(text_raw)
        self.vision_config = VisionConfig.from_dict(raw["vision_config"])
        self.model_config = ModelConfig.from_dict(
            {
                **raw,
                "text_config": text_raw,
                "vision_config": raw["vision_config"],
                "model_type": raw.get("model_type", "qwen3_vl"),
            }
        )
        self.model_config.text_config = self.text_config
        self.model_config.vision_config = self.vision_config

        self.language = None if self.stream_layers else Qwen3VLModel(self.text_config)
        self._stream_layer = Qwen3VLDecoderLayer(self.text_config, layer_idx=0) if self.stream_layers else None
        self._vision_enabled = bool(load_vision)
        self.vision = VisionModel(self.vision_config) if load_vision else None
        quant_path = model_dir / "quant_config.json"
        self.quantized = quant_path.exists()
        if self.quantized:
            import mlx.nn as nn
            from .quantize import apply_quantized_slots

            with quant_path.open() as handle:
                quant = json.load(handle)
            self.quant_config = quant

            def quantize_language(path, module):
                weight = getattr(module, "weight", None)
                should_quantize = (
                    hasattr(module, "to_quantized")
                    and isinstance(weight, mx.array)
                    and weight.ndim == 2
                    and weight.shape[-1] % int(quant["group_size"]) == 0
                )
                if not should_quantize:
                    return False
                return {
                    "group_size": int(quant["group_size"]),
                    "bits": int(quant["bits"]),
                    "mode": str(quant.get("mode", "affine")),
                }

            apply_quantized_slots(
                self._stream_layer if self.stream_layers else self.language,
                quantize_language,
            )
        if self.stream_layers:
            from .selective_loading import load_weight_map

            self._weight_map = load_weight_map(model_dir)
            self.skipped_tensors = len(self._weight_map) - sum(
                key == "model.language_model.embed_tokens.weight"
                or any(key.startswith(f"model.language_model.layers.{i}.") for i in range(num_layers))
                for key in self._weight_map
            )
            if verbose:
                precision = "quantized" if self.quantized else "full-precision"
                print(f"  text encoder: {precision} weights, streaming {num_layers} layers")
        else:
            self.quant_config = None
            self._load_weights(model_dir, dtype, verbose)

        self.image_token_id = raw["image_token_id"]
        self.video_token_id = raw.get("video_token_id", self.image_token_id)
        self.vision_start_token_id = raw["vision_start_token_id"]
        self.vision_end_token_id = raw["vision_end_token_id"]
        self.merge_size = self.vision_config.spatial_merge_size

        self._tokenizer = None
        self._processor = None
        self._model_dir = model_dir
        root = model_dir.parent
        self._tokenizer_dir = (
            Path(tokenizer_dir)
            if tokenizer_dir is not None
            else (root / "tokenizer" if (root / "tokenizer").exists() else model_dir)
        )
        self._processor_dir = (
            Path(processor_dir)
            if processor_dir is not None
            else (root / "processor" if (root / "processor").exists() else model_dir)
        )

    # -- loading ---------------------------------------------------------------------------

    def _wanted(self, key: str) -> str | None:
        """Map a checkpoint key onto this module's parameter path, or ``None`` to skip it."""
        if key.startswith("lm_head"):
            return None  # never evaluated
        if key.startswith("model.language_model."):
            rest = key[len("model.language_model.") :]
            if rest.startswith("layers."):
                index = int(rest.split(".")[1])
                if index >= self.num_layers:
                    return None  # beyond the conditioning layer
            # `norm` is loaded (it is 5120 floats) to keep the module tree complete, but it is never
            # applied: H3 reads the hidden state *before* the final norm.
            return ("language", rest)
        if key.startswith("model.visual."):
            if self.vision is None:
                return None
            return ("vision", key[len("model.visual.") :])
        return None

    def _load_weights(self, model_dir: Path, dtype: mx.Dtype, verbose: bool) -> None:
        from mlx.utils import tree_flatten, tree_unflatten

        shards = sorted(glob.glob(str(model_dir / "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No safetensors in {model_dir}.")

        expected = {
            "language": {k for k, _ in tree_flatten(self.language.parameters())},
            "vision": set() if self.vision is None else {k for k, _ in tree_flatten(self.vision.parameters())},
        }
        remaining = {bucket: set(keys) for bucket, keys in expected.items()}
        loaded = 0
        skipped = 0
        for shard in shards:
            updates: dict[str, list[tuple[str, mx.array]]] = {"language": [], "vision": []}
            for key, tensor in mx.load(shard).items():
                target = self._wanted(key)
                if target is None:
                    skipped += 1
                    continue
                bucket, path = target
                if path not in expected[bucket]:
                    skipped += 1
                    continue
                updates[bucket].append(
                    (path, tensor if self.quantized else tensor.astype(dtype))
                )
                remaining[bucket].discard(path)
                loaded += 1

            for bucket, module in (("language", self.language), ("vision", self.vision)):
                if module is None or not updates[bucket]:
                    continue
                bucket_updates = dict(updates[bucket])
                # The vision checkpoint stores conv weights in torch layout; the tower's own
                # sanitize moves ``patch_embed.proj`` into the channel-last layout mlx's conv3d
                # expects. Language keys are already stored in mlx layout.
                if bucket == "vision":
                    bucket_updates = module.sanitize(bucket_updates)
                module.update(tree_unflatten(list(bucket_updates.items())))
                mx.eval(*(tensor for _, tensor in bucket_updates.items()))
            if verbose:
                print(f"  {Path(shard).name}: {loaded} tensors loaded")

        for bucket in ("language", "vision"):
            missing = sorted(remaining[bucket])
            if missing:
                raise KeyError(
                    f"{bucket} encoder missing {len(missing)} tensors, e.g. {missing[:4]}."
                )
        self.skipped_tensors = skipped

    # -- tokenizer / processor -------------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(self._tokenizer_dir))
            required_token_id = max(
                self.image_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            )
            if len(tokenizer) <= required_token_id:
                raise ValueError(
                    f"Tokenizer at {self._tokenizer_dir} has only {len(tokenizer)} tokens, "
                    f"but the checkpoint requires token id {required_token_id}. "
                    "Pass the source FL2VA tokenizer directory via `tokenizer_dir`."
                )
            self._tokenizer = tokenizer
        return self._tokenizer

    @property
    def processor(self):
        if self._processor is None:
            from types import SimpleNamespace

            self._processor = SimpleNamespace(
                image_processor=self._load_image_processor(),
                tokenizer=self.tokenizer,
            )
        return self._processor

    def _load_image_processor(self):
        """Load the PIL-backend image processor without the torch-gated ``Auto`` path.

        The release's ``processor_class`` builds the video processor too, which requires torch;
        H3's conditioner only ever feeds still images. transformers 5.x gates ``AutoImageProcessor``
        behind the same import, so resolve the class through the registry directly and prefer the
        PIL backend.
        """
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING

        entry = IMAGE_PROCESSOR_MAPPING[CONFIG_MAPPING[self.model_config.model_type]]
        if isinstance(entry, dict):
            classes = [entry[key] for key in ("pil", "torchvision") if key in entry]
            classes += [value for key, value in entry.items() if key not in ("pil", "torchvision")]
        elif isinstance(entry, (tuple, list)):
            classes = list(entry)
        else:
            classes = [entry]
        last_error: Exception | None = None
        for cls in classes:
            try:
                return cls.from_pretrained(str(self._processor_dir))
            except Exception as error:  # probe every backend, keep the first that loads
                last_error = error
        raise RuntimeError(
            f"No usable image processor backend for {self._processor_dir}"
        ) from last_error

    # -- request presentation --------------------------------------------------------------

    def build_request(self, prompt: str, images: list | None = None):
        """Build H3's token sequence and its per-row modality tags.

        Returns ``(input_ids, token_tags, vision_inputs)``; ``vision_inputs`` is ``None`` for a
        text-only request, otherwise the processor's ``pixel_values`` / ``image_grid_thw``.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"`prompt` must be a single string, got {type(prompt).__name__}.")

        token_ids: list[int] = []
        token_tags: list[int] = []
        vision_inputs = None

        if images:
            vision = self.processor.image_processor(images=images, return_tensors="np")
            pixel_values = np.asarray(vision["pixel_values"])
            grid_thw = np.asarray(vision["image_grid_thw"])
            merge = self.processor.image_processor.merge_size**2
            start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            end = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")

            for index in range(len(images)):
                num_image_tokens = int(grid_thw[index].prod()) // merge
                label_ids = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = [start] + [pad] * num_image_tokens + [end]
                token_ids += label_ids + vision_ids
                # The whole vision block is tagged *video*; only the label stays text.
                token_tags += [TAG_TEXT] * len(label_ids) + [TAG_VIDEO] * len(vision_ids)
            vision_inputs = (pixel_values, grid_thw)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [TAG_TEXT] * len(prompt_ids)
        if not token_ids:
            raise ValueError(
                "The request produced no input token IDs. Check `tokenizer_dir` and provide "
                "a prompt that tokenizes to at least one token."
            )

        return (
            mx.array(np.array([token_ids], dtype=np.int32)),
            np.array(token_tags, dtype=np.int64),
            vision_inputs,
        )

    # -- forward ---------------------------------------------------------------------------

    def _load_stream_tensor(self, key: str) -> mx.array:
        from .selective_loading import load_selected_mlx_tensors

        if key not in self._weight_map:
            raise KeyError(f"text encoder is missing streamed tensor {key!r}")
        tensor = load_selected_mlx_tensors(self._model_dir, [key])[key]
        return tensor if self.quantized else tensor.astype(self.dtype)

    def _load_stream_layer(self, layer_idx: int) -> None:
        from mlx.utils import tree_flatten, tree_unflatten
        from .selective_loading import load_selected_mlx_tensors

        layer = self._stream_layer
        expected = {key for key, _ in tree_flatten(layer.parameters())}
        # The previous layer's forward has already been evaluated. Remove its arrays before reading
        # the next layer so peak residency is one layer rather than old+new during the handoff.
        layer.update(tree_unflatten([(key, mx.array(0, dtype=mx.uint32)) for key in expected]))
        gc.collect()
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
        prefix = f"model.language_model.layers.{layer_idx}."
        source_by_target = {
            key[len(prefix) :]: key for key in self._weight_map if key.startswith(prefix)
        }
        missing = sorted(expected - source_by_target.keys())
        if missing:
            raise KeyError(f"text encoder layer {layer_idx} is missing tensors, e.g. {missing[:4]}")
        loaded = load_selected_mlx_tensors(
            self._model_dir,
            [source_by_target[key] for key in sorted(expected)],
        )
        updates = [
            (
                key,
                loaded[source_by_target[key]]
                if self.quantized
                else loaded[source_by_target[key]].astype(self.dtype),
            )
            for key in sorted(expected)
        ]
        layer.update(tree_unflatten(updates))
        mx.eval(*(tensor for _, tensor in updates))

    def _load_vision_weights(self) -> None:
        """Stream the BF16 vision tower in for a single encode call.

        Streamed mode never loads weights at construction, so the tower's arrays do not exist
        until this runs. Its checkpoint keys live under ``model.visual.``; the deepstack mergers
        are part of the tower, so once its forward has produced the conditioning embeds nothing
        of it is needed again and the caller releases it before the first decoder layer. A released
        tower is rebuilt here on the next vision request.
        """
        from mlx.utils import tree_flatten, tree_unflatten

        from .selective_loading import load_selected_mlx_tensors

        if self.vision is None:
            from mlx_vlm.models.qwen3_vl.vision import VisionModel

            self.vision = VisionModel(self.vision_config)
        vision = self.vision
        expected = {key for key, _ in tree_flatten(vision.parameters())}
        prefix = "model.visual."
        source_by_target = {key[len(prefix) :]: key for key in self._weight_map if key.startswith(prefix)}
        missing = sorted(expected - source_by_target.keys())
        if missing:
            raise KeyError(f"vision tower is missing {len(missing)} tensors, e.g. {missing[:4]}")
        loaded = load_selected_mlx_tensors(
            self._model_dir,
            [source_by_target[key] for key in sorted(expected)],
        )
        # The checkpoint stores conv weights in torch layout; the tower's own sanitize moves
        # ``patch_embed.proj`` into the channel-last layout mlx's conv3d expects.
        loaded = vision.sanitize(loaded)
        updates = [
            (key, loaded[source_by_target[key]].astype(self.dtype))
            for key in sorted(expected)
        ]
        vision.update(tree_unflatten(updates))
        mx.eval(*(tensor for _, tensor in updates))

    def _release_vision(self) -> None:
        """Drop the vision tower and return its buffers once its outputs are materialized."""
        self.vision = None
        gc.collect()
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()

    def _streamed_input_embeddings(self, input_ids: mx.array) -> mx.array:
        """Materialize the request's token embeddings without keeping the embedding table resident."""
        embedding_key = "model.language_model.embed_tokens.weight"
        embedding = self._load_stream_tensor(embedding_key)
        if self.quantized:
            from .selective_loading import load_selected_mlx_tensors

            stem = embedding_key[: -len("weight")]
            aux_keys = [stem + "scales", stem + "biases"]
            aux = load_selected_mlx_tensors(self._model_dir, aux_keys)
            # QuantizedEmbedding cannot be used as the streamed layer slot. Select the prompt
            # rows while they are still packed, then dequantize only those rows; dequantizing
            # the complete 151936 x 5120 table would defeat low-memory text streaming.
            h = mx.dequantize(
                embedding[input_ids],
                aux[stem + "scales"][input_ids],
                aux[stem + "biases"][input_ids],
                group_size=int(self.quant_config["group_size"]),
                bits=int(self.quant_config["bits"]),
                mode=str(self.quant_config.get("mode", "affine")),
            )
            del aux
        else:
            h = embedding[input_ids]
        mx.eval(h)
        del embedding
        gc.collect()
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
        return h

    def _hidden_states(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        inputs_embeds: mx.array | None = None,
        visual_pos_masks: mx.array | None = None,
        deepstack_visual_embeds: list | None = None,
    ) -> mx.array:
        """Run the truncated stack and return the hidden state **before** the final norm."""
        from mlx_vlm.models.base import create_attention_mask

        if self.stream_layers:
            h = inputs_embeds if inputs_embeds is not None else self._streamed_input_embeddings(input_ids)
            mask = create_attention_mask(h, None)
            layer = self._stream_layer
            position_embeddings = None
            if position_ids is not None and not layer.self_attn.rotary_emb.fused_apply:
                position_embeddings = layer.self_attn.rotary_emb(h, position_ids)
            for layer_idx in range(self.num_layers):
                self._load_stream_layer(layer_idx)
                h = layer(h, mask, None, position_ids, position_embeddings)
                if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                    h = _deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[layer_idx])
                # Materialize before replacing this slot with the next layer's weights, ensuring
                # that at most one full decoder layer is resident.
                mx.eval(h)
                gc.collect()
            return h

        model = self.language
        h = model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        mask = create_attention_mask(h, None)

        position_embeddings = None
        if position_ids is not None and not model.layers[0].self_attn.rotary_emb.fused_apply:
            position_embeddings = model.layers[0].self_attn.rotary_emb(h, position_ids)

        for layer_idx, layer in enumerate(model.layers):
            h = layer(h, mask, None, position_ids, position_embeddings)
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                h = model._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[layer_idx])
        # No `model.norm(h)`: H3 conditions on the unnormalized state.
        return h

    def encode(self, prompt: str, images: list | None = None) -> tuple[mx.array, np.ndarray]:
        """Encode a request into ``((1, num_text_tokens, 5120), (num_text_tokens,))``."""
        input_ids, token_tags, vision_inputs = self.build_request(prompt, images)
        if vision_inputs is None:
            return self.encode_presentation(input_ids, token_tags, None, None, None, None)
        pixel_values, grid_np = vision_inputs
        return self.encode_presentation(input_ids, token_tags, pixel_values, grid_np, None, None)

    def encode_presentation(
        self,
        input_ids: mx.array,
        token_tags: np.ndarray,
        image_pixels: np.ndarray | None,
        image_grids: np.ndarray | None,
        video_pixels: np.ndarray | None,
        video_grids: np.ndarray | None,
    ) -> tuple[mx.array, np.ndarray]:
        """Encode a prebuilt request presentation.

        ``image_pixels`` / ``video_pixels`` are the processors' patch arrays —
        ``(num_patches, channels * temporal * patch * patch)`` each — and the grids are ``(n, 3)``. The
        vision tower sees images and videos in one forward (images first), and the feature rows are
        reordered into the combined vision mask's sequence order so both the splice and the
        deepstack merge line up with the request's interleaved vision blocks.
        """
        from mlx_vlm.models.qwen3_vl.language import LanguageModel

        inputs_embeds = None
        visual_pos_masks = None
        deepstack_embeds = None
        image_grid_thw = None
        video_grid_thw = None
        has_vision = image_pixels is not None or video_pixels is not None

        if has_vision:
            if not self._vision_enabled:
                raise ValueError("This encoder was built with `load_vision=False`; it cannot take images.")
            parts, grids = [], []
            if image_pixels is not None:
                parts.append(mx.array(np.asarray(image_pixels)))
                image_grid_thw = mx.array(np.asarray(image_grids, np.int32))
                grids.append(np.asarray(image_grids, np.int64))
            if video_pixels is not None:
                parts.append(mx.array(np.asarray(video_pixels)))
                video_grid_thw = mx.array(np.asarray(video_grids, np.int32))
                grids.append(np.asarray(video_grids, np.int64))
            if self.stream_layers:
                self._load_vision_weights()
            hidden, deepstack_embeds = self.vision(
                mx.concatenate(parts, axis=0).astype(self.dtype),
                mx.array(np.concatenate(grids, axis=0).astype(np.int32)),
                output_hidden_states=True,
            )
            if self.stream_layers:
                # The tower has produced everything the language stack needs. Materialize those
                # outputs and free the tower before the embedding table or any decoder layer can
                # push peak residency past one component at a time.
                mx.eval(hidden, *(embed for embed in deepstack_embeds))
                self._release_vision()

            image_mask = input_ids == self.image_token_id
            video_mask = input_ids == self.video_token_id
            visual_pos_masks = image_mask | video_mask
            if video_pixels is not None and bool(mx.max(video_mask).item()):
                # The tower emitted image rows then video rows, but the vision blocks appear in the
                # sequence in request order. Reorder the feature rows (and every deepstack level)
                # into the combined mask's sequence order so the splice and the merge line up.
                img_positions = np.where(np.asarray(image_mask[0]))[0]
                vid_positions = np.where(np.asarray(video_mask[0]))[0]
                feature_rows = np.concatenate(
                    [np.arange(len(img_positions)), len(img_positions) + np.arange(len(vid_positions))]
                )
                combined = np.concatenate([img_positions, vid_positions])
                rows = mx.array(feature_rows[np.argsort(combined)].astype(np.int32))
                hidden = hidden[rows]
                deepstack_embeds = [embed[rows] for embed in deepstack_embeds]
            if not self.stream_layers:
                inputs_embeds = _splice_vision_rows(
                    self.language.embed_tokens(input_ids), visual_pos_masks, hidden
                )

        # Qwen3-VL's 3D M-RoPE index, derived from the vision-start/pad token ids.
        position_ids, _ = LanguageModel.get_rope_index(
            self,
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=None,
        )

        if self.stream_layers and has_vision:
            # Streamed splicing: build the request embedding here so the vision rows replace the
            # vision-token rows before any decoder layer streams in, then let go of the table.
            text_embeds = self._streamed_input_embeddings(input_ids)
            inputs_embeds = _splice_vision_rows(text_embeds, visual_pos_masks, hidden)
            mx.eval(inputs_embeds)

        hidden_states = self._hidden_states(
            input_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_embeds if has_vision else None,
        )
        mx.eval(hidden_states)
        return hidden_states, token_tags

    # `LanguageModel.get_rope_index` reads `self.config`; expose the same attribute.
    @property
    def config(self):
        return self.model_config
