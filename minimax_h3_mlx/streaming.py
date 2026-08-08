"""Low-memory, one-block-at-a-time loading for quantized MiniMax-H3 DiT."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .config import DiTConfig
from .dit import MiniMaxH3DiT, TransformerBlock
from .quantize import QuantConfig, _class_predicate, apply_quantization_structure


class _BlockSlot(nn.Module):
    """One reusable block under the same `blocks.0` path as a checkpoint."""

    def __init__(self, config: DiTConfig, quantization: QuantConfig) -> None:
        super().__init__()
        self.blocks = [TransformerBlock(config)]
        apply_quantization_structure(self, quantization)


class QuantizedBlockProvider:
    """Load one quantized transformer block lazily from indexed safetensors."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.config = DiTConfig.from_json(self.model_dir / "config.json")
        with (self.model_dir / "quant_config.json").open() as handle:
            raw_quant = json.load(handle)
        self.quantization = QuantConfig(
            bits=int(raw_quant["bits"]),
            group_size=int(raw_quant["group_size"]),
            quantize_adaln=bool(raw_quant.get("quantize_adaln", False)),
            adaln_bits=int(raw_quant.get("adaln_bits") or 8),
        )
        with (self.model_dir / "model.safetensors.index.json").open() as handle:
            self.weight_map = json.load(handle)["weight_map"]
        self.slot = _BlockSlot(self.config, self.quantization)
        self.expected = {key for key, _ in tree_flatten(self.slot.parameters())}
        self.current_index: int | None = None
        self.logical_bytes_loaded = 0

    @property
    def block_count(self) -> int:
        return self.config.num_layers

    def load_block(
        self,
        index: int,
        *,
        include_adaln: bool = True,
        adaln_only: bool = False,
    ) -> TransformerBlock:
        if not 0 <= index < self.block_count:
            raise IndexError(f"block index {index} outside 0..{self.block_count - 1}")
        source_prefix = f"blocks.{index}."
        target_prefix = "blocks.0."
        source_keys = [key for key in self.weight_map if key.startswith(source_prefix)]
        if adaln_only:
            source_keys = [key for key in source_keys if ".adaln_proj." in key]
        elif not include_adaln:
            source_keys = [key for key in source_keys if ".adaln_proj." not in key]
        by_shard: dict[str, list[str]] = {}
        for key in source_keys:
            by_shard.setdefault(self.weight_map[key], []).append(key)

        updates: list[tuple[str, mx.array]] = []
        for shard_name, keys in by_shard.items():
            arrays = mx.load(str(self.model_dir / shard_name))
            for source_key in keys:
                target_key = target_prefix + source_key[len(source_prefix) :]
                if target_key not in self.expected:
                    raise KeyError(f"streaming slot has no parameter {target_key!r}")
                tensor = arrays[source_key]
                updates.append((target_key, tensor))
                self.logical_bytes_loaded += tensor.nbytes

        if adaln_only:
            expected = {key for key in self.expected if ".adaln_proj." in key}
        elif include_adaln:
            expected = self.expected
        else:
            expected = {key for key in self.expected if ".adaln_proj." not in key}
        found = {key for key, _ in updates}
        missing = sorted(expected - found)
        if missing:
            raise KeyError(f"block {index} is missing {len(missing)} tensors, e.g. {missing[:4]}")
        self.slot.update(tree_unflatten(updates))
        self.current_index = index
        return self.slot.blocks[0]


def load_streaming_dit(
    model_dir: str | Path,
    *,
    verbose: bool = False,
) -> tuple[MiniMaxH3DiT, QuantizedBlockProvider]:
    """Load static DiT weights and leave the 50 main blocks file-backed."""
    model_dir = Path(model_dir)
    provider = QuantizedBlockProvider(model_dir)
    model = MiniMaxH3DiT(provider.config, build_blocks=False)

    # Quantize token-refiner linears so the non-block checkpoint keys match.
    predicate = _class_predicate(provider.quantization)

    def static_predicate(path: str, module: nn.Module):
        if path.startswith("blocks."):
            return False
        return predicate(path, module)

    nn.quantize(
        model,
        group_size=provider.quantization.group_size,
        bits=provider.quantization.bits,
        class_predicate=static_predicate,
    )

    expected = {
        key
        for key, _ in tree_flatten(model.parameters())
        if not key.startswith("blocks.")
    }
    source_keys = [key for key in provider.weight_map if not key.startswith("blocks.")]
    by_shard: dict[str, list[str]] = {}
    for key in source_keys:
        by_shard.setdefault(provider.weight_map[key], []).append(key)

    updates: list[tuple[str, mx.array]] = []
    for shard_name, keys in by_shard.items():
        arrays = mx.load(str(model_dir / shard_name))
        for key in keys:
            if key in expected:
                updates.append((key, arrays[key]))

    found = {key for key, _ in updates}
    missing = sorted(expected - found)
    if missing:
        raise KeyError(f"streaming DiT static weights missing {len(missing)} tensors, e.g. {missing[:4]}")
    model.update(tree_unflatten(updates))
    mx.eval(*(tensor for _, tensor in updates))
    if verbose:
        size = sum(tensor.nbytes for _, tensor in updates) / 1e9
        print(f"loaded {len(updates)} static tensors ({size:.2f} GB); main blocks stream lazily")
    return model, provider


__all__ = ["QuantizedBlockProvider", "load_streaming_dit"]