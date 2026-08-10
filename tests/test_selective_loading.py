"""Tiny lossless selective-loading prototype tests.

Synthetic shards only: no real model downloads and no generation.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.quantize import QuantConfig
from minimax_h3_mlx.block_local_reshard import create_block_local_layout
from minimax_h3_mlx.selective_loading import (
    build_tensor_plan,
    fingerprint_tensors,
    load_selected_mlx_tensors,
    load_selected_tensors,
)
from minimax_h3_mlx.streaming import QuantizedBlockProvider, _BlockSlot


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def write_index(root: Path) -> dict[str, np.ndarray]:
    tensors = {
        "blocks.0.attn.qkv_proj.weight": np.arange(12, dtype=np.uint32).reshape(3, 4),
        "blocks.0.attn.qkv_proj.scales": (np.arange(6, dtype=np.float32) / 10).reshape(2, 3),
        "time_embedder.proj_in.bias": np.linspace(-1.0, 1.0, 5, dtype=np.float32),
    }
    save_file(
        {
            "blocks.0.attn.qkv_proj.weight": tensors["blocks.0.attn.qkv_proj.weight"],
            "blocks.0.attn.qkv_proj.scales": tensors["blocks.0.attn.qkv_proj.scales"],
        },
        str(root / "shard-a.safetensors"),
    )
    save_file(
        {"time_embedder.proj_in.bias": tensors["time_embedder.proj_in.bias"]},
        str(root / "shard-b.safetensors"),
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "blocks.0.attn.qkv_proj.weight": "shard-a.safetensors",
                    "blocks.0.attn.qkv_proj.scales": "shard-a.safetensors",
                    "time_embedder.proj_in.bias": "shard-b.safetensors",
                }
            }
        )
    )
    return tensors


def test_plan_and_exact_selected_values() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tensors = write_index(root)
        keys = ["blocks.0.attn.qkv_proj.weight", "time_embedder.proj_in.bias"]
        plan = build_tensor_plan(root, keys, cwd=root)
        assert_case("plan preserves selected tensor count", plan["tensor_count"] == 2)
        assert_case("plan opens only shards containing requested keys", plan["selected_shard_count"] == 2)
        expected_bytes = tensors[keys[0]].nbytes + tensors[keys[1]].nbytes
        assert_case("plan reports requested bytes from headers", plan["requested_tensor_bytes"] == expected_bytes)
        shard_keys = {item["shard"]: tuple(item["keys"]) for item in plan["shards"]}
        assert_case("unrequested tensor in selected shard is not in key plan", shard_keys["shard-a.safetensors"] == (keys[0],))

        loaded = load_selected_tensors(root, keys)
        assert_case("selected key order is preserved", list(loaded) == keys)
        assert_case("uint32 quantized payload is exact", np.array_equal(loaded[keys[0]], tensors[keys[0]]))
        assert_case("float payload is exact", np.array_equal(loaded[keys[1]], tensors[keys[1]]))
        fingerprints = fingerprint_tensors(loaded)
        assert_case("fingerprint records exact byte counts", fingerprints[keys[0]]["nbytes"] == tensors[keys[0]].nbytes)

        loaded_mlx = load_selected_mlx_tensors(root, keys)
        assert_case("MLX selected key order is preserved", list(loaded_mlx) == keys)
        assert_case("MLX uint32 payload is exact", np.array_equal(np.asarray(loaded_mlx[keys[0]]), tensors[keys[0]]))
        assert_case("MLX float payload is exact", np.array_equal(np.asarray(loaded_mlx[keys[1]]), tensors[keys[1]]))


def test_unknown_key_is_explicit() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_index(root)
        try:
            load_selected_tensors(root, ["missing.tensor"])
        except KeyError as exc:
            assert_case("unknown tensor key raises KeyError", "missing.tensor" in str(exc))
        else:
            raise AssertionError("unknown tensor key did not raise")


def test_mlx_bf16_selected_tensor_preserves_bits() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tensor = mx.array(np.asarray([[1.0, -2.5, 3.25]], dtype=np.float32)).astype(mx.bfloat16)
        bits = np.asarray(tensor.view(mx.uint16)).astype("<u2", copy=False)
        payload = bits.tobytes(order="C")
        header = {
            "bf16.weight": {
                "dtype": "BF16",
                "shape": list(bits.shape),
                "data_offsets": [0, len(payload)],
            }
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with (root / "bf16.safetensors").open("wb") as handle:
            handle.write(len(header_bytes).to_bytes(8, "little"))
            handle.write(header_bytes)
            handle.write(payload)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"bf16.weight": "bf16.safetensors"}})
        )
        loaded = load_selected_mlx_tensors(root, ["bf16.weight"])["bf16.weight"]
        mx.eval(loaded)
        assert_case("BF16 MLX dtype is preserved", loaded.dtype == mx.bfloat16)
        assert_case(
            "BF16 raw bits are preserved",
            np.array_equal(np.asarray(loaded.view(mx.uint16)), bits),
        )


def tiny_dit_config() -> DiTConfig:
    hidden = 8
    return DiTConfig(
        hidden_size=hidden,
        num_layers=2,
        token_refiner_num_layers=0,
        num_attention_heads=2,
        attention_head_dim=4,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=4,
        patch_size=(1, 1, 1),
        text_dim=8,
        timestep_input_dim=4,
        time_embed_hidden_size=8,
        time_embed_dim=4,
        adaln_out_features=3 * 6 * hidden,
        final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=1,
    )


def _array_for(value: mx.array, offset: int) -> np.ndarray:
    shape = tuple(int(dim) for dim in value.shape)
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if value.dtype == mx.uint32:
        return (np.arange(size, dtype=np.uint32).reshape(shape) + np.uint32(offset))
    dtype = np.float16 if value.dtype == mx.float16 else np.float32
    data = (np.arange(size, dtype=np.float32).reshape(shape) + float(offset)) / 17.0
    return data.astype(dtype)


def write_streaming_checkpoint(root: Path) -> None:
    config = tiny_dit_config()
    quant = QuantConfig(bits=4, group_size=4, quantize_adaln=True, adaln_bits=8)
    (root / "config.json").write_text(json.dumps(asdict(config)))
    (root / "quant_config.json").write_text(json.dumps(asdict(quant)))

    slot = _BlockSlot(config, quant)
    params = tree_flatten(slot.parameters())
    weight_map: dict[str, str] = {}
    for block_index in range(config.num_layers):
        shard_name = f"block-{block_index}.safetensors"
        shard_tensors: dict[str, np.ndarray] = {}
        for param_index, (target_key, value) in enumerate(params):
            source_key = f"blocks.{block_index}." + target_key[len("blocks.0.") :]
            shard_tensors[source_key] = _array_for(value, offset=1000 * block_index + param_index)
            weight_map[source_key] = shard_name
        save_file(shard_tensors, str(root / shard_name))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def write_mixed_streaming_checkpoint(root: Path) -> dict[str, np.ndarray]:
    config = tiny_dit_config()
    quant = QuantConfig(bits=4, group_size=4, quantize_adaln=True, adaln_bits=8)
    (root / "config.json").write_text(json.dumps(asdict(config)))
    (root / "quant_config.json").write_text(json.dumps(asdict(quant)))

    slot = _BlockSlot(config, quant)
    params = tree_flatten(slot.parameters())
    weight_map: dict[str, str] = {}
    shard_tensors: dict[str, np.ndarray] = {}
    for block_index in range(config.num_layers):
        for param_index, (target_key, value) in enumerate(params):
            source_key = f"blocks.{block_index}." + target_key[len("blocks.0.") :]
            shard_tensors[source_key] = _array_for(value, offset=1000 * block_index + param_index)
            weight_map[source_key] = "mixed-blocks.safetensors"
    save_file(shard_tensors, str(root / "mixed-blocks.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return shard_tensors


def _slot_fingerprints(provider: QuantizedBlockProvider, *, include_adaln: bool = True, adaln_only: bool = False) -> dict[str, dict[str, object]]:
    flat = tree_flatten(provider.slot.parameters())
    mx.eval(*(value for _, value in flat))
    if adaln_only:
        flat = [(key, value) for key, value in flat if ".adaln_proj." in key]
    elif not include_adaln:
        flat = [(key, value) for key, value in flat if ".adaln_proj." not in key]
    out: dict[str, dict[str, object]] = {}
    for key, value in flat:
        array = np.ascontiguousarray(np.asarray(value))
        out[key] = {
            "shape": [int(dim) for dim in array.shape],
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return out


def test_streaming_provider_rejects_archived_selective_mode() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_streaming_checkpoint(root)
        reference = QuantizedBlockProvider(root)
        assert_case("provider default remains mx.load", reference.block_load_mode == "mlx")

        try:
            QuantizedBlockProvider(root, block_load_mode="selective_safetensors")
        except ValueError as exc:
            assert_case("archived selective provider is rejected at construction", "selective_safetensors" in str(exc))
        else:
            raise AssertionError("archived selective provider remained constructible")

        try:
            reference.set_block_load_mode("selective_safetensors")
        except ValueError as exc:
            assert_case("archived selective provider is rejected at mode switch", "selective_safetensors" in str(exc))
        else:
            raise AssertionError("archived selective provider remained switchable")

        for kwargs in ({}, {"include_adaln": False}, {"adaln_only": True}):
            provider = QuantizedBlockProvider(root, block_load_mode="mlx")
            provider.load_block(1, **kwargs)
            assert_case(
                f"provider mx.load logical bytes positive for {kwargs or {'include_adaln': True}}",
                provider.logical_bytes_loaded > 0,
            )


def test_block_local_reshard_layout_matches_mx_load_and_uses_relative_index() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source"
        candidate_root = Path(directory) / "block-local"
        source.mkdir()
        write_mixed_streaming_checkpoint(source)

        result = create_block_local_layout(source, candidate_root, block_indices=[0, 1], include_static=False)
        index = json.loads((candidate_root / "model.safetensors.index.json").read_text())
        shard_names = set(index["weight_map"].values())
        assert_case("block-local index values are relative", all(not Path(name).is_absolute() for name in shard_names))
        assert_case("block-local layout splits core and AdaLN shards", any(name.endswith("-core.safetensors") for name in shard_names) and any(name.endswith("-adaln.safetensors") for name in shard_names))
        assert_case("manifest records copied tensor count", result.tensor_count == len(index["weight_map"]))

        for kwargs in ({}, {"include_adaln": False}, {"adaln_only": True}):
            reference = QuantizedBlockProvider(source, block_load_mode="mlx")
            candidate = QuantizedBlockProvider(candidate_root, block_load_mode="mlx")
            reference.load_block(1, **kwargs)
            candidate.load_block(1, **kwargs)
            assert_case(
                f"block-local mx.load matches original layout for {kwargs or {'include_adaln': True}}",
                _slot_fingerprints(reference, **kwargs) == _slot_fingerprints(candidate, **kwargs),
            )
            assert_case(
                f"block-local logical byte accounting matches for {kwargs or {'include_adaln': True}}",
                reference.logical_bytes_loaded == candidate.logical_bytes_loaded,
            )


def test_block_local_reshard_fails_on_missing_or_corrupt_source_tensor() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "missing"
        candidate_root = Path(directory) / "block-local-missing"
        source.mkdir()
        tensors = write_mixed_streaming_checkpoint(source)
        first_key = next(iter(tensors))
        save_file({key: value for key, value in tensors.items() if key != first_key}, str(source / "mixed-blocks.safetensors"))
        try:
            create_block_local_layout(source, candidate_root, block_indices=[0], include_static=False)
        except KeyError as exc:
            assert_case("missing indexed source tensor raises KeyError", first_key in str(exc))
        else:
            raise AssertionError("missing indexed source tensor did not raise")

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "corrupt"
        candidate_root = Path(directory) / "block-local-corrupt"
        source.mkdir()
        write_mixed_streaming_checkpoint(source)
        (source / "mixed-blocks.safetensors").write_bytes(b"not-a-valid-safetensors")
        try:
            create_block_local_layout(source, candidate_root, block_indices=[0], include_static=False)
        except ValueError as exc:
            assert_case("corrupt source shard raises ValueError", "safetensors header" in str(exc) or "too small" in str(exc))
        else:
            raise AssertionError("corrupt source shard did not raise")


def main() -> int:
    test_plan_and_exact_selected_values()
    test_unknown_key_is_explicit()
    test_mlx_bf16_selected_tensor_preserves_bits()
    test_streaming_provider_rejects_archived_selective_mode()
    test_block_local_reshard_layout_matches_mx_load_and_uses_relative_index()
    test_block_local_reshard_fails_on_missing_or_corrupt_source_tensor()
    print("selective loading focused tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
