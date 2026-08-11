#!/usr/bin/env python3
"""Audit an MLX affine-quantized artifact without loading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

FLOAT_DTYPES = {"BF16", "F16", "F32"}
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "U32": 4,
}


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise ValueError(f"invalid tensor shape {shape!r}")
        total *= dim
    return total


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: truncated safetensors length")
        header_length = struct.unpack("<Q", raw_length)[0]
        payload_bytes = path.stat().st_size - 8 - header_length
        if payload_bytes < 0:
            raise ValueError(f"{path}: header extends beyond end of file")
        header = handle.read(header_length)
        if len(header) != header_length:
            raise ValueError(f"{path}: truncated safetensors header")
    try:
        value = json.loads(header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid safetensors header") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: safetensors header is not an object")

    ranges: list[tuple[int, int, str]] = []
    for key, spec in value.items():
        if key == "__metadata__":
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: tensor {key} has an invalid header record")
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if dtype not in DTYPE_BYTES or not isinstance(shape, list):
            raise ValueError(f"{path}: tensor {key} has unsupported dtype/shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise ValueError(f"{path}: tensor {key} has invalid data_offsets")
        start, end = offsets
        expected_bytes = _numel(shape) * DTYPE_BYTES[dtype]
        if start < 0 or end < start or end > payload_bytes:
            raise ValueError(f"{path}: tensor {key} has out-of-range data_offsets")
        if end - start != expected_bytes:
            raise ValueError(
                f"{path}: tensor {key} payload is {end - start} bytes, expected {expected_bytes}"
            )
        ranges.append((start, end, key))
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"{path}: tensors {previous[2]} and {current[2]} overlap")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor_headers(model_dir: Path) -> tuple[dict[str, dict], list[Path]]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}.")
    with index_path.open() as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: missing weight_map")

    files = [model_dir / name for name in sorted(set(weight_map.values()))]
    missing_files = [str(path) for path in files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing shards: {missing_files[:4]}")

    tensors: dict[str, dict] = {}
    owners: dict[str, str] = {}
    for path in files:
        for key, spec in read_safetensors_header(path).items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"Duplicate tensor {key} in {owners[key]} and {path.name}.")
            tensors[key] = spec
            owners[key] = path.name

    expected = set(weight_map)
    actual = set(tensors)
    if expected != actual:
        raise ValueError(
            f"Index/header mismatch: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected."
        )
    wrong_owner = [key for key, owner in owners.items() if weight_map[key] != owner]
    if wrong_owner:
        raise ValueError(f"{len(wrong_owner)} tensors are stored in the wrong indexed shard.")
    return tensors, files


def audit(model_dir: Path, require_int8: bool = True, include_hashes: bool = False) -> dict:
    config_path = model_dir / "config.json"
    recipe_path = model_dir / "quant_config.json"
    if not config_path.is_file() or not recipe_path.is_file():
        raise FileNotFoundError(f"{model_dir}: config.json and quant_config.json are required")
    with recipe_path.open() as handle:
        recipe = json.load(handle)
    bits = recipe.get("bits")
    group_size = recipe.get("group_size")
    mode = recipe.get("mode", "affine")
    if require_int8 and bits != 8:
        raise ValueError(f"Expected INT8 recipe, found bits={bits!r}.")
    if not isinstance(bits, int) or not isinstance(group_size, int):
        raise ValueError("quant_config.json must contain integer bits and group_size.")
    if mode != "affine":
        raise ValueError(f"Expected affine quantization with biases, found mode={mode!r}.")

    tensors, shards = load_tensor_headers(model_dir)
    layers = sorted(key[: -len(".scales")] for key in tensors if key.endswith(".scales"))
    if not layers:
        raise ValueError("No quantized layers (no .scales tensors) found.")

    records = []
    errors = []
    for layer in layers:
        names = {
            "weight": f"{layer}.weight",
            "scales": f"{layer}.scales",
            "biases": f"{layer}.biases",
            "bias": f"{layer}.bias",
        }
        missing = [name for name in ("weight", "scales", "biases") if names[name] not in tensors]
        if missing:
            errors.append(f"{layer}: missing {', '.join(missing)}")
            continue

        weight = tensors[names["weight"]]
        scales = tensors[names["scales"]]
        biases = tensors[names["biases"]]
        weight_shape = tuple(weight["shape"])
        scales_shape = tuple(scales["shape"])
        packed_bits = weight_shape[-1] * 32
        if weight["dtype"] != "U32":
            errors.append(f"{layer}: packed weight dtype is {weight['dtype']}, expected U32")
        if packed_bits % bits:
            errors.append(f"{layer}: packed width {weight_shape[-1]} is incompatible with {bits} bits")
            continue
        input_dims = packed_bits // bits
        expected_aux = weight_shape[:-1] + (input_dims // group_size,)
        if input_dims % group_size:
            errors.append(f"{layer}: input width {input_dims} is not divisible by group {group_size}")
        if scales_shape != expected_aux:
            errors.append(f"{layer}: scales shape {scales_shape}, expected {expected_aux}")
        if tuple(biases["shape"]) != expected_aux:
            errors.append(f"{layer}: affine biases shape {biases['shape']}, expected {expected_aux}")
        if scales["dtype"] not in FLOAT_DTYPES:
            errors.append(f"{layer}: scales dtype is {scales['dtype']}")
        if biases["dtype"] != scales["dtype"]:
            errors.append(
                f"{layer}: affine biases dtype {biases['dtype']} != scales dtype {scales['dtype']}"
            )
        linear_bias = tensors.get(names["bias"])
        records.append(
            {
                "layer": layer,
                "bits": bits,
                "group_size": group_size,
                "packed_weight": {"dtype": weight["dtype"], "shape": list(weight_shape)},
                "scales": {"dtype": scales["dtype"], "shape": list(scales_shape)},
                "affine_biases": {"dtype": biases["dtype"], "shape": list(biases["shape"])},
                "linear_bias": (
                    None
                    if linear_bias is None
                    else {"dtype": linear_bias["dtype"], "shape": linear_bias["shape"]}
                ),
            }
        )

    bf16_layers = recipe.get("bf16_layers", [])
    if not isinstance(bf16_layers, list) or not all(isinstance(path, str) for path in bf16_layers):
        errors.append("quant_config bf16_layers must be a list of module paths")
        bf16_layers = []
    bf16_records = []
    for layer in sorted(set(bf16_layers)):
        weight_name = f"{layer}.weight"
        if f"{layer}.scales" in tensors or f"{layer}.biases" in tensors:
            errors.append(f"{layer}: BF16 exception still has quantized scales/biases")
            continue
        weight = tensors.get(weight_name)
        if weight is None:
            errors.append(f"{layer}: BF16 exception is missing {weight_name}")
            continue
        if weight["dtype"] != "BF16":
            errors.append(f"{layer}: BF16 exception weight dtype is {weight['dtype']}, expected BF16")
        bf16_records.append(
            {
                "layer": layer,
                "weight": {"dtype": weight["dtype"], "shape": weight["shape"]},
                "parameters": _numel(weight["shape"]),
            }
        )

    expected_counts = recipe.get("quantized_layers")
    if isinstance(expected_counts, dict):
        expected_total = sum(int(value) for value in expected_counts.values())
        if expected_total != len(layers):
            errors.append(
                f"quant_config records {expected_total} layers but headers contain {len(layers)}"
            )
    if errors:
        raise ValueError("Quantization audit failed:\n  " + "\n  ".join(errors[:32]))

    files = [config_path, recipe_path, model_dir / "model.safetensors.index.json", *shards]
    return {
        "artifact": str(model_dir.resolve()),
        "recipe": recipe,
        "format": {
            "mode": mode,
            "bits": bits,
            "group_size": group_size,
            "packed_weight_dtype": "U32",
            "scales_present": True,
            "affine_biases_present": True,
        },
        "quantized_layer_count": len(records),
        "bf16_exception_count": len(bf16_records),
        "bf16_exceptions": bf16_records,
        "tensor_count": len(tensors),
        "bytes": sum(os.path.getsize(path) for path in files),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                **({"sha256": sha256(path)} if include_hashes else {}),
            }
            for path in files
        ],
        "layers": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--out")
    parser.add_argument("--hash", action="store_true", help="SHA-256 every artifact file")
    parser.add_argument("--allow-non-int8", action="store_true")
    args = parser.parse_args()

    report = audit(Path(args.artifact), not args.allow_non_int8, args.hash)
    output = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(output + "\n")
    print(
        f"PASS: {report['quantized_layer_count']} affine INT{report['format']['bits']} layers, "
        f"group {report['format']['group_size']}, packed U32 + scales + biases"
    )
    print(f"artifact: {report['artifact']}")
    if args.out:
        print(f"report: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
