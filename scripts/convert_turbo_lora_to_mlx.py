#!/usr/bin/env python3
"""Validate and canonicalize a native MiniMax-H3 Turbo LoRA for MLX streaming."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

MAIN_RE = re.compile(
    r"^blocks\.(\d+)\."
    r"(attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2)|adaln_proj\.linear)\."
    r"lora_([AB])\.weight$"
)
REFINER_RE = re.compile(
    r"^token_refiner\.blocks\.(\d+)\."
    r"(attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\."
    r"lora_([AB])\.weight$"
)
FINAL_RE = re.compile(r"^final_layer\.adaln_proj\.linear\.lora_([AB])\.weight$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_shapes(config: dict, target: str, rank: int) -> dict[str, tuple[int, int]]:
    hidden = int(config["hidden_size"])
    inner = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    ffn = int(config["ffn_hidden_size"])
    time_dim = int(config["time_embed_dim"])
    shapes = {
        "attn.qkv_proj": {"A": (rank, hidden), "B": (3 * inner, rank)},
        "attn.out_proj": {"A": (rank, inner), "B": (hidden, rank)},
        "mlp.fc1": {"A": (rank, hidden), "B": (2 * ffn, rank)},
        "mlp.fc2": {"A": (rank, ffn), "B": (hidden, rank)},
        "adaln_proj.linear": {
            "A": (rank, time_dim),
            "B": (int(config["adaln_out_features"]), rank),
        },
        "final_layer.adaln_proj.linear": {
            "A": (rank, time_dim),
            "B": (int(config["final_adaln_out_features"]), rank),
        },
    }
    return shapes[target]


def inspect_source(source: Path, config: dict) -> tuple[list[dict], dict[str, str]]:
    records: list[dict] = []
    source_metadata: dict[str, str] = {}
    identities: dict[tuple[str, int, str, str], dict] = {}
    with safe_open(source, framework="np") as handle:
        source_metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            shape = tuple(int(v) for v in handle.get_slice(key).get_shape())
            dtype = str(handle.get_slice(key).get_dtype())
            match = MAIN_RE.fullmatch(key)
            if match:
                index, target, side = match.groups()
                family, logical_index = "block", int(index)
            else:
                match = REFINER_RE.fullmatch(key)
                if match:
                    index, target, side = match.groups()
                    family, logical_index = "refiner", int(index)
                else:
                    match = FINAL_RE.fullmatch(key)
                    if match:
                        (side,) = match.groups()
                        family, logical_index, target = (
                            "final",
                            0,
                            "final_layer.adaln_proj.linear",
                        )
                    else:
                        raise ValueError(f"unsupported Turbo LoRA tensor: {key}")
            identity = (family, logical_index, target, side)
            if identity in identities:
                raise ValueError(f"duplicate Turbo LoRA tensor: {identity}")
            record = {
                "source_key": key,
                "mlx_key": key,
                "family": family,
                "index": logical_index,
                "target": target,
                "side": side,
                "shape": list(shape),
                "dtype": dtype,
            }
            identities[identity] = record
            records.append(record)

    required: list[tuple[str, int, str]] = []
    for index in range(int(config["num_layers"])):
        for target in (
            "attn.qkv_proj",
            "attn.out_proj",
            "mlp.fc1",
            "mlp.fc2",
            "adaln_proj.linear",
        ):
            required.append(("block", index, target))
    for index in range(int(config["token_refiner_num_layers"])):
        for target in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            required.append(("refiner", index, target))
    required.append(("final", 0, "final_layer.adaln_proj.linear"))

    for family, index, target in required:
        a = identities.get((family, index, target, "A"))
        b = identities.get((family, index, target, "B"))
        if a is None or b is None:
            raise KeyError(f"missing Turbo LoRA pair: {(family, index, target)}")
        if a["dtype"] != "BF16" or b["dtype"] != "BF16":
            raise TypeError(f"Turbo LoRA pair is not BF16: {(family, index, target)}")
        rank = int(a["shape"][0])
        if int(b["shape"][-1]) != rank:
            raise ValueError(f"rank mismatch: {(family, index, target)}")
        expected = expected_shapes(config, target, rank)
        if tuple(a["shape"]) != expected["A"] or tuple(b["shape"]) != expected["B"]:
            raise ValueError(
                f"shape mismatch for {(family, index, target)}: "
                f"A{tuple(a['shape'])}/B{tuple(b['shape'])}, expected "
                f"A{expected['A']}/B{expected['B']}"
            )
        a["rank"] = b["rank"] = rank
        a["alpha"] = b["alpha"] = rank
        a["runtime_scale"] = b["runtime_scale"] = 1.0

    if len(identities) != len(required) * 2:
        raise ValueError(
            f"source has {len(identities)} tensors but contract consumes {len(required) * 2}"
        )
    return sorted(records, key=lambda row: row["source_key"]), source_metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--replace-generated",
        action="store_true",
        help="replace only files produced by this converter in an existing output directory",
    )
    args = parser.parse_args()

    source = Path(args.source)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output = output_dir / "turbo_lora.safetensors"
    manifest_path = output_dir / "conversion_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        allowed = {"turbo_lora.safetensors", "conversion_manifest.json", "adapter.safetensors.index.json", "shards"}
        present = {path.name for path in output_dir.iterdir()}
        if not args.replace_generated or not present <= allowed:
            parser.error(f"refusing to overwrite non-generated output directory: {output_dir}")
    config = json.loads(config_path.read_text())
    mapping, source_metadata = inspect_source(source, config)

    arrays = mx.load(str(source))
    if set(arrays) != {row["source_key"] for row in mapping}:
        raise RuntimeError("loaded tensor keys differ from audited source keys")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "minimax-h3-mlx-native-turbo-lora-v1",
        "base_model": "MiniMax-H3",
        "application": "W_eff = W + lora_B @ lora_A",
        "alpha_policy": "alpha=rank",
        "runtime_scale": "1.0",
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "source_sha256": sha256(source),
    }
    mx.save_safetensors(str(output), arrays, metadata=metadata)

    # The monolithic adapter is the portable archival artifact. The runtime index below points to
    # one small file per main block/refiner/final layer so 24 GB Macs never reload or retain all
    # 744 MB merely to apply one streamed block.
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    shard_records: list[dict] = []
    grouped_keys: dict[tuple[str, int], list[str]] = {}
    by_key = {row["source_key"]: row for row in mapping}
    for key in sorted(arrays):
        row = by_key[key]
        grouped_keys.setdefault((str(row["family"]), int(row["index"])), []).append(key)
    for (family, index), keys in sorted(grouped_keys.items()):
        name = f"adapter-{family}-{index:03d}.safetensors"
        relative = f"shards/{name}"
        shard = {key: arrays[key] for key in keys}
        mx.save_safetensors(str(shards_dir / name), shard, metadata=metadata)
        shard_records.append({
            "family": family,
            "index": index,
            "file": relative,
            "tensor_count": len(keys),
            "bytes": (shards_dir / name).stat().st_size,
            "sha256": sha256(shards_dir / name),
        })
        for key in keys:
            weight_map[key] = relative
    index_path = output_dir / "adapter.safetensors.index.json"
    index_path.write_text(json.dumps({
        "metadata": {
            "format": metadata["format"],
            "total_size": sum(int(value.nbytes) for value in arrays.values()),
            "source_sha256": metadata["source_sha256"],
        },
        "weight_map": weight_map,
    }, indent=2) + "\n")
    del arrays

    converted = mx.load(str(output))
    source_arrays = mx.load(str(source))
    mismatched = []
    for key in sorted(source_arrays):
        if key not in converted or not bool(mx.array_equal(source_arrays[key], converted[key]).item()):
            mismatched.append(key)
    if mismatched:
        raise RuntimeError(f"converted tensors differ from source: {mismatched[:8]}")
    del source_arrays, converted
    mx.clear_cache()

    ranks: dict[str, int] = {}
    target_pairs: dict[str, int] = {}
    for record in mapping:
        ranks[str(record["rank"])] = ranks.get(str(record["rank"]), 0) + 1
        target_pairs[record["target"]] = target_pairs.get(record["target"], 0) + (record["side"] == "A")
    manifest = {
        "schema_version": 1,
        "format": metadata["format"],
        "source": {
            "repo": args.source_repo,
            "revision": args.source_revision,
            "file": source.name,
            "bytes": source.stat().st_size,
            "sha256": metadata["source_sha256"],
            "metadata": source_metadata,
            "license": "apache-2.0 (source model card)",
        },
        "output": {
            "file": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "tensor_count": len(mapping),
            "dtype": "BF16",
            "streaming_index": index_path.name,
            "streaming_shard_count": len(shard_records),
            "streaming_shards": shard_records,
        },
        "contract": {
            "pair_count": len(mapping) // 2,
            "rank_tensor_counts": ranks,
            "target_pair_counts": target_pairs,
            "alpha_policy": "alpha=rank",
            "runtime_multiplier": 1.0,
            "tensor_transform": "identity; source already uses canonical fused MLX module paths",
            "all_source_tensors_consumed": True,
            "post_write_tensor_values_exact": True,
        },
        "mapping": mapping,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"PASS: {len(mapping)} BF16 tensors / {len(mapping) // 2} pairs; "
        f"output={output}; sha256={manifest['output']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
