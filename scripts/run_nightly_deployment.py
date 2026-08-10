#!/usr/bin/env python3
"""Safely orchestrate the local MiniMax-H3 nightly deployment sequence.

The runner is intentionally repo-local and conservative:

1. run the read-only asset preflight first;
2. stop before quantization or generation unless preflight exits 0;
3. verify the selected project-local/explicit ffmpeg and ffprobe route;
4. quantize the official FL2VA text encoder;
5. verify real prompt encode/release and streamed DiT/AdaLN/Turbo/projection access;
6. run the 320x192 -> 512x288 -> 960x544 generation ladder with an explicit
   sigma/NFE contract (default: 5 sigma points, 4 denoiser evaluations);
7. validate each MP4 with ffprobe and require both video and audio streams;
8. decode the selected audio stream and require finite, non-silent samples.

Every command is executed through /usr/bin/time -l when available and gets a
JSONL event with the command, exit code, wall time, peak RSS when reported,
MLX memory when a child process reports it, and relevant artifact paths.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.media_tools import (  # noqa: E402
    PROJECT_LOCAL_MEDIA_TOOL_DIRS,
    media_tool_candidate_paths as _common_media_tool_candidate_paths,
    resolve_media_tool as _common_resolve_media_tool,
)

DEFAULT_PROMPT = "a red fox leaps over a mossy log, natural motion, synchronized ambient sound"
DEFAULT_LADDER = "320x192,512x288,960x544"
DEFAULT_AUDIO_ACTIVITY_SAMPLE_RATE_HZ = 16_000
DEFAULT_AUDIO_ACTIVITY_NONZERO_EPSILON = 1e-8
DEFAULT_AUDIO_ACTIVITY_MIN_RMS = 1e-6
DEFAULT_AUDIO_ACTIVITY_MIN_PEAK = 1e-5
DEFAULT_AUDIO_ACTIVITY_MIN_NONZERO_SAMPLES = 16
SCHEMA_VERSION = 1
ASSET_FAILURE = 2
VALIDATION_FAILURE = 3
USAGE_ERROR = 64
_TIME_RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\b", re.MULTILINE)
_ARGUS_MLX_MEMORY_RE = re.compile(
    r"ARGUS_MLX_MEMORY\s+"
    r"peak_bytes=(?P<peak>\d+|None)\s+"
    r"active_bytes=(?P<active>\d+|None)\s+"
    r"cache_bytes=(?P<cache>\d+|None)"
)
_MLX_MEMORY_RES = [
    re.compile(
        r"mlx[^\n]*(?:peak|max(?:imum)?)\s+(?:memory|allocated)[^0-9\n]*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>bytes?|[kmgt]i?b|[kmgt]b)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:peak|max(?:imum)?)\s+mlx[^\n]*memory[^0-9\n]*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>bytes?|[kmgt]i?b|[kmgt]b)?",
        re.IGNORECASE,
    ),
]
_OOM_MARKERS = (
    "out of memory",
    "cannot allocate memory",
    "memoryerror",
    "mlx_error",
    "killed: 9",
    "signal 9",
)

Postprocess = Callable[[int | None, Path, Path], dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)


def _repo_rel(path: str | Path, *, root: Path = ROOT) -> str:
    """Render repo-local paths without resolving .venv/model symlinks."""
    path = Path(path)
    root_abs = root.absolute()
    path_abs = path if path.is_absolute() else root_abs / path
    try:
        return str(path_abs.relative_to(root_abs))
    except ValueError:
        return str(path_abs)


def _resolve_repo_path(path: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _cmd_path(path: str | Path, *, root: Path = ROOT) -> str:
    return _repo_rel(_resolve_repo_path(path, root=root), root=root)


def _parse_ladder(spec: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for raw_item in spec.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"ladder entry must look like WIDTHxHEIGHT, got {raw_item!r}")
        width_s, height_s = item.split("x", 1)
        width, height = int(width_s), int(height_s)
        if width <= 0 or height <= 0:
            raise ValueError(f"ladder dimensions must be positive, got {raw_item!r}")
        if width % 32 or height % 32:
            raise ValueError(f"ladder dimensions must be multiples of 32, got {raw_item!r}")
        result.append((width, height))
    if not result:
        raise ValueError("ladder cannot be empty")
    return result


def _duration_token(seconds: float) -> str:
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{str(seconds).replace('.', 'p')}s"


def _generation_schedule_contract(sigma_points: int) -> dict[str, Any]:
    """Durable runner evidence for the MiniMax-H3 --steps -> sigma/NFE mapping."""
    if sigma_points < 2:
        raise ValueError(f"--steps must request at least 2 sigma points, got {sigma_points}.")
    denoiser_evaluations = sigma_points - 1
    return {
        "cli_steps_argument": sigma_points,
        "sigma_points": sigma_points,
        "denoiser_evaluations": denoiser_evaluations,
        "nfe": denoiser_evaluations,
        "generate_py_mapping": "scripts/generate.py forwards --steps as pipe(..., num_inference_steps=args.steps)",
        "scheduler_mapping": "MiniMaxH3Scheduler.set_timesteps(N) builds N sigmas including terminal 0 and exposes timesteps=sigmas[:-1]",
        "pipeline_mapping": "MiniMaxH3Pipeline.__call__ iterates video_sched.timesteps and calls self.dit once per timestep",
    }


def _audio_activity_contract(
    *,
    sample_rate_hz: int,
    min_rms: float,
    min_peak: float,
    min_nonzero_samples: int,
    nonzero_epsilon: float = DEFAULT_AUDIO_ACTIVITY_NONZERO_EPSILON,
) -> dict[str, Any]:
    return {
        "decode_route": "ffmpeg -map 0:a:0 -vn -ac 1 -ar SAMPLE_RATE -f f32le pipe:1",
        "sample_format": "mono f32le",
        "sample_rate_hz": sample_rate_hz,
        "nonzero_epsilon": nonzero_epsilon,
        "min_rms": min_rms,
        "min_peak": min_peak,
        "min_nonzero_samples": min_nonzero_samples,
    }


def _audio_activity_stats(
    decoded: bytes,
    *,
    sample_rate_hz: int,
    min_rms: float,
    min_peak: float,
    min_nonzero_samples: int,
    nonzero_epsilon: float = DEFAULT_AUDIO_ACTIVITY_NONZERO_EPSILON,
) -> dict[str, Any]:
    byte_count = len(decoded)
    result: dict[str, Any] = {
        "audio_activity_contract": _audio_activity_contract(
            sample_rate_hz=sample_rate_hz,
            min_rms=min_rms,
            min_peak=min_peak,
            min_nonzero_samples=min_nonzero_samples,
            nonzero_epsilon=nonzero_epsilon,
        ),
        "decoded_audio_bytes": byte_count,
        "postcheck_ok": False,
    }
    if byte_count == 0:
        result.update(
            {
                "decoded_audio_sample_count": 0,
                "validation_error": "ffmpeg decoded no audio samples",
            }
        )
        return result
    if byte_count % 4:
        result.update(
            {
                "decoded_audio_sample_count": byte_count // 4,
                "validation_error": "decoded f32le byte count is not divisible by 4",
            }
        )
        return result

    sample_count = byte_count // 4
    finite_count = 0
    nonzero_count = 0
    max_abs = 0.0
    sum_squares = 0.0
    for (sample,) in struct.iter_unpack("<f", decoded):
        if not math.isfinite(sample):
            continue
        finite_count += 1
        abs_sample = abs(float(sample))
        if abs_sample > nonzero_epsilon:
            nonzero_count += 1
        if abs_sample > max_abs:
            max_abs = abs_sample
        sum_squares += abs_sample * abs_sample

    rms = math.sqrt(sum_squares / finite_count) if finite_count else 0.0
    result.update(
        {
            "decoded_audio_sample_count": sample_count,
            "decoded_audio_duration_seconds": sample_count / sample_rate_hz,
            "finite_sample_count": finite_count,
            "nonfinite_sample_count": sample_count - finite_count,
            "nonzero_sample_count": nonzero_count,
            "max_abs_sample": max_abs,
            "rms": rms,
        }
    )
    if finite_count != sample_count:
        result["validation_error"] = "decoded audio contains non-finite samples"
    elif nonzero_count < min_nonzero_samples or max_abs < min_peak or rms < min_rms:
        result["validation_error"] = "decoded audio is silent or below activity thresholds"
    else:
        result["postcheck_ok"] = True
    return result


def _sanitize_run_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return token or "run"


def _issue(code: str, message: str, path: str | Path, *, root: Path = ROOT) -> dict[str, str]:
    return {
        "severity": "error",
        "code": code,
        "path": _repo_rel(path, root=root) if isinstance(path, Path) else path,
        "message": message,
    }


def _read_json_object(path: Path, issues: list[dict[str, str]], *, code: str, root: Path) -> dict[str, Any] | None:
    if not path.is_file():
        issues.append(_issue("missing_required_file", f"required file is missing: {path.name}", path, root=root))
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        issues.append(_issue(code, f"JSON is not parseable: {type(exc).__name__}: {exc}", path, root=root))
        return None
    if not isinstance(payload, dict):
        issues.append(_issue(code, "JSON root must be an object", path, root=root))
        return None
    return payload


def _validate_quantized_text_encoder_dir(
    path: str | Path,
    *,
    bits: int,
    group_size: int,
    num_layers: int,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Header-only validation for a derived MLX text-encoder directory.

    This deliberately validates only the derived output: config/index/quant
    metadata, declared shard presence, non-empty safetensors files, and readable
    safetensors headers. It never opens or mutates the official source shards.
    """
    resolved = _resolve_repo_path(path, root=root)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "path": _repo_rel(resolved, root=root),
        "exists": resolved.exists() or resolved.is_symlink(),
        "ok": False,
        "status": "invalid",
        "issues": [],
        "expected_quant_config": {
            "bits": bits,
            "group_size": group_size,
            "num_layers": num_layers,
        },
    }
    issues: list[dict[str, str]] = result["issues"]
    if not result["exists"]:
        result["status"] = "missing"
        return result
    if not resolved.is_dir():
        issues.append(_issue("not_a_directory", "text encoder target exists but is not a directory", resolved, root=root))
        return result

    config = _read_json_object(resolved / "config.json", issues, code="invalid_config_json", root=root)
    quant = _read_json_object(resolved / "quant_config.json", issues, code="invalid_quant_config_json", root=root)
    index = _read_json_object(resolved / "model.safetensors.index.json", issues, code="invalid_index_json", root=root)
    if config is not None:
        result["config_model_type"] = config.get("model_type")
    if quant is not None:
        result["quant_config"] = {key: quant.get(key) for key in ("bits", "group_size", "num_layers")}
        for key, expected in result["expected_quant_config"].items():
            if quant.get(key) != expected:
                issues.append(
                    _issue(
                        "quant_config_mismatch",
                        f"quant_config.{key}={quant.get(key)!r} does not match expected {expected!r}",
                        resolved / "quant_config.json",
                        root=root,
                    )
                )

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    shard_names: list[str] = []
    if not isinstance(weight_map, dict) or not weight_map:
        issues.append(
            _issue(
                "invalid_index_weight_map",
                "model.safetensors.index.json must contain a non-empty weight_map object",
                resolved / "model.safetensors.index.json",
                root=root,
            )
        )
    else:
        for key, shard_name in sorted(weight_map.items()):
            if not isinstance(key, str) or not isinstance(shard_name, str):
                issues.append(
                    _issue(
                        "invalid_index_weight_map",
                        "weight_map keys and shard names must be strings",
                        resolved / "model.safetensors.index.json",
                        root=root,
                    )
                )
                continue
            shard_rel = Path(shard_name)
            if shard_rel.is_absolute() or ".." in shard_rel.parts:
                issues.append(
                    _issue(
                        "unsafe_shard_path",
                        f"index shard path must stay inside the text encoder directory: {shard_name!r}",
                        shard_name,
                        root=root,
                    )
                )
                continue
            shard_names.append(shard_name)

    total_size = 0
    tensor_count = 0
    for shard_name in sorted(set(shard_names)):
        shard_path = resolved / shard_name
        if not shard_path.is_file():
            issues.append(_issue("missing_shard_safetensors", "declared safetensors shard is missing", shard_path, root=root))
            continue
        size = shard_path.stat().st_size
        if size <= 0:
            issues.append(_issue("empty_shard_safetensors", "declared safetensors shard is empty", shard_path, root=root))
            continue
        total_size += size
        try:
            from safetensors import safe_open

            with safe_open(str(shard_path), framework="np") as handle:
                tensor_count += len(list(handle.keys()))
        except Exception as exc:
            issues.append(
                _issue(
                    "unreadable_shard_safetensors",
                    f"safetensors header is not readable: {type(exc).__name__}: {exc}",
                    shard_path,
                    root=root,
                )
            )

    result.update(
        {
            "shard_count": len(set(shard_names)),
            "tensor_count": tensor_count,
            "total_shard_bytes": total_size,
            "ok": not issues,
            "status": "valid" if not issues else "invalid",
        }
    )
    return result


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_text_tail(path: Path, limit: int = 512_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > limit:
            handle.seek(-limit, os.SEEK_END)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _parse_peak_rss(stderr_text: str) -> int | None:
    matches = _TIME_RSS_RE.findall(stderr_text)
    return int(matches[-1]) if matches else None


def _unit_to_bytes(value: float, unit: str | None) -> int:
    if not unit or unit.lower().startswith("byte"):
        return int(value)
    normalized = unit.lower().replace("ib", "b")
    multipliers = {
        "kb": 1_000,
        "mb": 1_000_000,
        "gb": 1_000_000_000,
        "tb": 1_000_000_000_000,
    }
    return int(value * multipliers.get(normalized, 1))


def _parse_mlx_memory(stdout_text: str, stderr_text: str) -> dict[str, Any]:
    combined = "\n".join([stdout_text, stderr_text])
    argus_match = _ARGUS_MLX_MEMORY_RE.search(combined)
    if argus_match:
        def parse_int(name: str) -> int | None:
            value = argus_match.group(name)
            return None if value == "None" else int(value)

        peak = parse_int("peak")
        return {
            "available": peak is not None,
            "peak_bytes": peak,
            "active_bytes": parse_int("active"),
            "cache_bytes": parse_int("cache"),
            "raw": argus_match.group(0).strip(),
        }
    for regex in _MLX_MEMORY_RES:
        match = regex.search(combined)
        if match:
            value = float(match.group("value"))
            unit = match.group("unit")
            return {
                "available": True,
                "peak_bytes": _unit_to_bytes(value, unit),
                "raw": match.group(0).strip(),
            }
    return {"available": False, "peak_bytes": None, "reason": "not_reported_by_child_process"}


def _classify_failure(exit_code: int | None, stderr_text: str, stdout_text: str) -> str | None:
    if exit_code == 0:
        return None
    text = f"{stderr_text}\n{stdout_text}".lower()
    if exit_code in {-9, 9, 137} or any(marker in text for marker in _OOM_MARKERS):
        return "oom_or_killed"
    if exit_code == 127:
        return "executable_not_found"
    return "command_failed"


def _executable_missing(command: list[str], *, root: Path) -> str | None:
    if not command:
        return "empty command"
    executable = command[0]
    if "/" in executable:
        path = Path(executable)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            return f"executable not found: {executable}"
        if not os.access(path, os.X_OK):
            return f"executable is not executable: {executable}"
        return None
    return None if shutil.which(executable) else f"executable not found on PATH: {executable}"


def _media_tool_candidate_paths(name: str, *, root: Path) -> list[Path]:
    local_dirs = [root / directory for directory in PROJECT_LOCAL_MEDIA_TOOL_DIRS]
    return _common_media_tool_candidate_paths(name, cwd=root, local_bin_dirs=local_dirs)


def _resolve_media_tool(name: str, requested: str | None, *, root: Path) -> dict[str, Any]:
    """Resolve a media tool through the package-wide release-safe resolver."""
    local_dirs = [root / directory for directory in PROJECT_LOCAL_MEDIA_TOOL_DIRS]
    resolution = _common_resolve_media_tool(
        name,
        requested,
        cwd=root,
        local_bin_dirs=local_dirs,
        allow_path=True,
    )
    payload = resolution.to_dict()
    path = payload.get("path")
    route = payload.get("route")
    if route == "explicit_path":
        payload["source"] = "explicit_path"
    elif route == "explicit_name_on_path":
        payload["source"] = "explicit_name_on_path"
    elif route == "local_candidate":
        payload["source"] = "project_local"
    elif route == "path_fallback":
        payload["source"] = "path"
    if isinstance(path, str) and path:
        rendered = _repo_rel(path, root=root)
        if route in {"explicit_path", "local_candidate"}:
            payload["command"] = rendered
    return payload


class DeploymentRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = ROOT
        self.args = args
        self.run_id = args.run_id or _run_id()
        self.run_dir = _resolve_repo_path(args.run_dir, root=self.root) if args.run_dir else self.root / "out" / "nightly_deployment" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = _resolve_repo_path(args.output_dir, root=self.root) if args.output_dir else self.run_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.records: list[dict[str, Any]] = []
        self.ladder = _parse_ladder(args.ladder)
        self.duration = float(args.duration)
        self.generation_schedule_contract = _generation_schedule_contract(int(args.steps))
        self.audio_activity_contract = _audio_activity_contract(
            sample_rate_hz=int(args.audio_activity_sample_rate),
            min_rms=float(args.audio_activity_min_rms),
            min_peak=float(args.audio_activity_min_peak),
            min_nonzero_samples=int(args.audio_activity_min_nonzero_samples),
        )
        self.python = _cmd_path(args.python, root=self.root)
        self.time_bin = str(args.time_bin)
        self.time_available = Path(self.time_bin).exists()
        self.ffmpeg_tool = _resolve_media_tool("ffmpeg", args.ffmpeg, root=self.root)
        self.ffprobe_tool = _resolve_media_tool("ffprobe", args.ffprobe, root=self.root)
        self.ffmpeg = str(self.ffmpeg_tool["command"])
        self.ffprobe = str(self.ffprobe_tool["command"])

    def _append_event(self, record: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_summary(self, *, status: str, exit_code: int, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "exit_code": exit_code,
            "run_id": self.run_id,
            "run_dir": _repo_rel(self.run_dir, root=self.root),
            "events_path": _repo_rel(self.events_path, root=self.root),
            "output_dir": _repo_rel(self.output_dir, root=self.root),
            "generation_schedule_contract": self.generation_schedule_contract,
            "audio_activity_contract": self.audio_activity_contract,
            "media_tools": {"ffmpeg": self.ffmpeg_tool, "ffprobe": self.ffprobe_tool},
            "created_at": self.records[0]["started_at"] if self.records else _utc_now(),
            "updated_at": _utc_now(),
            "steps": self.records,
        }
        if extra:
            payload.update(extra)
        _write_json_atomic(self.summary_path, payload)

    def _run_step(
        self,
        name: str,
        command: list[str],
        *,
        artifact_paths: dict[str, str | Path] | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        postprocess: Postprocess | None = None,
        dry_run: bool = False,
        record_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact_paths = artifact_paths or {}
        stdout_path = stdout_path or self.logs_dir / f"{name}.stdout"
        stderr_path = stderr_path or self.logs_dir / f"{name}.stderr"
        rendered_artifacts = {key: _repo_rel(value, root=self.root) for key, value in artifact_paths.items()}
        base: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "step": name,
            "command": command,
            "shell_command": shlex.join(command),
            "cwd": _repo_rel(self.root, root=self.root),
            "artifact_paths": rendered_artifacts,
            "stdout_path": _repo_rel(stdout_path, root=self.root),
            "stderr_path": _repo_rel(stderr_path, root=self.root),
        }
        if record_fields:
            base.update(record_fields)
        if dry_run:
            record = {
                **base,
                "event": "step_skipped",
                "skipped": True,
                "reason": "dry_run",
                "ok": False,
                "exit_code": None,
                "started_at": _utc_now(),
                "ended_at": _utc_now(),
                "wall_seconds": 0.0,
                "peak_rss_bytes": None,
                "mlx_memory": {"available": False, "peak_bytes": None, "reason": "dry_run"},
            }
            self.records.append(record)
            self._append_event(record)
            return record

        start = {**base, "event": "step_start", "started_at": _utc_now()}
        self._append_event(start)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        exit_code: int | None
        started = time.monotonic()
        missing = _executable_missing(command, root=self.root)
        if missing is not None:
            stdout_path.write_text("")
            stderr_path.write_text(missing + "\n")
            exit_code = 127
        else:
            timed_command = [self.time_bin, "-l", *command] if self.time_available else command
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                completed = subprocess.run(
                    timed_command,
                    cwd=self.root,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                )
            exit_code = completed.returncode
        wall_seconds = time.monotonic() - started

        stderr_text = _read_text_tail(stderr_path)
        stdout_text = _read_text_tail(stdout_path)
        peak_rss = _parse_peak_rss(stderr_text) if self.time_available else None
        record = {
            **base,
            "event": "step_complete",
            "skipped": False,
            "started_at": start["started_at"],
            "ended_at": _utc_now(),
            "wall_seconds": round(wall_seconds, 3),
            "exit_code": exit_code,
            "timed_with": self.time_bin if self.time_available else None,
            "peak_rss_bytes": peak_rss,
            "mlx_memory": _parse_mlx_memory(stdout_text, stderr_text),
            "failure_class": _classify_failure(exit_code, stderr_text, stdout_text),
        }
        if postprocess is not None:
            try:
                extra = postprocess(exit_code, stdout_path, stderr_path)
                record.update(extra)
            except Exception as exc:  # Postprocessing failures are evidence too.
                record["postprocess_error"] = f"{type(exc).__name__}: {exc}"
                record["postcheck_ok"] = False
        record["ok"] = exit_code == 0 and record.get("postcheck_ok", True) is not False
        self.records.append(record)
        self._append_event(record)
        return record

    def _skip_step(
        self,
        name: str,
        command: list[str],
        *,
        reason: str,
        artifact_paths: dict[str, str | Path] | None = None,
        record_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "schema_version": SCHEMA_VERSION,
            "event": "step_skipped",
            "step": name,
            "command": command,
            "shell_command": shlex.join(command),
            "cwd": _repo_rel(self.root, root=self.root),
            "artifact_paths": {key: _repo_rel(value, root=self.root) for key, value in (artifact_paths or {}).items()},
            "reason": reason,
            "skipped": True,
            "ok": False,
            "exit_code": None,
            "started_at": _utc_now(),
            "ended_at": _utc_now(),
            "wall_seconds": 0.0,
            "peak_rss_bytes": None,
            "mlx_memory": {"available": False, "peak_bytes": None, "reason": "skipped"},
        }
        if record_fields:
            record.update(record_fields)
        self.records.append(record)
        self._append_event(record)
        return record

    def preflight_command(self) -> list[str]:
        return [self.python, "scripts/preflight_assets.py", _cmd_path(self.args.models_root, root=self.root)]

    def quantize_command(self, output: str | Path | None = None) -> list[str]:
        target = self.args.text_encoder if output is None else output
        return [
            self.python,
            "scripts/run_with_mlx_memory.py",
            "scripts/quantize_text_encoder.py",
            "--source",
            _cmd_path(self.args.quant_source, root=self.root),
            "--output",
            _cmd_path(target, root=self.root),
            "--bits",
            str(self.args.bits),
            "--group-size",
            str(self.args.group_size),
            "--num-layers",
            str(self.args.num_layers),
        ]

    def verify_prompt_release_command(self) -> list[str]:
        return [
            self.python,
            "scripts/run_with_mlx_memory.py",
            "scripts/verify_nightly_gates.py",
            "--memory-limit-gb",
            str(self.args.memory_limit_gb),
            "prompt-release",
            "--text-encoder",
            _cmd_path(self.args.text_encoder, root=self.root),
            "--prompt",
            self.args.prompt,
        ]

    def verify_streaming_components_command(self) -> list[str]:
        command = [
            self.python,
            "scripts/run_with_mlx_memory.py",
            "scripts/verify_nightly_gates.py",
            "--memory-limit-gb",
            str(self.args.memory_limit_gb),
            "streaming-components",
            "--transformer",
            _cmd_path(self.args.transformer, root=self.root),
        ]
        if self.args.turbo_lora:
            command.extend(
                [
                    "--turbo-lora",
                    _cmd_path(self.args.turbo_lora, root=self.root),
                    "--turbo-lora-alpha",
                    str(self.args.turbo_lora_alpha),
                    "--turbo-lora-scale",
                    str(self.args.turbo_lora_scale),
                ]
            )
        return command

    def verify_ffmpeg_command(self) -> list[str]:
        return [self.ffmpeg, "-version"]

    def verify_ffprobe_command(self) -> list[str]:
        return [self.ffprobe, "-version"]

    def generation_output(self, width: int, height: int) -> Path:
        return self.output_dir / f"nightly_{width}x{height}_{_duration_token(self.duration)}_{self.args.steps}sigma.mp4"

    def generation_command(self, width: int, height: int, output: Path) -> list[str]:
        command = [
            self.python,
            "scripts/run_with_mlx_memory.py",
            "scripts/generate.py",
            self.args.prompt,
            "-o",
            _repo_rel(output, root=self.root),
            "-c",
            _cmd_path(self.args.checkpoint, root=self.root),
            "-t",
            _cmd_path(self.args.transformer, root=self.root),
            "--text-encoder",
            _cmd_path(self.args.text_encoder, root=self.root),
            "--low-memory",
            "--stream-blocks",
            "--height",
            str(height),
            "--width",
            str(width),
            "--duration",
            str(self.duration),
            "--steps",
            str(self.args.steps),
            "--seed",
            str(self.args.seed),
            "--memory-limit-gb",
            str(self.args.memory_limit_gb),
            "--ffmpeg",
            self.ffmpeg,
            "--require-muxed-mp4",
        ]
        if self.args.turbo_lora:
            command.extend(
                [
                    "--turbo-lora",
                    _cmd_path(self.args.turbo_lora, root=self.root),
                    "--turbo-lora-alpha",
                    str(self.args.turbo_lora_alpha),
                    "--turbo-lora-scale",
                    str(self.args.turbo_lora_scale),
                ]
            )
        return command

    def ffprobe_command(self, output: Path) -> list[str]:
        return [
            self.ffprobe,
            "-hide_banner",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            _repo_rel(output, root=self.root),
        ]

    def audio_activity_command(self, output: Path) -> list[str]:
        return [
            self.ffmpeg,
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-i",
            _repo_rel(output, root=self.root),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(self.args.audio_activity_sample_rate)),
            "-f",
            "f32le",
            "pipe:1",
        ]

    def _text_encoder_final_path(self) -> Path:
        return _resolve_repo_path(self.args.text_encoder, root=self.root)

    def _text_encoder_staging_path(self) -> Path:
        final = self._text_encoder_final_path()
        return final.parent / f".{final.name}.staging-{_sanitize_run_token(self.run_id)}"

    def _text_encoder_validation(self, path: str | Path) -> dict[str, Any]:
        return _validate_quantized_text_encoder_dir(
            path,
            bits=int(self.args.bits),
            group_size=int(self.args.group_size),
            num_layers=int(self.args.num_layers),
            root=self.root,
        )

    def _record_text_encoder_decision(
        self,
        *,
        status: str,
        reason: str,
        ok: bool,
        validation: dict[str, Any] | None,
        staging_validation: dict[str, Any] | None = None,
        exit_code: int | None = None,
    ) -> dict[str, Any]:
        final = self._text_encoder_final_path()
        staging = self._text_encoder_staging_path()
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event": "step_skipped" if ok else "step_blocked",
            "step": "quantize_text_encoder",
            "command": self.quantize_command(),
            "shell_command": shlex.join(self.quantize_command()),
            "cwd": _repo_rel(self.root, root=self.root),
            "artifact_paths": {
                "quant_source_dir": _repo_rel(self.args.quant_source, root=self.root),
                "text_encoder_dir": _repo_rel(final, root=self.root),
                "text_encoder_staging_dir": _repo_rel(staging, root=self.root),
            },
            "reason": reason,
            "skipped": True,
            "ok": ok,
            "exit_code": 0 if ok else exit_code,
            "started_at": _utc_now(),
            "ended_at": _utc_now(),
            "wall_seconds": 0.0,
            "peak_rss_bytes": None,
            "mlx_memory": {"available": False, "peak_bytes": None, "reason": "not_run"},
            "text_encoder_status": status,
            "text_encoder_final_exists": final.exists() or final.is_symlink(),
            "atomic_commit": {
                "method": "os.replace",
                "committed": False,
                "source_dir": _repo_rel(staging, root=self.root),
                "target_dir": _repo_rel(final, root=self.root),
                "reason": reason,
            },
        }
        if validation is not None:
            record["text_encoder_validation"] = validation
        if staging_validation is not None:
            record["text_encoder_staging_validation"] = staging_validation
        if not ok:
            record["failure_class"] = "validation_failed"
        self.records.append(record)
        self._append_event(record)
        return record

    def _quantization_postprocess(self, *, final: Path, staging: Path) -> Postprocess:
        def postprocess(exit_code: int | None, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
            commit: dict[str, Any] = {
                "method": "os.replace",
                "committed": False,
                "source_dir": _repo_rel(staging, root=self.root),
                "target_dir": _repo_rel(final, root=self.root),
            }
            result: dict[str, Any] = {
                "postcheck_ok": False,
                "text_encoder_final_dir": _repo_rel(final, root=self.root),
                "text_encoder_staging_dir": _repo_rel(staging, root=self.root),
                "text_encoder_final_exists": final.exists() or final.is_symlink(),
                "atomic_commit": commit,
            }
            if exit_code != 0:
                commit["reason"] = "quantization_command_failed"
                result["text_encoder_status"] = "quantization_command_failed"
                return result

            staging_validation = self._text_encoder_validation(staging)
            result["text_encoder_staging_validation"] = staging_validation
            if not staging_validation["ok"]:
                commit["reason"] = "staging_validation_failed"
                result["text_encoder_status"] = "staging_validation_failed"
                return result
            if final.exists() or final.is_symlink():
                commit["reason"] = "final_target_exists_after_quantization"
                result["text_encoder_status"] = "final_target_exists_after_quantization"
                result["text_encoder_validation"] = self._text_encoder_validation(final)
                return result

            try:
                os.replace(staging, final)
                _fsync_directory(final.parent)
            except Exception as exc:
                commit["reason"] = f"atomic_commit_failed: {type(exc).__name__}: {exc}"
                result["text_encoder_status"] = "atomic_commit_failed"
                return result

            commit.update({"committed": True, "committed_at": _utc_now()})
            final_validation = self._text_encoder_validation(final)
            result.update(
                {
                    "text_encoder_status": "created_atomic" if final_validation["ok"] else "committed_but_invalid",
                    "text_encoder_final_exists": final.exists() or final.is_symlink(),
                    "text_encoder_validation": final_validation,
                    "postcheck_ok": final_validation["ok"],
                }
            )
            if not final_validation["ok"]:
                commit["reason"] = "final_validation_failed_after_commit"
            return result

        return postprocess

    def _ensure_text_encoder(self) -> dict[str, Any]:
        final = self._text_encoder_final_path()
        staging = self._text_encoder_staging_path()
        if final.exists() or final.is_symlink():
            validation = self._text_encoder_validation(final)
            if validation["ok"]:
                return self._record_text_encoder_decision(
                    status="reused_existing_valid",
                    reason="valid_existing_text_encoder_reused",
                    ok=True,
                    validation=validation,
                    exit_code=0,
                )
            return self._record_text_encoder_decision(
                status="invalid_existing_text_encoder",
                reason="invalid_existing_text_encoder",
                ok=False,
                validation=validation,
                exit_code=VALIDATION_FAILURE,
            )
        if staging.exists() or staging.is_symlink():
            staging_validation = self._text_encoder_validation(staging)
            return self._record_text_encoder_decision(
                status="stale_text_encoder_staging_exists",
                reason="stale_text_encoder_staging_exists",
                ok=False,
                validation=None,
                staging_validation=staging_validation,
                exit_code=VALIDATION_FAILURE,
            )
        return self._run_step(
            "quantize_text_encoder",
            self.quantize_command(output=staging),
            artifact_paths={
                "quant_source_dir": self.args.quant_source,
                "text_encoder_dir": final,
                "text_encoder_staging_dir": staging,
            },
            postprocess=self._quantization_postprocess(final=final, staging=staging),
            record_fields={
                "text_encoder_status": "quantizing_to_staging",
                "atomic_commit_plan": {
                    "method": "os.replace",
                    "source_dir": _repo_rel(staging, root=self.root),
                    "target_dir": _repo_rel(final, root=self.root),
                    "requires_final_absent": True,
                },
            },
        )

    def _preflight_postprocess(self, exit_code: int | None, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
        result: dict[str, Any] = {"postcheck_ok": exit_code == 0}
        try:
            payload = json.loads(stdout_path.read_text())
        except Exception as exc:
            result.update({"preflight_json_parse_error": f"{type(exc).__name__}: {exc}", "postcheck_ok": False})
            return result
        result.update(
            {
                "preflight_ok": payload.get("ok"),
                "preflight_exit_code": payload.get("exit_code"),
                "preflight_summary": payload.get("summary"),
                "preflight_issue_count": len(payload.get("issues", [])),
            }
        )
        if self.args.update_preflight_json:
            target = _resolve_repo_path(self.args.preflight_json, root=self.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stdout_path, target)
            result["preflight_json_mirror"] = _repo_rel(target, root=self.root)
        return result

    def _media_tool_postprocess(self, tool_name: str, tool_record: dict[str, Any]) -> Postprocess:
        def postprocess(exit_code: int | None, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
            stdout_text = _read_text_tail(stdout_path, limit=4096)
            stderr_text = _read_text_tail(stderr_path, limit=4096)
            first_line = next((line for line in (stdout_text + "\n" + stderr_text).splitlines() if line.strip()), "")
            return {
                "postcheck_ok": exit_code == 0,
                "media_tool_name": tool_name,
                "media_tool": tool_record,
                "media_tool_version_line": first_line[:500],
            }

        return postprocess

    def _verification_postprocess(self, gate: str) -> Postprocess:
        def postprocess(exit_code: int | None, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
            result: dict[str, Any] = {"postcheck_ok": False, "verification_gate": gate}
            try:
                payload = json.loads(stdout_path.read_text())
            except Exception as exc:
                result["verification_error"] = f"verifier JSON is not parseable: {type(exc).__name__}: {exc}"
                return result
            result.update(
                {
                    "verification_ok": payload.get("ok"),
                    "verification_payload_path": _repo_rel(stdout_path, root=self.root),
                    "verification_error": payload.get("error"),
                    "verification_error_type": payload.get("error_type"),
                    "verification_summary": {
                        key: payload.get(key)
                        for key in (
                            "gate",
                            "component_released",
                            "detached_usable_after_release",
                            "token_tag_count",
                            "block_count",
                            "checked_block_indices",
                            "logical_bytes_loaded_total",
                        )
                        if key in payload
                    },
                }
            )
            if "turbo_lora" in payload:
                result["verification_turbo_lora"] = payload["turbo_lora"]
            result["postcheck_ok"] = exit_code == 0 and payload.get("ok") is True
            return result

        return postprocess

    def _ffprobe_postprocess(self, output: Path) -> Postprocess:
        def postprocess(exit_code: int | None, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
            size = output.stat().st_size if output.exists() else 0
            result: dict[str, Any] = {
                "media_path": _repo_rel(output, root=self.root),
                "media_size_bytes": size,
                "postcheck_ok": False,
            }
            if size <= 0:
                result["validation_error"] = "media file is missing or empty"
                return result
            if exit_code != 0:
                result["validation_error"] = "ffprobe returned non-zero"
                return result
            try:
                payload = json.loads(stdout_path.read_text())
            except Exception as exc:
                result["validation_error"] = f"ffprobe JSON is not parseable: {type(exc).__name__}: {exc}"
                return result
            streams = payload.get("streams", [])
            video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
            audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
            result.update(
                {
                    "video_stream_count": len(video_streams),
                    "audio_stream_count": len(audio_streams),
                    "format_name": payload.get("format", {}).get("format_name"),
                    "duration_seconds": payload.get("format", {}).get("duration"),
                    "postcheck_ok": bool(video_streams and audio_streams),
                }
            )
            if not video_streams:
                result["validation_error"] = "ffprobe found no video stream"
            elif not audio_streams:
                result["validation_error"] = "ffprobe found no audio stream"
            return result

        return postprocess

    def _audio_activity_postprocess(self, output: Path) -> Postprocess:
        def postprocess(exit_code: int | None, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
            size = output.stat().st_size if output.exists() else 0
            result: dict[str, Any] = {
                "media_path": _repo_rel(output, root=self.root),
                "media_size_bytes": size,
                "postcheck_ok": False,
            }
            if size <= 0:
                result["validation_error"] = "media file is missing or empty"
                return result
            if exit_code != 0:
                result["validation_error"] = "ffmpeg audio decode returned non-zero"
                return result
            result.update(
                _audio_activity_stats(
                    stdout_path.read_bytes(),
                    sample_rate_hz=int(self.args.audio_activity_sample_rate),
                    min_rms=float(self.args.audio_activity_min_rms),
                    min_peak=float(self.args.audio_activity_min_peak),
                    min_nonzero_samples=int(self.args.audio_activity_min_nonzero_samples),
                )
            )
            return result

        return postprocess

    def _skip_ladder(self, reason: str) -> None:
        for width, height in self.ladder:
            output = self.generation_output(width, height)
            self._skip_step(
                f"generate_{width}x{height}",
                self.generation_command(width, height, output),
                reason=reason,
                artifact_paths={"media_path": output},
                record_fields={"generation_schedule_contract": self.generation_schedule_contract},
            )
            self._skip_step(
                f"ffprobe_{width}x{height}",
                self.ffprobe_command(output),
                reason=reason,
                artifact_paths={"media_path": output},
            )
            self._skip_step(
                f"audio_activity_{width}x{height}",
                self.audio_activity_command(output),
                reason=reason,
                artifact_paths={"media_path": output},
                record_fields={"audio_activity_contract": self.audio_activity_contract},
            )

    def _skip_verification_steps(self, reason: str) -> None:
        self._skip_step(
            "verify_prompt_release",
            self.verify_prompt_release_command(),
            reason=reason,
            artifact_paths={"text_encoder_dir": self.args.text_encoder},
        )
        self._skip_step(
            "verify_streaming_components",
            self.verify_streaming_components_command(),
            reason=reason,
            artifact_paths={"transformer_dir": self.args.transformer},
        )

    def _skip_after_blocker(self, reason: str) -> None:
        self._skip_step(
            "quantize_text_encoder",
            self.quantize_command(),
            reason=reason,
            artifact_paths={"text_encoder_dir": self.args.text_encoder},
        )
        self._skip_verification_steps(reason)
        self._skip_ladder(reason)

    def run(self) -> int:
        if self.args.dry_run:
            self._run_step(
                "preflight",
                self.preflight_command(),
                artifact_paths={"preflight_json": self.run_dir / "preflight.json"},
                stdout_path=self.run_dir / "preflight.json",
                postprocess=None,
                dry_run=True,
            )
            self._skip_after_blocker("dry_run")
            self._write_summary(status="dry_run", exit_code=0)
            return 0

        preflight = self._run_step(
            "preflight",
            self.preflight_command(),
            artifact_paths={"preflight_json": self.run_dir / "preflight.json"},
            stdout_path=self.run_dir / "preflight.json",
            postprocess=self._preflight_postprocess,
        )
        if not preflight["ok"]:
            reason = f"preflight_failed_exit_{preflight['exit_code']}"
            self._skip_after_blocker(reason)
            exit_code = int(preflight["exit_code"] or ASSET_FAILURE)
            self._write_summary(
                status="blocked_preflight",
                exit_code=exit_code,
                extra={
                    "blocker": reason,
                    "preflight_summary": preflight.get("preflight_summary"),
                    "preflight_issue_count": preflight.get("preflight_issue_count"),
                },
            )
            return exit_code

        ffmpeg_verify = self._run_step(
            "verify_ffmpeg",
            self.verify_ffmpeg_command(),
            artifact_paths={"ffmpeg": self.ffmpeg},
            postprocess=self._media_tool_postprocess("ffmpeg", self.ffmpeg_tool),
        )
        ffprobe_verify = self._run_step(
            "verify_ffprobe",
            self.verify_ffprobe_command(),
            artifact_paths={"ffprobe": self.ffprobe},
            postprocess=self._media_tool_postprocess("ffprobe", self.ffprobe_tool),
        )
        if not (ffmpeg_verify["ok"] and ffprobe_verify["ok"]):
            failed = [record["step"] for record in (ffmpeg_verify, ffprobe_verify) if not record["ok"]]
            reason = "media_tool_verification_failed_" + "_".join(failed)
            self._skip_after_blocker(reason)
            exit_code = int(ffmpeg_verify.get("exit_code") or ffprobe_verify.get("exit_code") or VALIDATION_FAILURE)
            self._write_summary(
                status="blocked_media_tools",
                exit_code=exit_code,
                extra={
                    "blocker": reason,
                    "media_tool_failures": failed,
                },
            )
            return exit_code

        quantize = self._ensure_text_encoder()
        if not quantize["ok"]:
            text_status = quantize.get("text_encoder_status")
            if text_status in {"invalid_existing_text_encoder", "stale_text_encoder_staging_exists"}:
                reason = str(quantize.get("reason") or text_status)
                exit_code = int(quantize.get("exit_code") or VALIDATION_FAILURE)
                summary_status = "blocked_invalid_text_encoder" if text_status == "invalid_existing_text_encoder" else "blocked_text_encoder_staging"
            elif text_status in {
                "staging_validation_failed",
                "final_target_exists_after_quantization",
                "atomic_commit_failed",
                "committed_but_invalid",
            }:
                reason = str(text_status)
                exit_code = VALIDATION_FAILURE
                summary_status = "failed_text_encoder_atomic_commit"
            else:
                reason = f"quantization_failed_exit_{quantize['exit_code']}"
                exit_code = int(quantize["exit_code"] or 1)
                summary_status = "failed_quantization"
            self._skip_verification_steps(reason)
            self._skip_ladder(reason)
            self._write_summary(
                status=summary_status,
                exit_code=exit_code,
                extra={
                    "blocker": reason,
                    "text_encoder_status": text_status,
                    "text_encoder_validation": quantize.get("text_encoder_validation"),
                    "text_encoder_staging_validation": quantize.get("text_encoder_staging_validation"),
                },
            )
            return exit_code

        prompt_verify = self._run_step(
            "verify_prompt_release",
            self.verify_prompt_release_command(),
            artifact_paths={"text_encoder_dir": self.args.text_encoder},
            postprocess=self._verification_postprocess("prompt_release"),
        )
        if not prompt_verify["ok"]:
            reason = f"prompt_release_verification_failed_exit_{prompt_verify['exit_code']}"
            self._skip_step(
                "verify_streaming_components",
                self.verify_streaming_components_command(),
                reason=reason,
                artifact_paths={"transformer_dir": self.args.transformer},
            )
            self._skip_ladder(reason)
            exit_code = int(prompt_verify["exit_code"] or VALIDATION_FAILURE)
            self._write_summary(status="failed_prompt_release_verification", exit_code=exit_code)
            return exit_code

        streaming_verify = self._run_step(
            "verify_streaming_components",
            self.verify_streaming_components_command(),
            artifact_paths={"transformer_dir": self.args.transformer},
            postprocess=self._verification_postprocess("streaming_components"),
        )
        if not streaming_verify["ok"]:
            reason = f"streaming_components_verification_failed_exit_{streaming_verify['exit_code']}"
            self._skip_ladder(reason)
            exit_code = int(streaming_verify["exit_code"] or VALIDATION_FAILURE)
            self._write_summary(status="failed_streaming_verification", exit_code=exit_code)
            return exit_code

        for width, height in self.ladder:
            output = self.generation_output(width, height)
            generate = self._run_step(
                f"generate_{width}x{height}",
                self.generation_command(width, height, output),
                artifact_paths={"media_path": output},
                record_fields={"generation_schedule_contract": self.generation_schedule_contract},
            )
            if not generate["ok"]:
                for remaining_width, remaining_height in self.ladder[self.ladder.index((width, height)) + 1 :]:
                    remaining_output = self.generation_output(remaining_width, remaining_height)
                    self._skip_step(
                        f"generate_{remaining_width}x{remaining_height}",
                        self.generation_command(remaining_width, remaining_height, remaining_output),
                        reason=f"previous_generation_failed_{width}x{height}",
                        artifact_paths={"media_path": remaining_output},
                        record_fields={"generation_schedule_contract": self.generation_schedule_contract},
                    )
                    self._skip_step(
                        f"ffprobe_{remaining_width}x{remaining_height}",
                        self.ffprobe_command(remaining_output),
                        reason=f"previous_generation_failed_{width}x{height}",
                        artifact_paths={"media_path": remaining_output},
                    )
                    self._skip_step(
                        f"audio_activity_{remaining_width}x{remaining_height}",
                        self.audio_activity_command(remaining_output),
                        reason=f"previous_generation_failed_{width}x{height}",
                        artifact_paths={"media_path": remaining_output},
                        record_fields={"audio_activity_contract": self.audio_activity_contract},
                    )
                exit_code = int(generate["exit_code"] or 1)
                self._write_summary(status="failed_generation", exit_code=exit_code)
                return exit_code

            ffprobe_json = self.logs_dir / f"ffprobe_{width}x{height}.json"
            validate = self._run_step(
                f"ffprobe_{width}x{height}",
                self.ffprobe_command(output),
                artifact_paths={"media_path": output, "ffprobe_json": ffprobe_json},
                stdout_path=ffprobe_json,
                postprocess=self._ffprobe_postprocess(output),
            )
            if not validate["ok"]:
                exit_code = int(validate["exit_code"] or VALIDATION_FAILURE)
                self._write_summary(status="failed_media_validation", exit_code=exit_code)
                return exit_code

            decoded_audio_path = self.logs_dir / f"audio_activity_{width}x{height}.f32le"
            audio_activity = self._run_step(
                f"audio_activity_{width}x{height}",
                self.audio_activity_command(output),
                artifact_paths={"media_path": output, "decoded_audio_f32le": decoded_audio_path},
                stdout_path=decoded_audio_path,
                postprocess=self._audio_activity_postprocess(output),
                record_fields={"audio_activity_contract": self.audio_activity_contract},
            )
            if not audio_activity["ok"]:
                exit_code = int(audio_activity["exit_code"] or VALIDATION_FAILURE)
                self._write_summary(status="failed_media_validation", exit_code=exit_code)
                return exit_code

        self._write_summary(status="success", exit_code=0)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default=".venv/bin/python", help="repo-local Python interpreter")
    parser.add_argument("--time-bin", default="/usr/bin/time", help="time binary used with -l for RSS accounting")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable for MP4 muxing; default auto-prefers repo-local tools")
    parser.add_argument("--ffprobe", default=None, help="ffprobe executable for media/audio validation; default auto-prefers repo-local tools")
    parser.add_argument("--run-id", default=None, help="stable run id; defaults to UTC timestamp")
    parser.add_argument("--run-dir", type=Path, default=None, help="directory for JSONL logs and summary; default: out/nightly_deployment/RUN_ID")
    parser.add_argument("--output-dir", type=Path, default=None, help="media output directory; default: RUN_DIR/outputs")
    parser.add_argument("--models-root", default="models", help="root passed to scripts/preflight_assets.py")
    parser.add_argument("--checkpoint", "-c", default="models/MiniMax-H3/FL2VA", help="official FL2VA checkpoint directory")
    parser.add_argument("--transformer", "-t", default="models/MiniMax-H3-MLX-4bit", help="quantized DiT directory")
    parser.add_argument("--quant-source", default="models/MiniMax-H3/FL2VA/text_encoder", help="official FL2VA text encoder source")
    parser.add_argument("--text-encoder", default="models/MiniMax-H3/FL2VA/text_encoder-mlx-4bit", help="derived quantized text encoder output")
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 6, 8))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=50)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=5, help="sigma grid points; 5 means 4 NFE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0, help="MLX allocation guideline for low-memory verifier/generation steps")
    parser.add_argument("--audio-activity-sample-rate", type=int, default=DEFAULT_AUDIO_ACTIVITY_SAMPLE_RATE_HZ, help="sample rate for decoded mono f32le audio validation")
    parser.add_argument("--audio-activity-min-rms", type=float, default=DEFAULT_AUDIO_ACTIVITY_MIN_RMS, help="minimum decoded-audio RMS required for non-silent validation")
    parser.add_argument("--audio-activity-min-peak", type=float, default=DEFAULT_AUDIO_ACTIVITY_MIN_PEAK, help="minimum decoded-audio absolute peak required for non-silent validation")
    parser.add_argument("--audio-activity-min-nonzero-samples", type=int, default=DEFAULT_AUDIO_ACTIVITY_MIN_NONZERO_SAMPLES, help="minimum samples above the nonzero epsilon required for non-silent validation")
    parser.add_argument("--turbo-lora", default=None, help="optional MiniMax-H3 Turbo PEFT safetensors used by verifier and generation")
    parser.add_argument("--turbo-lora-alpha", type=float, default=8.0, help="training alpha for --turbo-lora")
    parser.add_argument("--turbo-lora-scale", type=float, default=1.0, help="runtime multiplier for --turbo-lora")
    parser.add_argument("--ladder", default=DEFAULT_LADDER, help="comma-separated WIDTHxHEIGHT list")
    parser.add_argument("--preflight-json", type=Path, default=Path("docs/NIGHTLY_ASSET_PREFLIGHT.json"), help="mirror latest preflight JSON here")
    parser.add_argument("--no-update-preflight-json", dest="update_preflight_json", action="store_false", help="do not mirror preflight stdout to docs/NIGHTLY_ASSET_PREFLIGHT.json")
    parser.set_defaults(update_preflight_json=True)
    parser.add_argument("--dry-run", action="store_true", help="log the planned sequence without executing commands")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _parse_ladder(args.ladder)
        _generation_schedule_contract(int(args.steps))
    except ValueError as exc:
        parser.error(str(exc))
    if args.audio_activity_sample_rate <= 0:
        parser.error("--audio-activity-sample-rate must be positive")
    if args.audio_activity_min_rms < 0 or args.audio_activity_min_peak < 0:
        parser.error("audio activity amplitude thresholds must be non-negative")
    if args.audio_activity_min_nonzero_samples < 0:
        parser.error("--audio-activity-min-nonzero-samples must be non-negative")
    runner = DeploymentRunner(args)
    exit_code = runner.run()
    status = json.loads(runner.summary_path.read_text())["status"]
    print(
        json.dumps(
            {
                "status": status,
                "exit_code": exit_code,
                "run_dir": _repo_rel(runner.run_dir, root=ROOT),
                "summary": _repo_rel(runner.summary_path, root=ROOT),
                "events": _repo_rel(runner.events_path, root=ROOT),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
