#!/usr/bin/env python3
"""Run the manifest-defined MiniMax-H3 320p quality-suite A/B matrix.

The runner is deliberately conservative: it performs a disk/process/asset/media
preflight first, executes at most one generation process at a time, validates
successful MP4s with the project-local ffprobe/ffmpeg route, and writes paired
metric JSONs only when the same case/seed reference and candidate artifacts both
exist.

This script is an orchestrator only; it does not download weights and it does
not claim lossless/near-lossless quality.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "quality_suite_manifest.json"
DEFAULT_SUITE_ROOT = ROOT / "out" / "quality_suite"
MIN_FREE_BYTES = 100_000_000_000
SCHEMA_VERSION = 1
TIME_RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\b", re.MULTILINE)
ARGUS_MLX_RE = re.compile(
    r"ARGUS_MLX_MEMORY\s+"
    r"peak_bytes=(?P<peak>\d+|None)\s+"
    r"active_bytes=(?P<active>\d+|None)\s+"
    r"cache_bytes=(?P<cache>\d+|None)"
)
OOM_MARKERS = (
    "out of memory",
    "cannot allocate memory",
    "memoryerror",
    "mlx_error",
    "killed: 9",
    "signal 9",
)
PROCESS_PATTERNS = (
    "scripts/generate.py",
    "scripts/run_quality_suite_matrix.py",
    "run_nightly_deployment.py",
    "run_with_mlx_memory.py scripts/generate.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_id() -> str:
    return "quality-suite-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported manifest schema_version={payload.get('schema_version')!r}")
    return payload


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_template(template: str, values: dict[str, Any]) -> Path:
    rendered = template.format(**values)
    path = Path(rendered)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class MatrixItem:
    case_id: str
    coverage_axis: str
    prompt: str
    seed: int
    condition_id: str
    steps_sigma_points: int
    turbo_lora_required: bool
    quality_role: str
    width: int
    height: int
    duration_seconds: float
    fps: int
    media_path: Path
    metadata_path: Path

    @property
    def denoiser_evaluations(self) -> int:
        return self.steps_sigma_points - 1


def build_matrix(manifest: dict[str, Any], *, width: int | None = None, height: int | None = None) -> list[MatrixItem]:
    defaults = manifest["shared_generation_defaults"]
    small_gate = manifest["generation_policy"]["small_gate_resolution"]
    width = int(width or small_gate["width"])
    height = int(height or small_gate["height"])
    duration_seconds = float(defaults["duration_seconds"])
    fps = int(defaults["fps"])
    layout = manifest["output_layout"]
    suite_id = manifest["suite_id"]
    matrix: list[MatrixItem] = []
    for case in manifest["cases"]:
        seed = int(case["seeds"][0])
        for condition in manifest["planned_initial_conditions"]:
            values = {
                "suite_id": suite_id,
                "case_id": case["case_id"],
                "condition_id": condition["condition_id"],
                "seed": seed,
                "width": width,
                "height": height,
                "duration_seconds": duration_seconds,
                "steps_sigma_points": int(condition["steps_sigma_points"]),
            }
            matrix.append(
                MatrixItem(
                    case_id=case["case_id"],
                    coverage_axis=case["coverage_axis"],
                    prompt=case["prompt"],
                    seed=seed,
                    condition_id=condition["condition_id"],
                    steps_sigma_points=int(condition["steps_sigma_points"]),
                    turbo_lora_required=bool(condition["turbo_lora_required"]),
                    quality_role=condition["quality_role"],
                    width=width,
                    height=height,
                    duration_seconds=duration_seconds,
                    fps=fps,
                    media_path=resolve_template(layout["media_template"], values),
                    metadata_path=resolve_template(layout["metadata_template"], values),
                )
            )
    return matrix


def pair_metric_path(manifest: dict[str, Any], *, case_id: str, seed: int, width: int, height: int,
                     reference_condition_id: str, candidate_condition_id: str) -> Path:
    defaults = manifest["shared_generation_defaults"]
    values = {
        "suite_id": manifest["suite_id"],
        "case_id": case_id,
        "seed": seed,
        "width": width,
        "height": height,
        "duration_seconds": float(defaults["duration_seconds"]),
        "reference_condition_id": reference_condition_id,
        "candidate_condition_id": candidate_condition_id,
    }
    return resolve_template(manifest["output_layout"]["pair_metrics_template"], values)


def blind_table_path(manifest: dict[str, Any], *, width: int, height: int) -> Path:
    defaults = manifest["shared_generation_defaults"]
    values = {
        "suite_id": manifest["suite_id"],
        "width": width,
        "height": height,
        "duration_seconds": float(defaults["duration_seconds"]),
    }
    return resolve_template(manifest["output_layout"]["blind_review_table_template"], values)


def run_command(cmd: list[str], *, cwd: Path = ROOT, timeout: float | None = None,
                stdout_path: Path | None = None, stderr_path: Path | None = None) -> dict[str, Any]:
    start = time.time()
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("wb") if stdout_path else subprocess.PIPE
    stderr_handle = stderr_path.open("wb") if stderr_path else subprocess.PIPE
    try:
        process = subprocess.run(cmd, cwd=cwd, stdout=stdout_handle, stderr=stderr_handle, timeout=timeout)
    finally:
        if stdout_path is not None:
            stdout_handle.close()
        if stderr_path is not None:
            stderr_handle.close()
    wall = round(time.time() - start, 3)
    stdout_text = stdout_path.read_text(errors="replace") if stdout_path else (process.stdout or b"").decode(errors="replace")
    stderr_text = stderr_path.read_text(errors="replace") if stderr_path else (process.stderr or b"").decode(errors="replace")
    rss_match = TIME_RSS_RE.search(stderr_text)
    mlx_match = ARGUS_MLX_RE.search(stderr_text)
    def parse_mlx(name: str) -> int | None:
        if not mlx_match:
            return None
        raw = mlx_match.group(name)
        return None if raw == "None" else int(raw)
    return {
        "cmd": cmd,
        "cmd_pretty": shlex.join(cmd),
        "exit_code": int(process.returncode),
        "wall_seconds": wall,
        "stdout_path": rel(stdout_path) if stdout_path else None,
        "stderr_path": rel(stderr_path) if stderr_path else None,
        "stdout_tail": stdout_text[-4000:],
        "stderr_tail": stderr_text[-4000:],
        "peak_rss_bytes": int(rss_match.group(1)) if rss_match else None,
        "mlx_memory": {
            "peak_bytes": parse_mlx("peak"),
            "active_bytes": parse_mlx("active"),
            "cache_bytes": parse_mlx("cache"),
        },
        "oom_or_killed_marker": find_oom_marker(stdout_text + "\n" + stderr_text),
    }


def find_oom_marker(text: str) -> str | None:
    lowered = text.lower()
    for marker in OOM_MARKERS:
        if marker in lowered:
            return marker
    return None


def disk_status(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        probe = path if path.exists() else path.parent
        usage = shutil.disk_usage(probe)
        result.append(
            {
                "path": rel(path),
                "probe_path": rel(probe),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "free_gb": round(usage.free / 1e9, 3),
                "meets_100gb_free": usage.free >= MIN_FREE_BYTES,
            }
        )
    return result


def process_matches() -> list[str]:
    ps = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,stat=,rss=,etime=,command="],
        text=True,
        capture_output=True,
        cwd=ROOT,
        timeout=20,
    )
    parsed: list[tuple[int, int, str, str]] = []
    parent_by_pid: dict[int, int] = {}
    for line in ps.stdout.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        parsed.append((pid, ppid, parts[5], line))
        parent_by_pid[pid] = ppid

    exclude = {os.getpid()}
    cursor = os.getppid()
    while cursor and cursor not in exclude:
        exclude.add(cursor)
        cursor = parent_by_pid.get(cursor, 0)

    matches: list[str] = []
    for pid, ppid, command, line in parsed:
        if pid in exclude or ppid in exclude:
            continue
        if any(pattern in command for pattern in PROCESS_PATTERNS):
            matches.append(line)
    return matches


def run_preflight(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    defaults = manifest["shared_generation_defaults"]
    checked_paths = [
        defaults["checkpoint"],
        defaults["transformer"],
        defaults["text_encoder"],
        defaults["turbo_lora"],
    ]
    logs = run_dir / "logs"
    asset = run_command(
        ["./.venv/bin/python", "scripts/preflight_assets.py", *checked_paths],
        stdout_path=logs / "asset_preflight.stdout",
        stderr_path=logs / "asset_preflight.stderr",
    )
    try:
        asset_payload = json.loads((logs / "asset_preflight.stdout").read_text())
    except Exception as exc:
        asset_payload = {"parse_error": repr(exc)}
    ffmpeg = run_command([defaults["ffmpeg"], "-version"], stdout_path=logs / "ffmpeg_version.stdout", stderr_path=logs / "ffmpeg_version.stderr")
    ffprobe = run_command([defaults["ffprobe"], "-version"], stdout_path=logs / "ffprobe_version.stdout", stderr_path=logs / "ffprobe_version.stderr")
    disks = disk_status([ROOT, ROOT / "models", ROOT / "out"])
    matches = process_matches()
    swap = None
    if platform.system() == "Darwin":
        swap = run_command(["sysctl", "vm.swapusage"], stdout_path=logs / "swapusage.stdout", stderr_path=logs / "swapusage.stderr")
    payload = {
        "created_at": utc_now(),
        "checked_paths": checked_paths,
        "disk": disks,
        "disk_ok": all(item["meets_100gb_free"] for item in disks),
        "process_patterns": PROCESS_PATTERNS,
        "matching_process_lines": matches,
        "no_competing_high_memory_process": len(matches) == 0,
        "asset_preflight": {
            "exit_code": asset["exit_code"],
            "ok": asset_payload.get("ok"),
            "summary": asset_payload.get("summary"),
            "issues": asset_payload.get("issues"),
            "command": asset["cmd"],
            "stdout_path": asset["stdout_path"],
            "stderr_path": asset["stderr_path"],
            "wall_seconds": asset["wall_seconds"],
        },
        "ffmpeg": {"exit_code": ffmpeg["exit_code"], "first_line": ffmpeg["stdout_tail"].splitlines()[:1], "command": ffmpeg["cmd"]},
        "ffprobe": {"exit_code": ffprobe["exit_code"], "first_line": ffprobe["stdout_tail"].splitlines()[:1], "command": ffprobe["cmd"]},
        "swapusage": swap,
    }
    payload["ok_for_generation"] = (
        payload["disk_ok"]
        and payload["no_competing_high_memory_process"]
        and asset["exit_code"] == 0
        and asset_payload.get("ok") is True
        and ffmpeg["exit_code"] == 0
        and ffprobe["exit_code"] == 0
    )
    return payload


def generation_command(item: MatrixItem, manifest: dict[str, Any]) -> list[str]:
    defaults = manifest["shared_generation_defaults"]
    cmd = [
        ".venv/bin/python",
        "scripts/run_with_mlx_memory.py",
        "scripts/generate.py",
        item.prompt,
        "--checkpoint",
        defaults["checkpoint"],
        "--transformer",
        defaults["transformer"],
        "--text-encoder",
        defaults["text_encoder"],
        "--low-memory",
        "--stream-blocks",
        "--width",
        str(item.width),
        "--height",
        str(item.height),
        "--duration",
        str(item.duration_seconds),
        "--steps",
        str(item.steps_sigma_points),
        "--seed",
        str(item.seed),
        "--ffmpeg",
        defaults["ffmpeg"],
        "--require-muxed-mp4",
        "--output",
        str(item.media_path),
    ]
    if item.turbo_lora_required:
        cmd.extend(["--turbo-lora", defaults["turbo_lora"]])
    return cmd


def validate_media(path: Path, manifest: dict[str, Any], run_dir: Path, label: str) -> dict[str, Any]:
    defaults = manifest["shared_generation_defaults"]
    logs = run_dir / "logs"
    ffprobe_json = logs / f"ffprobe_{label}.json"
    ffprobe_stderr = logs / f"ffprobe_{label}.stderr"
    probe_cmd = [
        defaults["ffprobe"],
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    probe = run_command(probe_cmd, stdout_path=ffprobe_json, stderr_path=ffprobe_stderr)
    media_ok = False
    video_streams: list[dict[str, Any]] = []
    audio_streams: list[dict[str, Any]] = []
    probe_payload: dict[str, Any] = {}
    if probe["exit_code"] == 0:
        try:
            probe_payload = json.loads(ffprobe_json.read_text())
            streams = probe_payload.get("streams") or []
            video_streams = [s for s in streams if s.get("codec_type") == "video"]
            audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
            media_ok = bool(video_streams and audio_streams)
        except Exception as exc:
            probe_payload = {"parse_error": repr(exc)}
    audio_activity: dict[str, Any] | None = None
    if media_ok:
        audio_raw = logs / f"audio_{label}.f32le"
        audio_stderr = logs / f"audio_{label}.stderr"
        decode_cmd = [
            defaults["ffmpeg"],
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            str(audio_raw),
        ]
        decode = run_command(decode_cmd, stdout_path=logs / f"audio_{label}.stdout", stderr_path=audio_stderr)
        audio_activity = {"decode": decode, "activity_ok": False}
        if decode["exit_code"] == 0 and audio_raw.exists() and audio_raw.stat().st_size % 4 == 0:
            import numpy as np
            from minimax_h3_mlx.media_metrics import audio_activity_metrics

            audio = np.fromfile(audio_raw, dtype="<f4").astype("float64")
            stats = audio_activity_metrics(audio, sample_rate_hz=16000)
            audio_activity.update(stats)
            audio_activity["activity_ok"] = bool(stats["activity_ok"])
    return {
        "path": rel(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "ffprobe": probe,
        "ffprobe_ok_audio_video": media_ok,
        "video_stream_count": len(video_streams),
        "audio_stream_count": len(audio_streams),
        "first_video_stream": video_streams[0] if video_streams else None,
        "first_audio_stream": audio_streams[0] if audio_streams else None,
        "audio_activity": audio_activity,
        "media_valid": bool(media_ok and (audio_activity is None or audio_activity.get("activity_ok") is True)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"at": utc_now(), **payload}, sort_keys=True) + "\n")


def write_generation_metadata(item: MatrixItem, payload: dict[str, Any]) -> None:
    write_json(item.metadata_path, payload)


def run_metrics_for_pairs(manifest: dict[str, Any], matrix_records: dict[tuple[str, str], dict[str, Any]], run_dir: Path,
                          *, width: int, height: int) -> list[dict[str, Any]]:
    defaults = manifest["shared_generation_defaults"]
    references = "4bit_non_turbo_9sigma"
    candidates = ["4bit_turbo_5sigma", "4bit_non_turbo_5sigma_diagnostic_only"]
    results: list[dict[str, Any]] = []
    logs = run_dir / "logs"
    for case in manifest["cases"]:
        case_id = case["case_id"]
        seed = int(case["seeds"][0])
        ref = matrix_records.get((case_id, references))
        if not ref or not ref.get("media_valid"):
            results.append({"case_id": case_id, "seed": seed, "status": "skipped_missing_valid_reference", "reference_condition_id": references})
            continue
        for candidate_id in candidates:
            cand = matrix_records.get((case_id, candidate_id))
            out = pair_metric_path(
                manifest,
                case_id=case_id,
                seed=seed,
                width=width,
                height=height,
                reference_condition_id=references,
                candidate_condition_id=candidate_id,
            )
            if not cand or not cand.get("media_valid"):
                results.append({
                    "case_id": case_id,
                    "seed": seed,
                    "reference_condition_id": references,
                    "candidate_condition_id": candidate_id,
                    "status": "skipped_missing_valid_candidate",
                    "out": rel(out),
                })
                continue
            label = f"metrics_{case_id}_{candidate_id}"
            cmd = [
                "./.venv/bin/python",
                "scripts/eval_media_pair.py",
                "--reference",
                str(ROOT / ref["path"]),
                "--candidate",
                str(ROOT / cand["path"]),
                "--reference-label",
                references,
                "--candidate-label",
                candidate_id,
                "--ffmpeg",
                defaults["ffmpeg"],
                "--ffprobe",
                defaults["ffprobe"],
                "--out",
                str(out),
            ]
            run = run_command(cmd, stdout_path=logs / f"{label}.stdout", stderr_path=logs / f"{label}.stderr")
            results.append({
                "case_id": case_id,
                "seed": seed,
                "reference_condition_id": references,
                "candidate_condition_id": candidate_id,
                "status": "ok" if run["exit_code"] == 0 and out.exists() else "failed",
                "out": rel(out),
                "command": run,
            })
    return results


def write_blind_table(manifest: dict[str, Any], records: list[dict[str, Any]], *, width: int, height: int) -> Path:
    path = blind_table_path(manifest, width=width, height=height)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["blind_id", "suite_id", "case_id", "seed", "width", "height", "media_path", "condition_id_unblind_after_review"],
        )
        writer.writeheader()
        counter = 1
        for record in records:
            if not record.get("media_valid"):
                continue
            writer.writerow(
                {
                    "blind_id": f"Q{counter:03d}",
                    "suite_id": manifest["suite_id"],
                    "case_id": record["case_id"],
                    "seed": record["seed"],
                    "width": width,
                    "height": height,
                    "media_path": record["path"],
                    "condition_id_unblind_after_review": record["condition_id"],
                }
            )
            counter += 1
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--suite-root", default=str(DEFAULT_SUITE_ROOT))
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="write the plan and preflight shape, but do not run preflight/generation/metrics")
    parser.add_argument("--resume-existing", action="store_true", help="validate existing MP4s instead of regenerating them when present")
    parser.add_argument("--max-generations", type=int, default=None, help="optional cap for supervised chunked execution")
    parser.add_argument("--continue-on-failure", action="store_true", help="continue after non-OOM generation/validation failures")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest_path = manifest_path if manifest_path.is_absolute() else ROOT / manifest_path
    manifest = load_manifest(manifest_path)
    matrix = build_matrix(manifest, width=args.width, height=args.height)
    width = matrix[0].width
    height = matrix[0].height
    run_dir = Path(args.suite_root)
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    run_dir = run_dir / manifest["suite_id"] / "runs" / args.run_id
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "created_at": utc_now(),
        "manifest": rel(manifest_path),
        "suite_id": manifest["suite_id"],
        "width": width,
        "height": height,
        "duration_seconds": float(manifest["shared_generation_defaults"]["duration_seconds"]),
        "matrix_count": len(matrix),
        "conditions": [condition["condition_id"] for condition in manifest["planned_initial_conditions"]],
        "case_count": len(manifest["cases"]),
        "first_seed_only": True,
        "generation_schedule_contract": {
            "steps_argument_is_sigma_points": True,
            "denoiser_evaluations": "steps_sigma_points - 1",
            "turbo_5sigma_nfe": 4,
            "non_turbo_9sigma_nfe": 8,
        },
        "items": [
            {
                "case_id": item.case_id,
                "coverage_axis": item.coverage_axis,
                "seed": item.seed,
                "condition_id": item.condition_id,
                "steps_sigma_points": item.steps_sigma_points,
                "denoiser_evaluations": item.denoiser_evaluations,
                "turbo_lora_required": item.turbo_lora_required,
                "media_path": rel(item.media_path),
                "metadata_path": rel(item.metadata_path),
                "command": generation_command(item, manifest),
            }
            for item in matrix
        ],
        "expected_pair_metrics": 2 * len(manifest["cases"]),
        "summary_path": rel(summary_path),
        "events_path": rel(events_path),
    }
    write_json(run_dir / "plan.json", plan)
    append_event(events_path, {"event": "plan_written", "path": rel(run_dir / "plan.json"), "matrix_count": len(matrix)})
    if args.dry_run:
        summary = {**plan, "status": "dry_run_plan_only", "completed_generations": 0, "metrics": []}
        write_json(summary_path, summary)
        print(summary_path)
        return 0

    preflight = run_preflight(manifest, run_dir)
    write_json(run_dir / "preflight.json", preflight)
    append_event(events_path, {"event": "preflight", "ok_for_generation": preflight["ok_for_generation"], "path": rel(run_dir / "preflight.json")})
    if not preflight["ok_for_generation"]:
        summary = {**plan, "status": "blocked_preflight", "preflight": preflight, "completed_generations": 0, "metrics": []}
        write_json(summary_path, summary)
        print(summary_path)
        return 2

    records: list[dict[str, Any]] = []
    record_by_case_condition: dict[tuple[str, str], dict[str, Any]] = {}
    status = "ok"
    generation_budget = len(matrix) if args.max_generations is None else min(len(matrix), max(0, args.max_generations))
    for index, item in enumerate(matrix[:generation_budget], start=1):
        label = f"{index:02d}_{item.case_id}_{item.condition_id}_seed{item.seed}"
        append_event(events_path, {"event": "generation_start", "label": label, "media_path": rel(item.media_path)})
        gen_record: dict[str, Any] = {
            "case_id": item.case_id,
            "coverage_axis": item.coverage_axis,
            "seed": item.seed,
            "condition_id": item.condition_id,
            "steps_sigma_points": item.steps_sigma_points,
            "denoiser_evaluations": item.denoiser_evaluations,
            "turbo_lora_required": item.turbo_lora_required,
            "path": rel(item.media_path),
            "metadata_path": rel(item.metadata_path),
            "started_at": utc_now(),
        }
        if args.resume_existing and item.media_path.exists() and item.media_path.stat().st_size > 0:
            gen_record["generation"] = {"status": "skipped_existing", "cmd": generation_command(item, manifest)}
        else:
            item.media_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path = run_dir / "logs" / f"generate_{label}.stdout"
            stderr_path = run_dir / "logs" / f"generate_{label}.stderr"
            gen_record["generation"] = run_command(generation_command(item, manifest), stdout_path=stdout_path, stderr_path=stderr_path)
        if gen_record["generation"].get("exit_code", 0) not in (0, None):
            gen_record["status"] = "failed_generation"
            gen_record["media_valid"] = False
            status = "failed_generation"
            records.append(gen_record)
            record_by_case_condition[(item.case_id, item.condition_id)] = gen_record
            write_generation_metadata(item, gen_record)
            append_event(events_path, {"event": "generation_failed", "label": label, "oom_or_killed_marker": gen_record["generation"].get("oom_or_killed_marker")})
            if gen_record["generation"].get("oom_or_killed_marker") or not args.continue_on_failure:
                break
            continue
        validation = validate_media(item.media_path, manifest, run_dir, label)
        gen_record["validation"] = validation
        gen_record["media_valid"] = validation["media_valid"]
        gen_record["status"] = "ok" if validation["media_valid"] else "failed_media_validation"
        gen_record["completed_at"] = utc_now()
        records.append(gen_record)
        record_by_case_condition[(item.case_id, item.condition_id)] = gen_record
        write_generation_metadata(item, gen_record)
        append_event(events_path, {"event": "generation_complete", "label": label, "status": gen_record["status"], "media_valid": gen_record["media_valid"]})
        if not validation["media_valid"]:
            status = "failed_media_validation"
            if not args.continue_on_failure:
                break

    metrics: list[dict[str, Any]] = []
    if records:
        metrics = run_metrics_for_pairs(manifest, record_by_case_condition, run_dir, width=width, height=height)
        for metric in metrics:
            append_event(events_path, {"event": "pair_metric", **{k: metric.get(k) for k in ("case_id", "candidate_condition_id", "status", "out")}})
        blind_path = write_blind_table(manifest, records, width=width, height=height)
    else:
        blind_path = blind_table_path(manifest, width=width, height=height)
    if status == "ok" and len(records) != len(matrix):
        status = "partial_max_generations" if generation_budget < len(matrix) else "partial_incomplete"

    summary = {
        **plan,
        "status": status,
        "preflight": preflight,
        "completed_generations": len(records),
        "valid_media_count": sum(1 for record in records if record.get("media_valid")),
        "records": records,
        "metrics": metrics,
        "blind_review_table": rel(blind_path),
        "claim_boundary": "No lossless, near-lossless, best-Pareto, or promotion conclusion is made by this runner; Reviewer sign-off is required.",
    }
    write_json(summary_path, summary)
    append_event(events_path, {"event": "summary", "status": status, "path": rel(summary_path)})
    print(summary_path)
    return 0 if status in {"ok", "partial_max_generations"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
