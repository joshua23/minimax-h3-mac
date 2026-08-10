#!/usr/bin/env python3
"""Build/install the package in a clean room and run the installed media smoke.

The smoke deliberately uses a fresh HOME, cache, and venv, clears PYTHONPATH,
runs from outside the development checkout, and invokes installed console scripts.
It does not load models or run MiniMax-H3 generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.media_tools import resolve_media_tool  # noqa: E402

SCHEMA_VERSION = 1
OK = 0
BLOCKED = 66
VALIDATION_FAILURE = 3


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: int = 120,
    capture_limit: int | None = 4000,
) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=None if env is None else dict(env),
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        stdout = proc.stdout if capture_limit is None else proc.stdout[-capture_limit:]
        stderr = proc.stderr if capture_limit is None else proc.stderr[-capture_limit:]
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "wall_seconds": time.time() - started,
            "stdout": stdout,
            "stderr": stderr,
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": 124,
            "wall_seconds": time.time() - started,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _build_artifacts(python: str, run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dist_dir = run_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    uv = shutil.which("uv")
    if uv:
        record = _run([uv, "build", "--sdist", "--wheel", "--out-dir", str(dist_dir)], cwd=ROOT, timeout=180)
        commands.append(record)
        if record["returncode"] != 0:
            return commands, []
    else:
        wheel_record = _run([python, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(dist_dir), "."], cwd=ROOT, timeout=180)
        commands.append(wheel_record)
        sdist_record = _run(
            [python, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:", "--dest", str(dist_dir), "."],
            cwd=ROOT,
            timeout=180,
        )
        commands.append(sdist_record)
        if wheel_record["returncode"] != 0 or sdist_record["returncode"] != 0:
            return commands, []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(dist_dir.iterdir()):
        if path.suffix == ".whl" or path.name.endswith(".tar.gz"):
            artifacts.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "kind": "wheel" if path.suffix == ".whl" else "sdist",
                }
            )
    return commands, artifacts


def _existing_static_ffmpeg_binaries() -> tuple[Path, Path, dict[str, Any]] | None:
    try:
        import static_ffmpeg.run as static_run  # type: ignore

        exe_dir = Path(static_run.get_platform_dir())
    except Exception:
        return None
    suffix = ".exe" if sys.platform == "win32" else ""
    ffmpeg = exe_dir / f"ffmpeg{suffix}"
    ffprobe = exe_dir / f"ffprobe{suffix}"
    crumb = exe_dir / "installed.crumb"
    if ffmpeg.exists() and ffprobe.exists() and os.access(ffmpeg, os.X_OK) and os.access(ffprobe, os.X_OK):
        return ffmpeg, ffprobe, {
            "source": "static_ffmpeg_existing_binary",
            "package_dir": str(exe_dir),
            "installed_crumb": crumb.read_text(errors="replace")[:500] if crumb.exists() else None,
            "redistribution_note": "Copied into the temporary smoke provider directory only; not bundled into the MiniMax-H3 package artifact.",
        }
    return None


def _copy_provider(ffmpeg: Path, ffprobe: Path, provider_dir: Path, provider_meta: dict[str, Any]) -> dict[str, Any]:
    bin_dir = provider_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    out_ffmpeg = bin_dir / "ffmpeg"
    out_ffprobe = bin_dir / "ffprobe"
    shutil.copy2(ffmpeg, out_ffmpeg)
    shutil.copy2(ffprobe, out_ffprobe)
    for path in (out_ffmpeg, out_ffprobe):
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return {
        **provider_meta,
        "bin_dir": str(bin_dir),
        "ffmpeg": {"path": str(out_ffmpeg), "sha256": _sha256(out_ffmpeg), "size_bytes": out_ffmpeg.stat().st_size},
        "ffprobe": {"path": str(out_ffprobe), "sha256": _sha256(out_ffprobe), "size_bytes": out_ffprobe.stat().st_size},
    }


def _prepare_provider(args: argparse.Namespace, run_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    if args.ffmpeg and args.ffprobe:
        ffmpeg = Path(args.ffmpeg).expanduser()
        ffprobe = Path(args.ffprobe).expanduser()
        if not ffmpeg.is_absolute():
            ffmpeg = (Path.cwd() / ffmpeg).resolve()
        if not ffprobe.is_absolute():
            ffprobe = (Path.cwd() / ffprobe).resolve()
        if ffmpeg.exists() and ffprobe.exists() and os.access(ffmpeg, os.X_OK) and os.access(ffprobe, os.X_OK):
            return _copy_provider(ffmpeg, ffprobe, run_dir / "media-provider", {"source": "explicit_operator_paths"}), issues
        issues.append("explicit --ffmpeg/--ffprobe paths were not both executable")
        return None, issues

    static_pair = _existing_static_ffmpeg_binaries()
    if static_pair is not None:
        ffmpeg, ffprobe, meta = static_pair
        return _copy_provider(ffmpeg, ffprobe, run_dir / "media-provider", meta), issues

    if args.allow_path_provider:
        ffmpeg_res = resolve_media_tool("ffmpeg", None, cwd=ROOT, allow_path=True)
        ffprobe_res = resolve_media_tool("ffprobe", None, cwd=ROOT, allow_path=True)
        if ffmpeg_res.ok and ffprobe_res.ok and ffmpeg_res.path and ffprobe_res.path:
            return _copy_provider(
                Path(ffmpeg_res.path),
                Path(ffprobe_res.path),
                run_dir / "media-provider",
                {"source": "path_provider_copied", "resolver": {"ffmpeg": ffmpeg_res.to_dict(), "ffprobe": ffprobe_res.to_dict()}},
            ), issues
        issues.append("PATH provider fallback was allowed but ffmpeg/ffprobe did not both resolve")
    else:
        issues.append("no explicit provider and no already-installed static_ffmpeg binary; PATH provider fallback is disabled")
    return None, issues


def _venv_bin(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _sanitize_env(base_env: Mapping[str, str], *, home: Path, cache: Path, venv_dir: Path, include_uv: bool) -> dict[str, str]:
    env = {key: value for key, value in base_env.items() if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(cache)
    env["PYTHONPATH"] = ""
    path_parts = [str(_venv_bin(venv_dir))]
    uv_path = shutil.which("uv") if include_uv else None
    if uv_path:
        path_parts.append(str(Path(uv_path).parent))
    path_parts.extend(["/usr/bin", "/bin", "/usr/sbin", "/sbin"])
    env["PATH"] = os.pathsep.join(dict.fromkeys(path_parts))
    return env


def _parse_json(stdout: str) -> Any | None:
    try:
        return json.loads(stdout)
    except Exception:
        return None


def run_cleanroom_smoke(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or f"cleanroom-{_utc_stamp()}"
    evidence_dir = Path(args.out_dir or ROOT / "out" / "release_cleanroom_smoke" / run_id).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix=f"{run_id}-"))
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = evidence_dir / "report.json"
    commands: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "run_release_cleanroom_smoke.py",
        "ok": False,
        "exit_code": OK,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "evidence_dir": str(evidence_dir),
        "report_path": str(report_path),
        "created_at": _utc_stamp(),
        "source_checkout": str(ROOT),
        "cleanroom_outside_checkout": not (run_dir == ROOT or ROOT in run_dir.parents),
        "python": {
            "builder_executable": args.python,
            "builder_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "commands": commands,
        "artifacts": [],
        "cleanroom": {},
        "provider": None,
        "results": {},
        "issues": [],
    }

    provider, provider_issues = _prepare_provider(args, run_dir)
    report["provider"] = provider
    report["issues"].extend(provider_issues)
    if provider is None:
        report["exit_code"] = BLOCKED
        _write_json(report_path, report)
        return report

    build_commands, artifacts = _build_artifacts(args.python, evidence_dir)
    commands.extend(build_commands)
    report["artifacts"] = artifacts
    wheels = [item for item in artifacts if item["kind"] == "wheel"]
    sdists = [item for item in artifacts if item["kind"] == "sdist"]
    if not wheels or not sdists:
        report["exit_code"] = BLOCKED
        report["issues"].append("package build did not produce both a wheel and an sdist")
        _write_json(report_path, report)
        return report

    home = run_dir / "home"
    cache = run_dir / "cache"
    output_dir = run_dir / "outputs"
    venv_dir = run_dir / "venv"
    for directory in (home, cache, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    venv_record = _run([args.python, "-m", "venv", str(venv_dir)], cwd=run_dir, timeout=120)
    commands.append(venv_record)
    if venv_record["returncode"] != 0:
        report["exit_code"] = BLOCKED
        report["issues"].append("temporary venv creation failed")
        _write_json(report_path, report)
        return report

    env = _sanitize_env(os.environ, home=home, cache=cache, venv_dir=venv_dir, include_uv=not args.exclude_uv_from_path)
    venv_python = str(_venv_bin(venv_dir) / "python")
    install_record = _run([venv_python, "-m", "pip", "install", "--no-deps", wheels[0]["path"]], cwd=run_dir, env=env, timeout=120)
    commands.append(install_record)
    if install_record["returncode"] != 0:
        report["exit_code"] = BLOCKED
        report["issues"].append("wheel install into temporary venv failed")
        _write_json(report_path, report)
        return report

    report["cleanroom"] = {
        "home": str(home),
        "cache": str(cache),
        "venv": str(venv_dir),
        "cwd_for_installed_commands": str(run_dir),
        "sanitized_path": env["PATH"],
        "pythonpath": env.get("PYTHONPATH", ""),
        "path_ffmpeg": shutil.which("ffmpeg", path=env["PATH"]),
        "path_ffprobe": shutil.which("ffprobe", path=env["PATH"]),
        "checkout_removed_from_cwd": not (run_dir == ROOT or ROOT in run_dir.parents),
    }

    import_record = _run(
        [
            venv_python,
            "-c",
            "import json, minimax_h3_mlx; print(json.dumps({'package_file': minimax_h3_mlx.__file__}))",
        ],
        cwd=run_dir,
        env=env,
        timeout=30,
    )
    commands.append(import_record)
    imported = _parse_json(import_record["stdout"])
    report["results"]["package_import"] = imported
    package_file = Path(str(imported.get("package_file", ""))).resolve() if imported else ROOT
    if not imported or package_file == ROOT or ROOT in package_file.parents:
        report["exit_code"] = BLOCKED
        report["issues"].append("installed package import did not prove checkout isolation")
        _write_json(report_path, report)
        return report

    console_bin = _venv_bin(venv_dir)
    ffmpeg_path = provider["ffmpeg"]["path"]
    ffprobe_path = provider["ffprobe"]["path"]
    doctor_cmd = [
        str(console_bin / "minimax-h3-doctor"),
        "--json",
        "--min-free-gib",
        "0",
        "--min-memory-gib",
        "0",
        "--cache-dir",
        str(cache / "minimax-h3"),
        "--output-dir",
        str(output_dir),
        "--ffmpeg",
        ffmpeg_path,
        "--ffprobe",
        ffprobe_path,
        "--no-strict-assets",
    ]
    doctor_record = _run(doctor_cmd, cwd=run_dir, env=env, timeout=60, capture_limit=None)
    commands.append(doctor_record)
    report["results"]["doctor"] = {
        "returncode": doctor_record["returncode"],
        "json": _parse_json(doctor_record["stdout"]),
    }

    generate_help_record = _run([str(console_bin / "minimax-h3-generate"), "--help"], cwd=run_dir, env=env, timeout=30)
    commands.append(generate_help_record)
    report["results"]["generate_help"] = {"returncode": generate_help_record["returncode"]}

    media_report_path = run_dir / "media_smoke.json"
    media_output = output_dir / "synthetic-media-smoke.mp4"
    media_cmd = [
        str(console_bin / "minimax-h3-media-smoke"),
        "--output",
        str(media_output),
        "--report",
        str(media_report_path),
        "--work-dir",
        str(run_dir / "media-work"),
        "--ffmpeg",
        ffmpeg_path,
        "--ffprobe",
        ffprobe_path,
        "--no-system-path",
    ]
    media_record = _run(media_cmd, cwd=run_dir, env=env, timeout=90)
    commands.append(media_record)
    media_json = json.loads(media_report_path.read_text()) if media_report_path.exists() else _parse_json(media_record["stdout"])
    report["results"]["media_smoke"] = {
        "returncode": media_record["returncode"],
        "report_path": str(media_report_path),
        "json": media_json,
    }

    ok = (
        install_record["returncode"] == 0
        and import_record["returncode"] == 0
        and generate_help_record["returncode"] == 0
        and doctor_record["returncode"] == 0
        and media_record["returncode"] == 0
        and bool(media_json and media_json.get("ok"))
    )
    report["ok"] = ok
    report["exit_code"] = OK if ok else VALIDATION_FAILURE
    if not ok:
        report["issues"].append("one or more clean-room installed-package checks failed")
    _write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/install MiniMax-H3 package in a temp clean room and prove media mux/probe.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", default=None, help="persistent evidence/report directory; defaults under out/release_cleanroom_smoke")
    parser.add_argument("--work-dir", default=None, help="temporary clean-room HOME/cache/venv parent; defaults outside the checkout")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ffmpeg", default=None, help="explicit provider ffmpeg path to copy into the clean-room provider dir")
    parser.add_argument("--ffprobe", default=None, help="explicit provider ffprobe path to copy into the clean-room provider dir")
    parser.add_argument("--allow-path-provider", action="store_true", help="last-resort provider discovery from PATH; disabled by default")
    parser.add_argument("--exclude-uv-from-path", action="store_true", help="do not include the host uv directory in the sanitized PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_cleanroom_smoke(args)
    print(json.dumps({"ok": report["ok"], "exit_code": report["exit_code"], "report_path": report["report_path"]}, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
