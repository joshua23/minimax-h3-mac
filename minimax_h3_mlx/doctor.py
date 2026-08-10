"""Deterministic release doctor for the MiniMax-H3 Apple Silicon package.

The doctor checks only lightweight release prerequisites. It never runs generation,
loads model tensors, downloads assets, or creates model/cache directories. Model
asset validation, when requested, is delegated to the existing header-only
``asset_preflight`` scanner.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from minimax_h3_mlx.media_tools import (
    FFMPEG_ENV_VAR,
    FFPROBE_ENV_VAR,
    PROJECT_LOCAL_MEDIA_TOOL_DIRS,
    resolve_media_tool,
)

OK = 0
CONTRACT_FAILURE = 1

CHECKPOINT_ENV_VAR = "MINIMAX_H3_CHECKPOINT"
TRANSFORMER_ENV_VAR = "MINIMAX_H3_TRANSFORMER"
TEXT_ENCODER_ENV_VAR = "MINIMAX_H3_TEXT_ENCODER"
TURBO_LORA_ENV_VAR = "MINIMAX_H3_TURBO_LORA"
CACHE_ENV_VAR = "MINIMAX_H3_CACHE_DIR"
OUTPUT_ENV_VAR = "MINIMAX_H3_OUTPUT_DIR"
LICENSE_ENV_VAR = "MINIMAX_H3_ACCEPT_LICENSE"

DEFAULT_MIN_MACOS = "15.0"
DEFAULT_MIN_PYTHON = "3.11"
DEFAULT_MIN_MEMORY_GIB = 24.0
DEFAULT_MIN_FREE_GIB = 100.0
_TRUE_VALUES = {"1", "true", "yes", "y", "on", "accepted"}
_FORBIDDEN_PATH_PARTS = {".argus-skill"}


@dataclass(frozen=True)
class DoctorConfig:
    """Inputs that define the release contract check."""

    cwd: Path = field(default_factory=lambda: Path.cwd())
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    checkpoint: str | None = None
    transformer: str | None = None
    text_encoder: str | None = None
    turbo_lora: str | None = None
    cache_dir: str | None = None
    output_dir: str | None = None
    asset_roots: tuple[str, ...] = ()
    ffmpeg: str | None = None
    ffprobe: str | None = None
    min_macos: str = DEFAULT_MIN_MACOS
    min_python: str = DEFAULT_MIN_PYTHON
    min_memory_gib: float = DEFAULT_MIN_MEMORY_GIB
    min_free_gib: float = DEFAULT_MIN_FREE_GIB
    require_model_paths: bool = False
    require_license_acceptance: bool = False
    strict_assets: bool = True
    asset_manifest: str | None = None
    asset_cache_dir: str | None = None
    manifest_assets: tuple[str, ...] = ()
    strict_manifest_assets: bool = False


@dataclass(frozen=True)
class ProbeSet:
    """Injectable probes used by tests to avoid depending on the host machine."""

    system: Callable[[], str] = platform.system
    machine: Callable[[], str] = platform.machine
    mac_ver: Callable[[], tuple[str, tuple[str, str, str], str]] = platform.mac_ver
    python_version: Callable[[], str] = platform.python_version
    python_executable: Callable[[], str] = lambda: sys.executable
    python_prefix: Callable[[], str] = lambda: sys.prefix
    python_base_prefix: Callable[[], str] = lambda: sys.base_prefix
    which: Callable[[str], str | None] = shutil.which
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage
    memory_bytes: Callable[[], int | None] = lambda: _host_memory_bytes()
    path_exists: Callable[[Path], bool] = lambda path: path.exists()
    path_is_dir: Callable[[Path], bool] = lambda path: path.is_dir()
    path_is_file: Callable[[Path], bool] = lambda path: path.is_file()
    path_executable: Callable[[Path], bool] = lambda path: os.access(path, os.X_OK)
    path_writable: Callable[[Path], bool] = lambda path: os.access(path, os.W_OK)
    geteuid: Callable[[], int | None] = lambda: os.geteuid() if hasattr(os, "geteuid") else None


def _host_memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=False,
                text=True,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            return None
        if proc.returncode == 0:
            try:
                return int(proc.stdout.strip())
            except ValueError:
                return None
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size)
        except (OSError, ValueError, TypeError):
            return None
    return None


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts or [0])


def _version_at_least(observed: str, required: str) -> bool:
    left = list(_version_tuple(observed))
    right = list(_version_tuple(required))
    width = max(len(left), len(right))
    left.extend([0] * (width - len(left)))
    right.extend([0] * (width - len(right)))
    return tuple(left) >= tuple(right)


def _gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 ** 3)


def _new_check(check_id: str, title: str) -> dict[str, Any]:
    return {"id": check_id, "title": title, "ok": True, "issues": [], "observed": {}}


def _add_issue(
    check: dict[str, Any],
    *,
    severity: str,
    code: str,
    message: str,
    fatal: bool,
    **details: Any,
) -> None:
    issue: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
        "fatal": bool(fatal),
    }
    issue.update({key: value for key, value in details.items() if value is not None})
    check["issues"].append(issue)
    if fatal:
        check["ok"] = False


def _resolve_path(value: str | None, cwd: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path


def _path_for_report(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd.resolve()))
    except Exception:
        return str(path)


def _nearest_existing_parent(path: Path, probes: ProbeSet) -> Path | None:
    candidate = path if probes.path_is_dir(path) else path.parent
    while True:
        if probes.path_exists(candidate):
            return candidate
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent


def _is_forbidden_argus_path(path: Path) -> bool:
    return bool(_FORBIDDEN_PATH_PARTS.intersection(path.parts))


def _configured_value(
    *,
    explicit: str | None,
    env: Mapping[str, str],
    env_var: str,
    default: str | None = None,
) -> tuple[str | None, str]:
    if explicit is not None:
        return explicit, "argument"
    if env.get(env_var):
        return env[env_var], f"env:{env_var}"
    return default, "default" if default is not None else "unset"


def _license_accepted(env: Mapping[str, str]) -> bool:
    return env.get(LICENSE_ENV_VAR, "").strip().lower() in _TRUE_VALUES


def _check_platform(config: DoctorConfig, probes: ProbeSet) -> dict[str, Any]:
    check = _new_check("platform", "Apple Silicon macOS contract")
    system = probes.system()
    machine = probes.machine()
    mac_version = probes.mac_ver()[0] or ""
    check["observed"] = {
        "system": system,
        "machine": machine,
        "macos_version": mac_version,
        "minimum_macos_version": config.min_macos,
        "requires_native_arm64": True,
        "requires_rosetta": False,
    }
    if system != "Darwin":
        _add_issue(
            check,
            severity="error",
            code="unsupported_os",
            message="MiniMax-H3 release package is contracted for macOS on Apple Silicon.",
            fatal=True,
            observed=system,
            expected="Darwin",
        )
    if machine != "arm64":
        _add_issue(
            check,
            severity="error",
            code="unsupported_architecture",
            message="Python must run as native arm64; Rosetta/x86_64 is outside the release contract.",
            fatal=True,
            observed=machine,
            expected="arm64",
        )
    if system == "Darwin" and (not mac_version or not _version_at_least(mac_version, config.min_macos)):
        _add_issue(
            check,
            severity="error",
            code="macos_too_old",
            message="macOS is older than the release contract minimum.",
            fatal=True,
            observed=mac_version or None,
            expected=f">= {config.min_macos}",
        )
    return check


def _check_memory(config: DoctorConfig, probes: ProbeSet) -> dict[str, Any]:
    check = _new_check("memory", "Unified memory contract")
    total = probes.memory_bytes()
    total_gib = _gib(total)
    check["observed"] = {"total_bytes": total, "total_gib": total_gib, "minimum_gib": config.min_memory_gib}
    if total_gib is None:
        _add_issue(
            check,
            severity="error",
            code="memory_probe_failed",
            message="Could not determine physical/unified memory without loading models.",
            fatal=True,
        )
    elif total_gib + 1e-6 < config.min_memory_gib:
        _add_issue(
            check,
            severity="error",
            code="insufficient_memory",
            message="Machine has less unified memory than the Apple M4 Pro 24GB release contract.",
            fatal=True,
            observed_gib=round(total_gib, 3),
            expected_gib=config.min_memory_gib,
        )
    return check


def _check_python(config: DoctorConfig, probes: ProbeSet) -> dict[str, Any]:
    check = _new_check("python", "Python runtime contract")
    version = probes.python_version()
    executable = probes.python_executable()
    prefix = probes.python_prefix()
    base_prefix = probes.python_base_prefix()
    in_venv = prefix != base_prefix or bool(config.env.get("VIRTUAL_ENV"))
    check["observed"] = {
        "version": version,
        "executable": executable,
        "prefix": prefix,
        "base_prefix": base_prefix,
        "in_virtual_environment": in_venv,
        "minimum_python": config.min_python,
    }
    if not _version_at_least(version, config.min_python):
        _add_issue(
            check,
            severity="error",
            code="python_too_old",
            message="Python version is older than the package contract.",
            fatal=True,
            observed=version,
            expected=f">= {config.min_python}",
        )
    if not in_venv:
        _add_issue(
            check,
            severity="warning",
            code="python_not_in_virtualenv",
            message="Doctor is not running from an isolated virtual environment; release installs should avoid global Python pollution.",
            fatal=False,
        )
    return check


def _tool_result(
    name: str,
    *,
    explicit: str | None,
    env: Mapping[str, str],
    env_var: str | None,
    probes: ProbeSet,
    cwd: Path,
) -> dict[str, Any]:
    if name in {"ffmpeg", "ffprobe"}:
        python_bin = Path(probes.python_executable()).parent
        local_dirs = [python_bin]
        local_dirs.extend(cwd / directory for directory in PROJECT_LOCAL_MEDIA_TOOL_DIRS)
        resolution = resolve_media_tool(
            name,
            explicit,
            env=env,
            env_var=env_var,
            cwd=cwd,
            local_bin_dirs=local_dirs,
            allow_path=True,
            which=probes.which,
            path_exists=probes.path_exists,
            path_executable=probes.path_executable,
        )
        return resolution.to_dict()

    requested = explicit
    source = "argument" if explicit else None
    if requested is None and env_var and env.get(env_var):
        requested = env[env_var]
        source = f"env:{env_var}"
    if requested is not None:
        if "/" in requested:
            path = Path(requested).expanduser()
            if not path.is_absolute():
                path = cwd / path
            exists = probes.path_exists(path)
            executable = probes.path_executable(path) if exists else False
            return {
                "name": name,
                "requested": requested,
                "source": source,
                "path": str(path),
                "ok": bool(exists and executable),
                "exists": exists,
                "executable": executable,
            }
        resolved = probes.which(requested)
        return {"name": name, "requested": requested, "source": source, "path": resolved, "ok": resolved is not None}
    resolved = probes.which(name)
    return {"name": name, "requested": name, "source": "PATH", "path": resolved, "ok": resolved is not None}


def _check_tools(config: DoctorConfig, probes: ProbeSet) -> dict[str, Any]:
    check = _new_check("tools", "Required local executables")
    tools = [
        _tool_result("uv", explicit=None, env=config.env, env_var=None, probes=probes, cwd=config.cwd),
        _tool_result("ffmpeg", explicit=config.ffmpeg, env=config.env, env_var=FFMPEG_ENV_VAR, probes=probes, cwd=config.cwd),
        _tool_result("ffprobe", explicit=config.ffprobe, env=config.env, env_var=FFPROBE_ENV_VAR, probes=probes, cwd=config.cwd),
    ]
    check["observed"]["tools"] = tools
    for item in tools:
        if not item["ok"]:
            _add_issue(
                check,
                severity="error",
                code="missing_required_tool",
                message=f"Required executable {item['name']!r} was not found or is not executable.",
                fatal=True,
                tool=item["name"],
                requested=item.get("requested"),
                source=item.get("source"),
            )

    xcode_select = probes.which("xcode-select")
    xcode_record: dict[str, Any] = {"xcode_select": xcode_select, "ok": False, "path": None}
    if xcode_select is None:
        _add_issue(
            check,
            severity="error",
            code="missing_xcode_select",
            message="Xcode Command Line Tools are required but xcode-select is not on PATH.",
            fatal=True,
        )
    else:
        try:
            proc = probes.run(
                [xcode_select, "-p"],
                check=False,
                text=True,
                capture_output=True,
                timeout=10,
            )
        except Exception as exc:
            _add_issue(
                check,
                severity="error",
                code="xcode_select_failed",
                message="Could not query the Xcode Command Line Tools path.",
                fatal=True,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            clt_path = Path(proc.stdout.strip()) if proc.stdout.strip() else None
            xcode_record.update({"returncode": proc.returncode, "path": str(clt_path) if clt_path else None})
            if proc.returncode != 0 or clt_path is None:
                _add_issue(
                    check,
                    severity="error",
                    code="xcode_clt_unselected",
                    message="Xcode Command Line Tools are not selected.",
                    fatal=True,
                    stderr=proc.stderr.strip()[:500] if proc.stderr else None,
                )
            elif not probes.path_exists(clt_path):
                _add_issue(
                    check,
                    severity="error",
                    code="xcode_clt_path_missing",
                    message="xcode-select reports a Command Line Tools path that does not exist.",
                    fatal=True,
                    path=str(clt_path),
                )
            else:
                xcode_record["ok"] = True
    check["observed"]["xcode_clt"] = xcode_record
    return check


def _check_dependency_lock(config: DoctorConfig, probes: ProbeSet) -> dict[str, Any]:
    check = _new_check("dependency_lock", "Install metadata and uv lockfile")
    pyproject = config.cwd / "pyproject.toml"
    lockfile = config.cwd / "uv.lock"
    pyproject_exists = probes.path_is_file(pyproject)
    lock_exists = probes.path_is_file(lockfile)
    check["observed"] = {
        "pyproject": _path_for_report(pyproject, config.cwd),
        "pyproject_exists": pyproject_exists,
        "uv_lock": _path_for_report(lockfile, config.cwd),
        "uv_lock_exists": lock_exists,
    }
    if not pyproject_exists:
        _add_issue(
            check,
            severity="warning",
            code="pyproject_not_in_cwd",
            message="No pyproject.toml is visible from the current source/archive directory; lockfile check is not applicable.",
            fatal=False,
        )
    elif not lock_exists:
        _add_issue(
            check,
            severity="error",
            code="missing_uv_lock",
            message="pyproject.toml is present but uv.lock is missing; dependency resolution is not locked.",
            fatal=True,
        )
    return check


def _check_disk(config: DoctorConfig, probes: ProbeSet) -> dict[str, Any]:
    check = _new_check("disk", "Free disk-space contract")
    raw_paths = [
        ("cache_dir", config.cache_dir),
        ("output_dir", config.output_dir),
    ]
    seen: set[str] = set()
    volumes: list[dict[str, Any]] = []
    for name, raw in raw_paths:
        path = _resolve_path(raw, config.cwd)
        if path is None:
            continue
        parent = _nearest_existing_parent(path, probes)
        record = {
            "name": name,
            "path": _path_for_report(path, config.cwd),
            "nearest_existing_parent": str(parent) if parent else None,
            "free_gib": None,
            "minimum_free_gib": config.min_free_gib,
            "writable_parent": None,
        }
        if parent is None:
            _add_issue(
                check,
                severity="error",
                code="path_has_no_existing_parent",
                message="Configured cache/output path has no existing parent to inspect.",
                fatal=True,
                path=str(path),
                role=name,
            )
            volumes.append(record)
            continue
        key = str(parent.resolve()) if probes.path_exists(parent) else str(parent)
        record["writable_parent"] = probes.path_writable(parent)
        try:
            usage = probes.disk_usage(parent)
            free_gib = _gib(int(usage.free))
            record["free_gib"] = free_gib
        except Exception as exc:
            _add_issue(
                check,
                severity="error",
                code="disk_probe_failed",
                message="Could not determine free disk space for a configured path.",
                fatal=True,
                path=str(parent),
                error=f"{type(exc).__name__}: {exc}",
            )
            volumes.append(record)
            continue
        if not record["writable_parent"]:
            _add_issue(
                check,
                severity="error",
                code="path_parent_not_writable",
                message="Configured cache/output parent is not writable by the current user.",
                fatal=True,
                path=str(parent),
                role=name,
            )
        if key not in seen and free_gib is not None and free_gib + 1e-6 < config.min_free_gib:
            _add_issue(
                check,
                severity="error",
                code="insufficient_free_disk",
                message="Configured cache/output volume has less free space than the release contract minimum.",
                fatal=True,
                path=str(parent),
                observed_gib=round(free_gib, 3),
                expected_gib=config.min_free_gib,
            )
        seen.add(key)
        volumes.append(record)
    check["observed"]["volumes"] = volumes
    return check


def _check_paths(config: DoctorConfig, probes: ProbeSet) -> dict[str, Any]:
    check = _new_check("paths", "Configurable model/cache/output paths")
    entries: list[dict[str, Any]] = []
    model_roles = [
        ("checkpoint", config.checkpoint, CHECKPOINT_ENV_VAR),
        ("transformer", config.transformer, TRANSFORMER_ENV_VAR),
        ("text_encoder", config.text_encoder, TEXT_ENCODER_ENV_VAR),
        ("turbo_lora", config.turbo_lora, TURBO_LORA_ENV_VAR),
    ]
    for role, explicit, env_var in model_roles:
        value, source = _configured_value(explicit=explicit, env=config.env, env_var=env_var)
        path = _resolve_path(value, config.cwd)
        entry: dict[str, Any] = {"role": role, "source": source, "path": None, "exists": None}
        if path is None:
            if role == "turbo_lora":
                entry["optional"] = True
                entries.append(entry)
                continue
            severity = "error" if config.require_model_paths else "warning"
            fatal = bool(config.require_model_paths)
            _add_issue(
                check,
                severity=severity,
                code="model_path_not_configured",
                message=f"Model path {role!r} is not configured; doctor will not infer a local checkout/cache default.",
                fatal=fatal,
                role=role,
                env_var=env_var,
            )
            entries.append(entry)
            continue
        exists = probes.path_exists(path)
        entry.update({"path": _path_for_report(path, config.cwd), "exists": exists, "is_absolute": path.is_absolute()})
        if _is_forbidden_argus_path(path):
            _add_issue(
                check,
                severity="error",
                code="forbidden_argus_path",
                message="Configured release path points into Argus state, which is outside the package contract.",
                fatal=True,
                role=role,
                path=str(path),
            )
        if not exists:
            severity = "error" if config.require_model_paths and role != "turbo_lora" else "warning"
            fatal = bool(config.require_model_paths and role != "turbo_lora")
            _add_issue(
                check,
                severity=severity,
                code="configured_model_path_missing",
                message="Configured model path does not exist; no download is attempted by doctor.",
                fatal=fatal,
                role=role,
                path=str(path),
            )
        entries.append(entry)

    for role, raw in (("cache_dir", config.cache_dir), ("output_dir", config.output_dir)):
        path = _resolve_path(raw, config.cwd)
        entry = {"role": role, "source": "configured", "path": _path_for_report(path, config.cwd) if path else None}
        if path and _is_forbidden_argus_path(path):
            _add_issue(
                check,
                severity="error",
                code="forbidden_argus_path",
                message="Configured cache/output path points into Argus state, which is outside the package contract.",
                fatal=True,
                role=role,
                path=str(path),
            )
        entries.append(entry)
    check["observed"]["paths"] = entries
    return check


def _check_license(config: DoctorConfig) -> dict[str, Any]:
    check = _new_check("license", "Model license acceptance boundary")
    accepted = _license_accepted(config.env)
    check["observed"] = {
        "accepted_env_var": LICENSE_ENV_VAR,
        "accepted": accepted,
        "model_weights_bundled": False,
        "doctor_downloads_models": False,
    }
    if not accepted:
        _add_issue(
            check,
            severity="error" if config.require_license_acceptance else "warning",
            code="model_license_not_accepted",
            message="Model weights are not bundled; a later download flow must require the user to accept upstream licenses.",
            fatal=config.require_license_acceptance,
            env_var=LICENSE_ENV_VAR,
        )
    return check


def _check_assets(config: DoctorConfig) -> dict[str, Any]:
    check = _new_check("assets", "Header-only model asset preflight")
    if not config.asset_roots:
        check["observed"] = {"asset_roots": [], "preflight_run": False}
        _add_issue(
            check,
            severity="warning",
            code="asset_preflight_not_run",
            message="No --asset-root was provided; doctor did not inspect model asset headers.",
            fatal=False,
        )
        return check

    from minimax_h3_mlx.asset_preflight import run_preflight

    roots = [str(_resolve_path(root, config.cwd) or Path(root)) for root in config.asset_roots]
    result = run_preflight(roots, cwd=config.cwd)
    check["observed"] = {
        "asset_roots": roots,
        "preflight_run": True,
        "preflight_ok": result["ok"],
        "preflight_exit_code": result["exit_code"],
        "preflight_summary": result.get("summary", {}),
        "preflight_issues": result.get("issues", []),
    }
    if not result["ok"]:
        _add_issue(
            check,
            severity="error" if config.strict_assets else "warning",
            code="asset_preflight_failed",
            message="Header-only asset preflight reported model asset issues.",
            fatal=config.strict_assets,
            preflight_exit_code=result["exit_code"],
        )
    return check


def _check_asset_manifest(config: DoctorConfig) -> dict[str, Any]:
    check = _new_check("asset_manifest", "Versioned release asset manifest status")
    try:
        from minimax_h3_mlx.asset_manager import (
            asset_status_report,
            default_cache_dir as default_asset_cache_dir,
            load_manifest,
        )

        cache_dir = config.asset_cache_dir or config.cache_dir or str(default_asset_cache_dir(config.env))
        manifest = load_manifest(config.asset_manifest)
        report = asset_status_report(
            manifest,
            cache_dir,
            asset_ids=config.manifest_assets,
            hash_mode=False,
            include_preflight=False,
            cwd=config.cwd,
        )
    except Exception as exc:
        fallback_cache_dir = config.asset_cache_dir or config.cache_dir or "<unresolved>"
        check["observed"] = {"manifest": config.asset_manifest or "package:release_assets.json", "cache_dir": fallback_cache_dir}
        _add_issue(
            check,
            severity="error",
            code="asset_manifest_unavailable",
            message="Could not load or evaluate the release asset manifest.",
            fatal=True,
            error=f"{type(exc).__name__}: {exc}",
        )
        return check

    check["observed"] = {
        "manifest": config.asset_manifest or "package:release_assets.json",
        "manifest_version": report.get("manifest_version"),
        "cache_dir": cache_dir,
        "hash_mode": report.get("hash_mode"),
        "downloads_models": report.get("downloads_models"),
        "runs_generation": report.get("runs_generation"),
        "summary": report.get("summary", {}),
        "assets": report.get("assets", []),
        "issues": report.get("issues", []),
    }
    if not report.get("ok", False):
        _add_issue(
            check,
            severity="error" if config.strict_manifest_assets else "warning",
            code="manifest_assets_not_verified",
            message="One or more manifest assets are missing or invalid in the configured cache; doctor did not download or hash them.",
            fatal=config.strict_manifest_assets,
            asset_errors=report.get("summary", {}).get("errors"),
        )
    return check



def _check_user_privileges(probes: ProbeSet) -> dict[str, Any]:
    check = _new_check("privileges", "No-sudo execution contract")
    euid = probes.geteuid()
    check["observed"] = {"effective_uid": euid, "requires_root": False}
    if euid == 0:
        _add_issue(
            check,
            severity="error",
            code="running_as_root",
            message="Release install/generation must not require sudo or root execution.",
            fatal=True,
        )
    return check


def run_doctor(config: DoctorConfig | None = None, probes: ProbeSet | None = None) -> dict[str, Any]:
    """Run lightweight release checks and return a machine-readable report."""

    config = config or DoctorConfig()
    probes = probes or ProbeSet()
    cwd = config.cwd.resolve()
    config = DoctorConfig(
        cwd=cwd,
        env=config.env,
        checkpoint=config.checkpoint,
        transformer=config.transformer,
        text_encoder=config.text_encoder,
        turbo_lora=config.turbo_lora,
        cache_dir=config.cache_dir,
        output_dir=config.output_dir,
        asset_roots=config.asset_roots,
        ffmpeg=config.ffmpeg,
        ffprobe=config.ffprobe,
        min_macos=config.min_macos,
        min_python=config.min_python,
        min_memory_gib=config.min_memory_gib,
        min_free_gib=config.min_free_gib,
        require_model_paths=config.require_model_paths,
        require_license_acceptance=config.require_license_acceptance,
        strict_assets=config.strict_assets,
        asset_manifest=config.asset_manifest,
        asset_cache_dir=config.asset_cache_dir,
        manifest_assets=config.manifest_assets,
        strict_manifest_assets=config.strict_manifest_assets,
    )

    checks = [
        _check_platform(config, probes),
        _check_memory(config, probes),
        _check_python(config, probes),
        _check_user_privileges(probes),
        _check_tools(config, probes),
        _check_dependency_lock(config, probes),
        _check_paths(config, probes),
        _check_disk(config, probes),
        _check_license(config),
        _check_asset_manifest(config),
        _check_assets(config),
    ]
    issues = []
    for check in checks:
        for issue in check["issues"]:
            enriched = {"check": check["id"], **issue}
            issues.append(enriched)
    fatal_issues = [issue for issue in issues if issue.get("fatal")]
    exit_code = OK if not fatal_issues else CONTRACT_FAILURE
    return {
        "schema_version": 1,
        "tool": "minimax-h3-doctor",
        "ok": exit_code == OK,
        "exit_code": exit_code,
        "contract": {
            "platform": "Apple Silicon Mac",
            "minimum_macos": config.min_macos,
            "minimum_memory_gib": config.min_memory_gib,
            "minimum_free_disk_gib": config.min_free_gib,
            "minimum_python": config.min_python,
            "requires_uv": True,
            "requires_xcode_command_line_tools": True,
            "requires_ffmpeg_ffprobe": True,
            "requires_sudo": False,
            "requires_rosetta": False,
            "bundles_model_weights": False,
            "runs_generation": False,
            "downloads_models": False,
        },
        "checks": checks,
        "issues": issues,
        "summary": {
            "checks": len(checks),
            "fatal_issues": len(fatal_issues),
            "warnings": sum(1 for issue in issues if issue["severity"] == "warning"),
            "errors": sum(1 for issue in issues if issue["severity"] == "error"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the lightweight MiniMax-H3 M4 Pro 24GB release contract.")
    parser.add_argument("--pretty", action="store_true", help="pretty-print the JSON report")
    parser.add_argument("--json", action="store_true", help="accepted for compatibility; doctor always emits JSON")
    parser.add_argument("--checkpoint", default=None, help=f"upstream FL2VA/Ref2VA root, or {CHECKPOINT_ENV_VAR}")
    parser.add_argument("--transformer", default=None, help=f"MLX transformer root, or {TRANSFORMER_ENV_VAR}")
    parser.add_argument("--text-encoder", default=None, help=f"quantized text encoder root, or {TEXT_ENCODER_ENV_VAR}")
    parser.add_argument("--turbo-lora", default=None, help=f"optional Turbo LoRA safetensors, or {TURBO_LORA_ENV_VAR}")
    parser.add_argument("--cache-dir", default=None, help=f"model/cache directory, or {CACHE_ENV_VAR}; default is ~/.cache/minimax-h3")
    parser.add_argument("--output-dir", default=None, help=f"output directory, or {OUTPUT_ENV_VAR}; default is ./out")
    parser.add_argument("--asset-root", action="append", default=[], help="model root/file to inspect with header-only asset preflight")
    parser.add_argument("--asset-manifest", default=None, help="release asset manifest JSON; defaults to package release_assets.json")
    parser.add_argument("--asset-cache-dir", default=None, help="asset-manager cache root; defaults to --cache-dir")
    parser.add_argument("--manifest-asset", action="append", default=[], help="asset id/alias to include in manifest status; defaults to all")
    parser.add_argument("--strict-manifest-assets", action="store_true", help="make missing/invalid manifest assets fatal")
    parser.add_argument("--ffmpeg", default=None, help=f"ffmpeg executable, or {FFMPEG_ENV_VAR}, before PATH")
    parser.add_argument("--ffprobe", default=None, help=f"ffprobe executable, or {FFPROBE_ENV_VAR}, before PATH")
    parser.add_argument("--min-macos", default=DEFAULT_MIN_MACOS)
    parser.add_argument("--min-python", default=DEFAULT_MIN_PYTHON)
    parser.add_argument("--min-memory-gib", type=float, default=DEFAULT_MIN_MEMORY_GIB)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--require-model-paths", action="store_true", help="make missing checkpoint/transformer/text paths fatal")
    parser.add_argument("--require-license-acceptance", action="store_true", help=f"make missing {LICENSE_ENV_VAR}=1 fatal")
    parser.add_argument("--no-strict-assets", dest="strict_assets", action="store_false", help="report asset preflight failures as warnings")
    parser.set_defaults(strict_assets=True)
    return parser


def _default_cache_dir(env: Mapping[str, str]) -> str:
    if env.get(CACHE_ENV_VAR):
        return env[CACHE_ENV_VAR]
    base = env.get("XDG_CACHE_HOME") or str(Path(env.get("HOME", str(Path.home()))) / ".cache")
    return str(Path(base) / "minimax-h3")


def _default_output_dir(env: Mapping[str, str]) -> str:
    return env.get(OUTPUT_ENV_VAR, "out")


def config_from_args(args: argparse.Namespace, *, env: Mapping[str, str] | None = None) -> DoctorConfig:
    env_map = os.environ if env is None else env
    return DoctorConfig(
        env=env_map,
        checkpoint=args.checkpoint,
        transformer=args.transformer,
        text_encoder=args.text_encoder,
        turbo_lora=args.turbo_lora,
        cache_dir=args.cache_dir or _default_cache_dir(env_map),
        output_dir=args.output_dir or _default_output_dir(env_map),
        asset_roots=tuple(args.asset_root or ()),
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        min_macos=args.min_macos,
        min_python=args.min_python,
        min_memory_gib=args.min_memory_gib,
        min_free_gib=args.min_free_gib,
        require_model_paths=args.require_model_paths,
        require_license_acceptance=args.require_license_acceptance,
        strict_assets=args.strict_assets,
        asset_manifest=args.asset_manifest,
        asset_cache_dir=args.asset_cache_dir,
        manifest_assets=tuple(args.manifest_asset or ()),
        strict_manifest_assets=args.strict_manifest_assets,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_doctor(config_from_args(args))
    json.dump(result, sys.stdout, indent=2 if args.pretty else None, sort_keys=True)
    sys.stdout.write("\n")
    return int(result["exit_code"])


if __name__ == "__main__":  # pragma: no cover - exercised by tests and console entrypoint.
    raise SystemExit(main())
