"""Synthetic regression tests for the MiniMax-H3 nightly deployment runner.

These tests use fake child commands and tiny temp files only; they never touch
real model weights, ffmpeg, ffprobe, or the LaunchAgent-managed download.

Run with:
    ./.venv/bin/python tests/test_nightly_deployment_runner.py
"""

from __future__ import annotations

import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler
from scripts.run_nightly_deployment import _classify_failure, _generation_schedule_contract

FAKE_RSS_BYTES = 424_242
FAKE_MLX_PEAK_BYTES = 111_222_333
FAKE_MLX_ACTIVE_BYTES = 22_333_444
FAKE_MLX_CACHE_BYTES = 33_444_555


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def write_executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def write_tiny_safetensors(path: Path) -> None:
    header = {
        "model.language_model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0\0\0\0")


def write_valid_text_encoder_dir(path: Path, *, bits: int = 4, group_size: int = 64, num_layers: int = 50) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}) + "\n")
    write_tiny_safetensors(path / "model.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.embed_tokens.weight": "model.safetensors",
                }
            },
            sort_keys=True,
        )
        + "\n"
    )
    (path / "quant_config.json").write_text(
        json.dumps(
            {
                "bits": bits,
                "group_size": group_size,
                "num_layers": num_layers,
                "source_bytes": 4,
                "output_bytes": 4,
                "quantized_tensors": 1,
            },
            sort_keys=True,
        )
        + "\n"
    )


def write_invalid_text_encoder_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}) + "\n")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.language_model.embed_tokens.weight": "missing.safetensors"}}) + "\n"
    )


def write_fake_tools(directory: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    call_log = directory / "python-calls.jsonl"
    ffprobe_log = directory / "ffprobe-calls.jsonl"
    fake_python = directory / "fake-python"
    fake_time = directory / "fake-time"
    fake_ffmpeg = directory / "fake-ffmpeg"
    fake_ffprobe = directory / "fake-ffprobe"

    write_executable(
        fake_python,
        f"""#!{sys.executable}
import json
import os
import struct
import sys
from pathlib import Path

MLX_LINE = (
    "ARGUS_MLX_MEMORY "
    "peak_bytes={FAKE_MLX_PEAK_BYTES} "
    "active_bytes={FAKE_MLX_ACTIVE_BYTES} "
    "cache_bytes={FAKE_MLX_CACHE_BYTES}"
)


def append_call():
    path = Path(os.environ["FAKE_CALL_LOG"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd()}}, sort_keys=True) + "\\n")


def preflight_payload(ok):
    return {{
        "schema_version": 1,
        "ok": ok,
        "exit_code": 0 if ok else 2,
        "summary": {{"errors": 0 if ok else 1, "warnings": 0}},
        "issues": [] if ok else [{{
            "severity": "error",
            "code": "missing_shard_safetensors",
            "path": "models/fake-missing.safetensors",
            "message": "synthetic missing asset",
        }}],
    }}


def option_value(option, default=None):
    if option not in sys.argv:
        return default
    index = sys.argv.index(option)
    if index + 1 >= len(sys.argv):
        return default
    return sys.argv[index + 1]


def write_tiny_safetensors(path):
    header = {{
        "model.language_model.embed_tokens.weight": {{
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        }}
    }}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\\0\\0\\0\\0")


def write_valid_text_encoder_dir(path):
    bits = int(option_value("--bits", "4"))
    group_size = int(option_value("--group-size", "64"))
    num_layers = int(option_value("--num-layers", "50"))
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({{"model_type": "qwen3_vl"}}) + "\\n")
    write_tiny_safetensors(path / "model.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({{
        "weight_map": {{"model.language_model.embed_tokens.weight": "model.safetensors"}},
    }}, sort_keys=True) + "\\n")
    (path / "quant_config.json").write_text(json.dumps({{
        "bits": bits,
        "group_size": group_size,
        "num_layers": num_layers,
        "source_bytes": 4,
        "output_bytes": 4,
        "quantized_tensors": 1,
    }}, sort_keys=True) + "\\n")


def main():
    append_call()
    mode = os.environ.get("FAKE_RUNNER_MODE", "success")
    if len(sys.argv) < 2:
        print("missing target", file=sys.stderr)
        return 64
    target = sys.argv[1]

    if target == "scripts/preflight_assets.py":
        ok = mode != "preflight_blocked"
        print(json.dumps(preflight_payload(ok), sort_keys=True))
        return 0 if ok else 2

    if target == "scripts/run_with_mlx_memory.py":
        wrapped = sys.argv[2] if len(sys.argv) > 2 else ""
        if wrapped == "scripts/quantize_text_encoder.py":
            print(MLX_LINE, file=sys.stderr)
            output = Path(option_value("--output", ""))
            if mode == "quantization_failed":
                output.mkdir(parents=True, exist_ok=True)
                (output / "config.json").write_text(json.dumps({{"model_type": "qwen3_vl"}}) + "\\n")
                print("synthetic quantization failure after partial staging", file=sys.stderr)
                return 1
            write_valid_text_encoder_dir(output)
            return 0
        if wrapped == "scripts/verify_nightly_gates.py":
            print(MLX_LINE, file=sys.stderr)
            subcommand = sys.argv[sys.argv.index("scripts/verify_nightly_gates.py") + 1:]
            # Drop wrapper-level options before the verifier subcommand.
            while subcommand and subcommand[0].startswith("--"):
                option = subcommand.pop(0)
                if option in {{"--memory-limit-gb"}} and subcommand:
                    subcommand.pop(0)
            gate = subcommand[0] if subcommand else ""
            if mode == "prompt_verification_failed" and gate == "prompt-release":
                print(json.dumps({{"ok": False, "gate": "prompt_release", "error": "synthetic prompt verifier failure"}}))
                return 3
            if mode == "streaming_verification_failed" and gate == "streaming-components":
                print(json.dumps({{"ok": False, "gate": "streaming_components", "error": "synthetic streaming verifier failure"}}))
                return 3
            if gate == "prompt-release":
                print(json.dumps({{
                    "ok": True,
                    "gate": "prompt_release",
                    "component_released": True,
                    "detached_usable_after_release": True,
                    "token_tag_count": 8,
                }}, sort_keys=True))
                return 0
            if gate == "streaming-components":
                print(json.dumps({{
                    "ok": True,
                    "gate": "streaming_components",
                    "block_count": 50,
                    "checked_block_indices": [0, 49],
                    "logical_bytes_loaded_total": 123456,
                    "turbo_lora": {{"configured": False, "status": "not_configured", "verified": False}},
                }}, sort_keys=True))
                return 0
            print("unexpected verifier gate: " + gate, file=sys.stderr)
            return 70
        if wrapped == "scripts/generate.py":
            print(MLX_LINE, file=sys.stderr)
            if mode == "oom_generation":
                print("Killed: 9 -- synthetic MLX out of memory", file=sys.stderr)
                return 137
            try:
                output = Path(sys.argv[sys.argv.index("-o") + 1])
            except Exception:
                print("missing -o", file=sys.stderr)
                return 64
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"synthetic non-empty mp4 placeholder")
            return 0

    print("unexpected fake-python command: " + json.dumps(sys.argv[1:]), file=sys.stderr)
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
""",
    )

    write_executable(
        fake_time,
        f"""#!{sys.executable}
import os
import subprocess
import sys

args = sys.argv[1:]
if args and args[0] == "-l":
    args = args[1:]
completed = subprocess.run(args, check=False)
rss = os.environ.get("FAKE_RSS_BYTES", "{FAKE_RSS_BYTES}")
print(f"        {{rss}}  maximum resident set size", file=sys.stderr)
raise SystemExit(completed.returncode)
""",
    )

    write_executable(
        fake_ffmpeg,
        f"""#!{sys.executable}
import os
import struct
import sys

if "-version" in sys.argv[1:]:
    print("ffmpeg version synthetic-1.0")
    raise SystemExit(0)

if "-f" in sys.argv[1:] and "f32le" in sys.argv[1:] and "pipe:1" in sys.argv[1:]:
    mode = os.environ.get("FAKE_FFMPEG_AUDIO_MODE", "active")
    if mode == "active":
        samples = [0.02 if index % 2 else -0.02 for index in range(64)]
        sys.stdout.buffer.write(struct.pack("<" + "f" * len(samples), *samples))
        raise SystemExit(0)
    if mode == "silent":
        samples = [0.0 for _ in range(64)]
        sys.stdout.buffer.write(struct.pack("<" + "f" * len(samples), *samples))
        raise SystemExit(0)
    if mode == "undecodable":
        print("synthetic audio decode failure", file=sys.stderr)
        raise SystemExit(1)
    print("unexpected FAKE_FFMPEG_AUDIO_MODE=" + mode, file=sys.stderr)
    raise SystemExit(70)

print("unexpected fake-ffmpeg command: " + " ".join(sys.argv[1:]), file=sys.stderr)
raise SystemExit(70)
""",
    )

    write_executable(
        fake_ffprobe,
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

if "-version" in sys.argv[1:]:
    print("ffprobe version synthetic-1.0")
    raise SystemExit(0)

path = Path(os.environ["FAKE_FFPROBE_LOG"])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a") as handle:
    handle.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd()}}, sort_keys=True) + "\\n")

mode = os.environ.get("FAKE_FFPROBE_MODE", "av")
streams = [{{"codec_type": "video", "codec_name": "h264"}}]
if mode == "av":
    streams.append({{"codec_type": "audio", "codec_name": "aac"}})
elif mode != "video_only":
    print("unexpected FAKE_FFPROBE_MODE=" + mode, file=sys.stderr)
    raise SystemExit(70)
print(json.dumps({{
    "streams": streams,
    "format": {{"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "5.000000"}},
}}, sort_keys=True))
""",
    )

    return fake_python, fake_time, fake_ffmpeg, fake_ffprobe, call_log, ffprobe_log


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_synthetic_runner(
    *,
    mode: str,
    ffprobe_mode: str = "av",
    ffmpeg_audio_mode: str = "active",
    ladder: str = "320x192,512x288",
    steps: int = 5,
    preexisting_text_encoder: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict, list[dict], list[dict]]:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        fake_python, fake_time, fake_ffmpeg, fake_ffprobe, call_log, ffprobe_log = write_fake_tools(tmp)
        run_dir = tmp / "run"
        output_dir = tmp / "outputs"
        quant_source = tmp / "official_text_encoder_source"
        quant_source.mkdir()
        (quant_source / "source_sentinel.txt").write_text("official source must not be modified\n")
        source_before = sorted((p.relative_to(quant_source).as_posix(), p.read_bytes()) for p in quant_source.rglob("*") if p.is_file())
        text_encoder = tmp / "text_encoder-mlx-4bit"
        if preexisting_text_encoder == "valid":
            write_valid_text_encoder_dir(text_encoder, bits=4, group_size=64, num_layers=50)
        elif preexisting_text_encoder == "invalid":
            write_invalid_text_encoder_dir(text_encoder)
        elif preexisting_text_encoder is not None:
            raise AssertionError(f"unknown preexisting_text_encoder mode: {preexisting_text_encoder}")
        env = os.environ.copy()
        env.update(
            {
                "FAKE_RUNNER_MODE": mode,
                "FAKE_FFPROBE_MODE": ffprobe_mode,
                "FAKE_FFMPEG_AUDIO_MODE": ffmpeg_audio_mode,
                "FAKE_CALL_LOG": str(call_log),
                "FAKE_FFPROBE_LOG": str(ffprobe_log),
                "FAKE_RSS_BYTES": str(FAKE_RSS_BYTES),
            }
        )
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_nightly_deployment.py"),
            "--python",
            str(fake_python),
            "--time-bin",
            str(fake_time),
            "--ffmpeg",
            str(fake_ffmpeg),
            "--ffprobe",
            str(fake_ffprobe),
            "--run-id",
            f"synthetic-{mode}-{ffprobe_mode}",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--quant-source",
            str(quant_source),
            "--text-encoder",
            str(text_encoder),
            "--ladder",
            ladder,
            "--steps",
            str(steps),
            "--no-update-preflight-json",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=env, check=False)
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise AssertionError(
                "runner did not write summary.json\n"
                f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )
        summary = json.loads(summary_path.read_text())
        source_after = sorted((p.relative_to(quant_source).as_posix(), p.read_bytes()) for p in quant_source.rglob("*") if p.is_file())
        summary["_test_source_unchanged"] = source_before == source_after
        summary["_test_final_text_encoder_exists"] = text_encoder.exists()
        summary["_test_staging_dirs"] = sorted(path.name for path in tmp.glob(f".{text_encoder.name}.staging-*"))
        return proc, summary, read_jsonl(call_log), read_jsonl(ffprobe_log)


def step(summary: dict, name: str) -> dict:
    matches = [record for record in summary["steps"] if record["step"] == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one step named {name!r}, found {len(matches)}")
    return matches[0]


def wrapped_call_count(calls: list[dict], script_name: str) -> int:
    return sum(
        1
        for call in calls
        if len(call["argv"]) >= 2
        and call["argv"][0] == "scripts/run_with_mlx_memory.py"
        and call["argv"][1] == script_name
    )


def wrapped_argvs(calls: list[dict], script_name: str) -> list[list[str]]:
    return [
        call["argv"]
        for call in calls
        if len(call["argv"]) >= 2
        and call["argv"][0] == "scripts/run_with_mlx_memory.py"
        and call["argv"][1] == script_name
    ]


def option_value(argv: list[str], option: str) -> str:
    if option not in argv:
        raise AssertionError(f"{option!r} missing from argv {argv!r}")
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise AssertionError(f"{option!r} missing value in argv {argv!r}")
    return argv[index + 1]


def test_generation_schedule_contract_is_explicit() -> None:
    expected = _generation_schedule_contract(5)
    assert_case("schedule helper maps 5 sigma points to 4 NFE", expected["sigma_points"] == 5 and expected["nfe"] == 4)

    for shift, label in ((12.0, "video"), (3.0, "audio")):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(5)
        assert_case(f"{label} scheduler keeps 5 sigma points", len(scheduler.sigmas.tolist()) == 5)
        assert_case(f"{label} scheduler exposes 4 denoiser timesteps", len(scheduler.timesteps.tolist()) == 4)
        assert_case(f"{label} scheduler num_inference_steps is 4", scheduler.num_inference_steps == 4)

    proc, summary, calls, _ffprobe_calls = run_synthetic_runner(mode="success", ladder="320x192,512x288")
    assert_case("schedule-contract runner succeeds", proc.returncode == 0 and summary["status"] == "success")
    assert_case("summary records explicit schedule contract", summary["generation_schedule_contract"] == expected)

    generate_argvs = wrapped_argvs(calls, "scripts/generate.py")
    assert_case("two ladder generations executed", len(generate_argvs) == 2)
    assert_case("each generate command passes --steps 5", all(option_value(argv, "--steps") == "5" for argv in generate_argvs))

    for name in ("generate_320x192", "generate_512x288"):
        record = step(summary, name)
        contract = record["generation_schedule_contract"]
        assert_case(f"{name} records 5 sigma points", contract["sigma_points"] == 5)
        assert_case(f"{name} records 4 denoiser evaluations", contract["denoiser_evaluations"] == 4)
        assert_case(f"{name} command keeps --steps 5", option_value(record["command"], "--steps") == "5")
        assert_case(f"{name} media artifact is labeled 5sigma", "5sigma" in record["artifact_paths"]["media_path"])


def test_text_encoder_quantization_is_atomic_and_recoverable() -> None:
    proc, summary, calls, _ffprobe_calls = run_synthetic_runner(mode="success", ladder="320x192")
    quantize = step(summary, "quantize_text_encoder")
    command_output = option_value(quantize["command"], "--output")
    staging_dir = quantize["artifact_paths"]["text_encoder_staging_dir"]
    final_dir = quantize["artifact_paths"]["text_encoder_dir"]
    assert_case("atomic text-encoder runner succeeds", proc.returncode == 0 and summary["status"] == "success")
    assert_case("absent text encoder quantizes to staging path", command_output == staging_dir and command_output != final_dir)
    assert_case("staging path is clearly marked", ".staging-" in staging_dir)
    assert_case("atomic commit is recorded", quantize["atomic_commit"]["method"] == "os.replace" and quantize["atomic_commit"]["committed"] is True)
    assert_case("staging output validates before commit", quantize["text_encoder_staging_validation"]["ok"] is True)
    assert_case("final text encoder validates after commit", quantize["text_encoder_validation"]["ok"] is True)
    assert_case("final text encoder exists after atomic commit", summary["_test_final_text_encoder_exists"] is True)
    assert_case("official source sentinel unchanged by quantization", summary["_test_source_unchanged"] is True)
    assert_case("fake quantizer executed exactly once", wrapped_call_count(calls, "scripts/quantize_text_encoder.py") == 1)

    proc, summary, calls, ffprobe_calls = run_synthetic_runner(mode="quantization_failed", ladder="320x192")
    quantize = step(summary, "quantize_text_encoder")
    assert_case("failed quantization returns command failure", proc.returncode == 1 and summary["status"] == "failed_quantization")
    assert_case("failed quantization never commits final target", quantize["atomic_commit"]["committed"] is False and summary["_test_final_text_encoder_exists"] is False)
    assert_case("partial output is only staging", len(summary["_test_staging_dirs"]) == 1)
    assert_case("verifiers skipped after quantization failure", step(summary, "verify_prompt_release")["reason"] == "quantization_failed_exit_1")
    assert_case("generation skipped after quantization failure", step(summary, "generate_320x192")["reason"] == "quantization_failed_exit_1")
    assert_case("ffprobe not run after quantization failure", ffprobe_calls == [])


def test_existing_text_encoder_is_validated_before_reuse_or_block() -> None:
    proc, summary, calls, _ffprobe_calls = run_synthetic_runner(
        mode="success", ladder="320x192", preexisting_text_encoder="valid"
    )
    quantize = step(summary, "quantize_text_encoder")
    assert_case("valid existing text encoder runner succeeds", proc.returncode == 0 and summary["status"] == "success")
    assert_case("valid existing text encoder is reused", quantize["text_encoder_status"] == "reused_existing_valid")
    assert_case("reuse is accepted only after validation", quantize["text_encoder_validation"]["ok"] is True)
    assert_case("reuse skips expensive quantizer", wrapped_call_count(calls, "scripts/quantize_text_encoder.py") == 0)
    assert_case("official source unchanged on reuse", summary["_test_source_unchanged"] is True)

    proc, summary, calls, ffprobe_calls = run_synthetic_runner(
        mode="success", ladder="320x192", preexisting_text_encoder="invalid"
    )
    quantize = step(summary, "quantize_text_encoder")
    assert_case("invalid existing text encoder returns validation failure", proc.returncode == 3 and summary["status"] == "blocked_invalid_text_encoder")
    assert_case("invalid existing text encoder has explicit status", quantize["text_encoder_status"] == "invalid_existing_text_encoder")
    assert_case("invalid existing text encoder validation failed", quantize["text_encoder_validation"]["ok"] is False)
    assert_case("invalid existing text encoder is not overwritten", wrapped_call_count(calls, "scripts/quantize_text_encoder.py") == 0)
    assert_case("prompt verifier skipped behind invalid text encoder", step(summary, "verify_prompt_release")["reason"] == "invalid_existing_text_encoder")
    assert_case("generation skipped behind invalid text encoder", step(summary, "generate_320x192")["reason"] == "invalid_existing_text_encoder")
    assert_case("ffprobe not run behind invalid text encoder", ffprobe_calls == [])


def test_preflight_exit_two_skips_downstream() -> None:
    proc, summary, calls, ffprobe_calls = run_synthetic_runner(mode="preflight_blocked")
    assert_case("preflight-blocked runner exits with asset failure", proc.returncode == 2)
    assert_case("summary records blocked_preflight", summary["status"] == "blocked_preflight")
    assert_case("preflight record keeps exit 2", step(summary, "preflight")["exit_code"] == 2)
    assert_case("preflight RSS parsed from timed stderr", step(summary, "preflight")["peak_rss_bytes"] == FAKE_RSS_BYTES)
    assert_case("only fake preflight executed", [call["argv"][0] for call in calls] == ["scripts/preflight_assets.py"])
    assert_case("ffprobe not executed behind failed preflight", ffprobe_calls == [])

    skipped = [record for record in summary["steps"] if record.get("skipped")]
    skipped_names = {record["step"] for record in skipped}
    assert_case("quantization skipped after failed preflight", "quantize_text_encoder" in skipped_names)
    assert_case("prompt-release verifier skipped after failed preflight", "verify_prompt_release" in skipped_names)
    assert_case("streaming verifier skipped after failed preflight", "verify_streaming_components" in skipped_names)
    assert_case("generation skipped after failed preflight", "generate_320x192" in skipped_names and "generate_512x288" in skipped_names)
    assert_case("media validation skipped after failed preflight", "ffprobe_320x192" in skipped_names and "ffprobe_512x288" in skipped_names)
    assert_case("all downstream skips name preflight blocker", all(record["reason"] == "preflight_failed_exit_2" for record in skipped))


def test_oom_generation_classified_once_no_blind_retry() -> None:
    assert_case("signal-kill classifier is OOM/killed", _classify_failure(-9, "", "") == "oom_or_killed")

    proc, summary, calls, ffprobe_calls = run_synthetic_runner(mode="oom_generation")
    assert_case("OOM runner returns child failure code", proc.returncode == 137)
    assert_case("summary records failed_generation", summary["status"] == "failed_generation")

    quantize = step(summary, "quantize_text_encoder")
    generate = step(summary, "generate_320x192")
    assert_case("quantization MLX peak parsed", quantize["mlx_memory"]["peak_bytes"] == FAKE_MLX_PEAK_BYTES)
    assert_case("generation RSS parsed", generate["peak_rss_bytes"] == FAKE_RSS_BYTES)
    assert_case("generation MLX peak parsed", generate["mlx_memory"]["peak_bytes"] == FAKE_MLX_PEAK_BYTES)
    assert_case("generation failure classified as OOM/killed", generate["failure_class"] == "oom_or_killed")
    assert_case("failed command is not marked ok", generate["ok"] is False)
    assert_case("same generation command not blindly retried", wrapped_call_count(calls, "scripts/generate.py") == 1)
    assert_case("quantization executed once", wrapped_call_count(calls, "scripts/quantize_text_encoder.py") == 1)
    assert_case("both verifier gates executed before generation", wrapped_call_count(calls, "scripts/verify_nightly_gates.py") == 2)
    assert_case("prompt-release verifier succeeded before OOM", step(summary, "verify_prompt_release")["verification_ok"] is True)
    assert_case("streaming verifier succeeded before OOM", step(summary, "verify_streaming_components")["verification_ok"] is True)
    assert_case("ffprobe not run after failed generation", ffprobe_calls == [])
    assert_case("later ladder generation skipped", step(summary, "generate_512x288")["reason"] == "previous_generation_failed_320x192")
    assert_case("later ladder ffprobe skipped", step(summary, "ffprobe_512x288")["reason"] == "previous_generation_failed_320x192")


def test_verifier_failures_stop_before_generation() -> None:
    proc, summary, calls, ffprobe_calls = run_synthetic_runner(mode="prompt_verification_failed", ladder="320x192")
    assert_case("prompt verifier failure returns validation failure", proc.returncode == 3)
    assert_case("prompt verifier failure has explicit status", summary["status"] == "failed_prompt_release_verification")
    assert_case("streaming verifier skipped after prompt verifier failure", step(summary, "verify_streaming_components")["reason"] == "prompt_release_verification_failed_exit_3")
    assert_case("generation skipped after prompt verifier failure", step(summary, "generate_320x192")["reason"] == "prompt_release_verification_failed_exit_3")
    assert_case("no generation after prompt verifier failure", wrapped_call_count(calls, "scripts/generate.py") == 0)
    assert_case("no ffprobe after prompt verifier failure", ffprobe_calls == [])

    proc, summary, calls, ffprobe_calls = run_synthetic_runner(mode="streaming_verification_failed", ladder="320x192")
    assert_case("streaming verifier failure returns validation failure", proc.returncode == 3)
    assert_case("streaming verifier failure has explicit status", summary["status"] == "failed_streaming_verification")
    assert_case("prompt verifier ran before streaming failure", step(summary, "verify_prompt_release")["verification_ok"] is True)
    assert_case("generation skipped after streaming verifier failure", step(summary, "generate_320x192")["reason"] == "streaming_components_verification_failed_exit_3")
    assert_case("no generation after streaming verifier failure", wrapped_call_count(calls, "scripts/generate.py") == 0)
    assert_case("no ffprobe after streaming verifier failure", ffprobe_calls == [])


def test_ffprobe_requires_audio_and_video() -> None:
    proc, summary, calls, ffprobe_calls = run_synthetic_runner(mode="success", ffprobe_mode="av", ladder="320x192")
    validate = step(summary, "ffprobe_320x192")
    audio_activity = step(summary, "audio_activity_320x192")
    streaming = step(summary, "verify_streaming_components")
    generate = step(summary, "generate_320x192")
    assert_case("audio+video runner succeeds", proc.returncode == 0 and summary["status"] == "success")
    assert_case("ffmpeg tool path is recorded", summary["media_tools"]["ffmpeg"]["source"] == "explicit_path")
    assert_case("ffprobe tool path is recorded", summary["media_tools"]["ffprobe"]["source"] == "explicit_path")
    assert_case("ffmpeg verifier ran", step(summary, "verify_ffmpeg")["media_tool_version_line"].startswith("ffmpeg version"))
    assert_case("ffprobe verifier ran", step(summary, "verify_ffprobe")["media_tool_version_line"].startswith("ffprobe version"))
    assert_case("generation receives explicit ffmpeg", option_value(generate["command"], "--ffmpeg").endswith("fake-ffmpeg"))
    assert_case("generation forbids frames fallback", "--require-muxed-mp4" in generate["command"])
    assert_case("prompt-release verifier ran", step(summary, "verify_prompt_release")["verification_summary"]["component_released"] is True)
    assert_case("streaming verifier ran", streaming["verification_summary"]["checked_block_indices"] == [0, 49])
    assert_case("unconfigured Turbo is not reported as verified", streaming["verification_turbo_lora"] == {"configured": False, "status": "not_configured", "verified": False})
    assert_case("ffprobe success requires video stream", validate["video_stream_count"] == 1)
    assert_case("ffprobe success requires audio stream", validate["audio_stream_count"] == 1)
    assert_case("ffprobe step marked ok with audio+video", validate["ok"] is True)
    assert_case("decoded-audio activity step marked ok", audio_activity["ok"] is True and audio_activity["postcheck_ok"] is True)
    assert_case("decoded-audio activity is machine-recorded", audio_activity["nonzero_sample_count"] >= 16 and audio_activity["rms"] > 0)
    assert_case("one ffprobe executed on success", len(ffprobe_calls) == 1)
    assert_case("generation command executed once on success", wrapped_call_count(calls, "scripts/generate.py") == 1)

    proc, summary, _calls, ffprobe_calls = run_synthetic_runner(mode="success", ffprobe_mode="video_only", ladder="320x192")
    validate = step(summary, "ffprobe_320x192")
    assert_case("missing-audio runner returns validation failure", proc.returncode == 3 and summary["status"] == "failed_media_validation")
    assert_case("ffprobe itself exited zero in missing-audio case", validate["exit_code"] == 0)
    assert_case("missing-audio validation not ok", validate["ok"] is False and validate["postcheck_ok"] is False)
    assert_case("missing-audio error is explicit", validate["validation_error"] == "ffprobe found no audio stream")
    assert_case("missing-audio still saw video", validate["video_stream_count"] == 1 and validate["audio_stream_count"] == 0)
    assert_case("one ffprobe executed in missing-audio case", len(ffprobe_calls) == 1)


def test_audio_activity_requires_decoded_non_silent_audio() -> None:
    proc, summary, _calls, _ffprobe_calls = run_synthetic_runner(
        mode="success", ffprobe_mode="av", ffmpeg_audio_mode="active", ladder="320x192"
    )
    audio_activity = step(summary, "audio_activity_320x192")
    assert_case("non-silent decoded audio runner succeeds", proc.returncode == 0 and summary["status"] == "success")
    assert_case("non-silent decoded audio passes activity gate", audio_activity["ok"] is True and audio_activity["postcheck_ok"] is True)
    assert_case("activity gate records decoded sample count", audio_activity["decoded_audio_sample_count"] == 64)
    assert_case("activity gate records RMS and peak", audio_activity["rms"] > 0 and audio_activity["max_abs_sample"] > 0)

    proc, summary, _calls, _ffprobe_calls = run_synthetic_runner(
        mode="success", ffprobe_mode="av", ffmpeg_audio_mode="silent", ladder="320x192"
    )
    audio_activity = step(summary, "audio_activity_320x192")
    assert_case("silent decoded audio returns validation failure", proc.returncode == 3 and summary["status"] == "failed_media_validation")
    assert_case("silent decoded audio command itself exited zero", audio_activity["exit_code"] == 0)
    assert_case("silent decoded audio fails postcheck", audio_activity["ok"] is False and audio_activity["postcheck_ok"] is False)
    assert_case("silent decoded audio error is explicit", audio_activity["validation_error"] == "decoded audio is silent or below activity thresholds")
    assert_case("silent decoded audio has zero activity metrics", audio_activity["nonzero_sample_count"] == 0 and audio_activity["rms"] == 0.0)

    proc, summary, _calls, _ffprobe_calls = run_synthetic_runner(
        mode="success", ffprobe_mode="av", ffmpeg_audio_mode="undecodable", ladder="320x192"
    )
    audio_activity = step(summary, "audio_activity_320x192")
    assert_case("undecodable audio returns media-validation failure", proc.returncode == 1 and summary["status"] == "failed_media_validation")
    assert_case("undecodable audio command failure is explicit", audio_activity["validation_error"] == "ffmpeg audio decode returned non-zero")
    assert_case("undecodable audio step is not ok", audio_activity["ok"] is False and audio_activity["exit_code"] == 1)


def main() -> None:
    test_generation_schedule_contract_is_explicit()
    test_text_encoder_quantization_is_atomic_and_recoverable()
    test_existing_text_encoder_is_validated_before_reuse_or_block()
    test_preflight_exit_two_skips_downstream()
    test_oom_generation_classified_once_no_blind_retry()
    test_verifier_failures_stop_before_generation()
    test_ffprobe_requires_audio_and_video()
    test_audio_activity_requires_decoded_non_silent_audio()
    print("nightly deployment runner focused tests passed")


if __name__ == "__main__":
    main()
