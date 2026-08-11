#!/usr/bin/env bash
# Run the parity suite. The smoke test needs only MLX; the rest compare against the
# `minimax-h3` branch of diffusers and transformers in .venv (see requirements.txt).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-./.venv/bin/python}
if [ ! -x "$PY" ]; then
  echo "Python environment not found or not executable: $PY" >&2
  exit 2
fi
fail=0
run() {
  echo
  echo "=== $1 ==="
  if ! "$PY" "$1" 2>&1 | grep -vE "^(Modular|/opt/homebrew.*Warning|  WeightNorm)"; then
    fail=1
  fi
}

run tests/test_dit_smoke.py
run tests/test_dit_parity.py
run tests/test_video_vae_parity.py
run tests/test_audio_vae_parity.py
run tests/test_text_encoder_parity.py
run tests/test_quant_roundtrip.py
run tests/test_activation_quant.py
run tests/test_quant_audit.py
run tests/test_int8_quality.py
run tests/test_packing_parity.py
run tests/test_turbo_lora.py
run tests/test_native_turbo_lora.py
run tests/test_streaming_block.py
echo
[ $fail -eq 0 ] && echo "ALL SUITES PASSED" || echo "SOME SUITES FAILED"
exit $fail
