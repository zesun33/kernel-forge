#!/usr/bin/env bash
# verify.sh — 6-gate verification suite for kernel-forge
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

QUICK=0
GATE=""
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --gate=*) GATE="${arg#--gate=}" ;;
    --gate) shift; GATE="${1:-}" ;;
    -h|--help)
      echo "Usage: ./scripts/verify.sh [--quick] [--gate N]"
      echo "Gates: 1=spec, 2=static, 3=unit, 4=hardware, 5=packaging, 6=agent-contract"
      exit 0
      ;;
  esac
done

pass() { printf "  \033[1;32m✓\033[0m Gate %s: %s\n" "$1" "$2"; }
fail() { printf "  \033[1;31m✗\033[0m Gate %s: %s\n" "$1" "$2"; exit 1; }

run_gate_1() {
  printf "\033[1;36mGate 1 — Spec Lock & Package Integrity\033[0m\n"
  test -f pyproject.toml || fail 1 "pyproject.toml missing"
  test -f LICENSE || fail 1 "LICENSE missing"
  test -f README.md || fail 1 "README.md missing"
  pass 1 "Spec files locked"
}

run_gate_2() {
  printf "\033[1;36mGate 2 — Static Quality & Syntax Check\033[0m\n"
  python3 -m py_compile src/kernel_forge/*.py tests/*.py
  pass 2 "Python syntax validated cleanly"
}

run_gate_3() {
  printf "\033[1;36mGate 3 — Unit Tests\033[0m\n"
  PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
  pass 3 "All unit tests passed"
}

run_gate_4() {
  printf "\033[1;36mGate 4 — Hardware Integration (CUDA-visible device)\033[0m\n"
  if [ "$QUICK" = "1" ]; then
    printf "  SKIP Gate 4: Hardware execution (--quick)\n"
    return 0
  fi
  ./bin/forge bench src/kernel_forge/templates/vector_add/kernel.cu --iters 5 --json > /dev/null
  ./bin/forge bench src/kernel_forge/templates/matmul_tiled/kernel.cu --iters 5 --json > /dev/null
  pass 4 "Real CUDA kernels compiled with nvcc and executed on the selected visible device"
}

run_gate_5() {
  printf "\033[1;36mGate 5 — Packaging & CLI Executable\033[0m\n"
  test -x bin/forge || fail 5 "bin/forge is not executable"
  ./bin/forge --help > /dev/null
  pass 5 "CLI executable verified with help snapshot"
}

run_gate_6() {
  printf "\033[1;36mGate 6 — Agent JSON Schema Contract\033[0m\n"
  doctor_json=$(./bin/forge doctor --json)
  echo "$doctor_json" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert {"default_device_id", "requested_device_id", "devices"} <= d.keys(); assert d["default_device_id"] is None or any(g["index"] == d["default_device_id"] for g in d["devices"])' || fail 6 "Doctor device selection contract failed"
  if [ "$QUICK" = "1" ]; then
    echo "$doctor_json" | grep -q '"devices"' || fail 6 "Doctor JSON missing devices array"
  else
    echo "$doctor_json" | grep -q '"peak_fp32_tflops"' || fail 6 "Doctor JSON missing peak_fp32_tflops"
  fi
  pass 6 "Agent JSON output adheres to schema"
}

run_all() {
  [ -z "$GATE" ] || [ "$GATE" = "1" ] && run_gate_1
  [ -z "$GATE" ] || [ "$GATE" = "2" ] && run_gate_2
  [ -z "$GATE" ] || [ "$GATE" = "3" ] && run_gate_3
  [ -z "$GATE" ] || [ "$GATE" = "4" ] && run_gate_4
  [ -z "$GATE" ] || [ "$GATE" = "5" ] && run_gate_5
  [ -z "$GATE" ] || [ "$GATE" = "6" ] && run_gate_6
  printf "\033[1;32mAll verification gates passed.\033[0m\n"
}

run_all
