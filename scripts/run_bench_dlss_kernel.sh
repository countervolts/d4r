#!/usr/bin/env bash
set -euo pipefail

# usage: run_bench_dlss_kernel.sh FATBIN REPLAY_DIR [ITERATIONS [OUTPUT_PREFIX]]
# Replays a launch captured with D4R_CUDA_REPLAY_DUMP_DIR (wine_nvcuda_bridge.c).
if [[ $# -lt 2 ]]; then
  printf 'usage: %s FATBIN REPLAY_DIR [ITERATIONS [OUTPUT_PREFIX]]\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZLUDA_ROOT="${ZLUDA_ROOT:-}"
ROCM_ROOT="${ROCM_ROOT:-}"
if [[ -z "$ZLUDA_ROOT" || -z "$ROCM_ROOT" || ! -f "$ZLUDA_ROOT/libcuda.so" || ! -f "$ROCM_ROOT/lib/libamdhip64.so" ]]; then
  printf 'Set ZLUDA_ROOT and ROCM_ROOT to the ZLUDA and HIP runtimes\n' >&2
  exit 2
fi
[[ -x "$ROOT/build/bench_dlss_kernel" && "$ROOT/build/bench_dlss_kernel" -nt "$ROOT/tools/bench_dlss_kernel.cpp" ]] ||
  "$ROOT/scripts/build_bench_dlss_kernel.sh" >/dev/null
LD_LIBRARY_PATH="$ZLUDA_ROOT:$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
  "$ROOT/build/bench_dlss_kernel" "$@"
