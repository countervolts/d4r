#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s PATH_TO_LOCALLY_CAPTURED_DLSS_HISTOGRAM_FATBIN\n' "$0" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZLUDA_ROOT="${ZLUDA_ROOT:-}"
ROCM_ROOT="${ROCM_ROOT:-}"
if [[ -z "$ZLUDA_ROOT" || -z "$ROCM_ROOT" || ! -f "$ZLUDA_ROOT/libcuda.so" || ! -f "$ROCM_ROOT/lib/libamdhip64.so" ]]; then
  printf 'Set ZLUDA_ROOT and ROCM_ROOT to the ZLUDA and HIP runtimes\n' >&2
  exit 2
fi

"$ROOT/scripts/build_replay_dlss_histogram.sh"
LD_LIBRARY_PATH="$ZLUDA_ROOT:$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
  "$ROOT/build/replay_dlss_histogram" "$1"
