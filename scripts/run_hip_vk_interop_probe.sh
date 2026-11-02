#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"

"$ROOT/scripts/build_hip_vk_interop_probe.sh"
LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
  "$ROOT/build/hip_vk_interop_probe"
