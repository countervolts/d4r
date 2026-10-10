#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build/native-engine}"
CXX="${CXX:-g++}"
mkdir -p "$OUT"
# The native HIP network (hip_net.h) is shared by both runners when ROCm is installed; without it they fall back
# to the engine's own Vulkan network. The library itself never links HIP.
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
HIP=()
if [[ -f "$ROCM/include/hip/hip_runtime_api.h" ]]; then
    HIP=(-DD4R_ENGINE_HIP "$ROOT/engine/hip_net.cpp" -I"$ROCM/include" -L"$ROCM/lib" -lamdhip64 "-Wl,-rpath,$ROCM/lib")
fi
"$CXX" -std=c++17 -O3 -Wall -Wextra -Wno-missing-field-initializers -fPIC -shared \
    "$ROOT/engine/runtime.cpp" "$ROOT/engine/game.cpp" "$ROOT/engine/game_m.cpp" -lvulkan -o "$OUT/libd4r_engine.so"
"$CXX" -std=c++17 -O3 -Wall -Wextra -Wno-missing-field-initializers \
    "$ROOT/engine/replay.cpp" "${HIP[@]}" -L"$OUT" -ld4r_engine -lvulkan '-Wl,-rpath,$ORIGIN' -o "$OUT/d4r-k"
"$CXX" -std=c++17 -O3 -Wall -Wextra -Wno-missing-field-initializers \
    "$ROOT/engine/replay_m.cpp" "${HIP[@]}" -L"$OUT" -ld4r_engine -lvulkan '-Wl,-rpath,$ORIGIN' -o "$OUT/d4r-m"
printf 'Built %s and %s\n' "$OUT/d4r-k" "$OUT/d4r-m"
