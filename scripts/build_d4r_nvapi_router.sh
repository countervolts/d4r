#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/build"
"${MINGW_CXX:-x86_64-w64-mingw32-g++}" -std=c++20 -O2 -Wall -Wextra -Werror \
  -shared -static "$ROOT/tools/d4r_nvapi_router.cpp" -o "$ROOT/build/d4r_nvapi64.dll"
printf 'Built %s\n' "$ROOT/build/d4r_nvapi64.dll"
