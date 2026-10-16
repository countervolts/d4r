#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"

mkdir -p "$ROOT/build"
"$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -Werror \
  "$ROOT/tools/ngx_load_probe.cpp" \
  -o "$ROOT/build/ngx_load_probe.exe"
printf 'Built %s\n' "$ROOT/build/ngx_load_probe.exe"
