#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-g++}"

mkdir -p "$ROOT/build"
"$CXX" -std=c++20 -O2 -Wall -Wextra \
  "$ROOT/tools/replay_dlss_fp8_ops.cpp" -ldl \
  -o "$ROOT/build/replay_dlss_fp8_ops"
printf 'Built %s\n' "$ROOT/build/replay_dlss_fp8_ops"
