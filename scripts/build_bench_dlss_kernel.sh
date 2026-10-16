#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-g++}"

mkdir -p "$ROOT/build"
"$CXX" -std=c++20 -O2 -Wall -Wextra \
  "$ROOT/tools/bench_dlss_kernel.cpp" -ldl \
  -o "$ROOT/build/bench_dlss_kernel"
printf 'Built %s\n' "$ROOT/build/bench_dlss_kernel"
