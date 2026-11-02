#!/usr/bin/env bash
# Experimental backend; no installation or modification of the reference path.
# usage: build_microcuda.sh [OUTPUT_DIR]
# Modules come from a pack made separately with tools/microcuda/make_pack.py (D4R_MICROCUDA_PACK at run time).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -le 1 ]] || { echo "usage: $0 [OUTPUT_DIR]" >&2; exit 2; }
rocm="${D4R_ROCM_DIR:-/opt/rocm}"
out="$(realpath -m "${1:-$root/build/microcuda}")"
mkdir -p "$out"
"${CXX:-g++}" -std=c++20 -O2 -fPIC -shared -Wall -Wextra -Werror -Wno-deprecated-declarations \
  -I"$rocm/include" "$root/tools/microcuda/runtime.cpp" \
  -L"$rocm/lib" -Wl,-rpath,"$rocm/lib" -lamdhip64 -lcrypto -lpthread \
  -Wl,-z,defs -Wl,-soname,libd4r_microcuda.so -o "$out/libd4r_microcuda.so"
echo "Built $out/libd4r_microcuda.so (opt-in; modules from a make_pack.py pack)"
