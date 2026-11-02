#!/usr/bin/env bash
# Builds d3d12_dlssg.exe (mingw-w64; d3d12.dll at run time comes from vkd3d-proton under Proton).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wall -Wno-unused-function -static -static-libgcc -static-libstdc++ \
  -o d3d12_dlssg.exe d3d12_dlssg.cpp -ld3d12 -ldxgi -ldxguid
