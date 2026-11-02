#!/usr/bin/env bash
# Builds the Windows-side pieces with mingw-w64 (no NVIDIA code involved).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wall -Wextra -Wno-cast-function-type -shared -static -static-libgcc \
  -static-libstdc++ -o build/nvngx.dll_dlssg_host.dll dlssg_host.cpp -ldxguid -luuid
x86_64-w64-mingw32-strip build/nvngx.dll_dlssg_host.dll
echo "built build/nvngx.dll_dlssg_host.dll"
