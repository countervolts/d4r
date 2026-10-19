#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
revision=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
sources=$(sha256sum tools/*.h tools/d4r_nvngx_shim.cpp tools/d4r_vulkan_frontend.inl tools/wine_nvcuda_bridge.c \
  engine/*.h engine/*.cpp engine/*.inc | sha256sum | cut -c1-16)
printf '%s-%s\n' "$revision" "$sources"
