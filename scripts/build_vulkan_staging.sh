#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# the shaders and the generated header come from build_vulkan_staging.py (shared with the Windows tools build)
python3 "$ROOT/scripts/build_vulkan_staging.py" "${1:-$ROOT/build/vulkan-staging}"
