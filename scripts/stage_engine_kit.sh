#!/usr/bin/env bash
# Puts the engine-model builder into a release's d4r folder: d4r/tools/build-engine-models.sh (install.sh runs it)
# and d4r/tools/models/ with d4r's model compilers and shader sources, the D3D12 test app and glslangValidator.
# Only d4r's own code and Khronos' glslang go in; the models are built on the user's machine from their DLLs.
#
# usage: scripts/stage_engine_kit.sh D4R_FOLDER
#   D4R_GLSLANG      glslang release archive or extracted folder (bin/glslangValidator; Khronos 16.6.0 tested)
#   D4R_GLSLANG_LICENSE  its LICENSE.txt (the release archives have none; the tag's LICENSE.txt in KhronosGroup/glslang)
#   D4R_HARNESS_EXE  the D3D12 test app (default build/d3d12_dlss_harness.exe, scripts/build_d3d12_dlss_harness.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D4R="$(realpath "${1:?usage: stage_engine_kit.sh D4R_FOLDER}")"
: "${D4R_GLSLANG:?set D4R_GLSLANG to a glslang release (glslang-16.6.0-linux-x86_64-release.tar.gz)}"
: "${D4R_GLSLANG_LICENSE:?set D4R_GLSLANG_LICENSE to the LICENSE.txt of glslang}"
HARNESS="${D4R_HARNESS_EXE:-$ROOT/build/d3d12_dlss_harness.exe}"
[[ -f "$HARNESS" ]] || { echo "missing $HARNESS (scripts/build_d3d12_dlss_harness.sh)" >&2; exit 2; }
KIT="$D4R/tools/models"
rm -rf "$KIT"
mkdir -p "$KIT/bin" "$KIT/engine/rr" "$KIT/kernels/tools" "$KIT/kernels/rr" "$KIT/vulkan" "$D4R/licenses"
copy() {   # copy SRC_DIR DST_DIR PATTERN...: never tests
  local src=$1 dst=$2 pat f
  shift 2
  for pat in "$@"; do
    for f in "$ROOT/$src"/$pat; do
      [[ -f "$f" ]] || continue
      case "$(basename "$f")" in test_*) continue ;; esac
      cp "$f" "$dst/"
    done
  done
}
copy engine "$KIT/engine" '*.py' '*.comp' '*.glsl'
copy engine/rr "$KIT/engine/rr" '*.py' '*.comp' '*.glsl' '*.txt' '*.npy' '*.sh'
copy kernels/tools "$KIT/kernels/tools" '*.py'
copy kernels/rr "$KIT/kernels/rr" '*.py'
copy vulkan "$KIT/vulkan" '*.py'   # ptx2glsl.py: engine/rr/ptx2rr.py builds on it
# the model compilers import these test modules' packing helpers
cp "$ROOT/engine/test_k.py" "$ROOT/engine/test_m.py" "$KIT/engine/"
cp "$ROOT/engine/rr/test_layer.py" "$KIT/engine/rr/"
cp "$HARNESS" "$KIT/bin/d3d12_dlss_harness.exe"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
GL="$D4R_GLSLANG"
if [[ -f "$GL" ]]; then
  case "$GL" in *.zip) unzip -q "$GL" -d "$TMP" ;; *) tar xf "$GL" -C "$TMP" ;; esac
  GL="$TMP"
fi
cp "$GL/bin/glslangValidator" "$KIT/bin/glslangValidator"
chmod +x "$KIT/bin/glslangValidator"
cp "$D4R_GLSLANG_LICENSE" "$D4R/licenses/glslang-LICENSE.txt"
cp "$ROOT/packaging/build_engine_models.sh" "$D4R/tools/build-engine-models.sh"
cp "$ROOT/packaging/model_tools.py" "$KIT/model_tools.py"
chmod +x "$D4R/tools/build-engine-models.sh"
printf 'Staged the engine-model builder in %s (%s)\n' "$D4R/tools" "$(du -sh "$D4R/tools" | cut -f1)"
