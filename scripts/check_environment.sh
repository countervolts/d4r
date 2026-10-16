#!/usr/bin/env bash
set -euo pipefail

section() {
  printf '\n## %s\n' "$1"
}

show_if_available() {
  local tool="$1"
  shift
  if command -v "$tool" >/dev/null 2>&1; then
    "$tool" "$@" 2>&1 || true
  else
    printf '%s: not found\n' "$tool"
  fi
}

section 'Operating system'
uname -a

section 'Display adapters and kernel drivers'
show_if_available lspci -nnk | rg -A4 -i 'vga|3d|display' || true

section 'ROCm / HIP'
show_if_available rocminfo | head -80 || true
show_if_available hipcc --version | head -20 || true
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname --showdriverversion 2>&1 | head -50 || true
elif [[ -x /opt/rocm/bin/rocm-smi ]]; then
  /opt/rocm/bin/rocm-smi --showproductname --showdriverversion 2>&1 | head -50 || true
fi

section 'Vulkan'
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary 2>&1 | rg -i 'VULKANINFO|Instance Version|GPU[0-9]|apiVersion|driverVersion|vendorID|deviceID|deviceName|driverID|driverName|driverInfo' || true
else
  printf 'vulkaninfo: not found\n'
fi

section 'Wine / Proton'
show_if_available wine --version
show_if_available protontricks --version
find "$HOME/.local/share/Steam/compatibilitytools.d" -maxdepth 3 -type f -name proton -print 2>/dev/null | head -20 || true

section 'Build and PE inspection tools'
for tool in x86_64-w64-mingw32-g++ cmake ninja llvm-objdump llvm-readobj objdump strings; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%-28s %s\n' "$tool" "$(command -v "$tool")"
  else
    printf '%-28s %s\n' "$tool" 'not found'
  fi
done

section 'Installed relevant Arch packages'
if command -v pacman >/dev/null 2>&1; then
  pacman -Q 2>/dev/null | rg -i '(^| )(rocm|hip|vulkan|wine|mingw|clang|mesa)(-| )' || true
else
  printf 'pacman: not found\n'
fi
