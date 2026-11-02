#!/usr/bin/env bash
# Fetches the ROCm user-space runtime the release bundles in d4r/rocm: HIP, HSA, comgr and
# rocprofiler-register from AMD's ROCm 7.2.4 packages for Ubuntu 22.04, plus the libelf and libnuma
# they need, which Steam's container runtime lacks. These builds need only glibc 2.34, so they run
# in SteamLinuxRuntime_4 on any distribution (a rolling distribution's ROCm may need a newer glibc).
#
# usage: scripts/fetch_rocm_runtime.sh [OUT_DIR]          (default: ~/.cache/d4r-rocm-runtime)
# Result: OUT_DIR/lib/<soname> for each library and OUT_DIR/licenses/.
set -euo pipefail

OUT="$(realpath -m "${1:-$HOME/.cache/d4r-rocm-runtime}")"
AMD=https://repo.radeon.com/rocm/apt/7.2.4/pool/main
UBUNTU=http://archive.ubuntu.com/ubuntu/pool/main
# url sha256
PACKAGES=(
  "$AMD/h/hip-runtime-amd/hip-runtime-amd_7.2.53211.70204-93~22.04_amd64.deb 2b0fc15d4456bc35370f33672b202b43f2af1aceac47a42c22d1370b2f46968f"
  "$AMD/c/comgr/comgr_3.0.0.70204-93~22.04_amd64.deb 58460d117131369a248d334ba749609abea8591cd05752ac572c9a10f31ac006"
  "$AMD/h/hsa-rocr/hsa-rocr_1.18.0.70204-93~22.04_amd64.deb 7fccbb91e38fadd126538f15e02c66e20db717a60eb03f2a045e47497bf67405"
  "$AMD/r/rocprofiler-register/rocprofiler-register_0.6.0.70204-93~22.04_amd64.deb c2780e37a7217329b5d3d40e682ae6979481b1a59b86415a0cdf849b9c142bfd"
  "$UBUNTU/e/elfutils/libelf1_0.186-1ubuntu0.1_amd64.deb 2cc781614b045b450a1e073ee3b0902f393ee58ede1d2ea4526ad805fcccf538"
  "$UBUNTU/n/numactl/libnuma1_2.0.14-3ubuntu2_amd64.deb 0721c89001fbbd1ada23e89da5d60e762763c1a7b3dc814a2e9a518480a8043d"
)
# soname, the file it names inside the packages
LIBRARIES=(
  "libamdhip64.so.7 opt/rocm-7.2.4/lib/libamdhip64.so.7"
  "libamd_comgr.so.3 opt/rocm-7.2.4/lib/libamd_comgr.so.3"
  "libhsa-runtime64.so.1 opt/rocm-7.2.4/lib/libhsa-runtime64.so.1"
  "librocprofiler-register.so.0 opt/rocm-7.2.4/lib/librocprofiler-register.so.0"
  "libelf.so.1 usr/lib/x86_64-linux-gnu/libelf.so.1"
  "libnuma.so.1 usr/lib/x86_64-linux-gnu/libnuma.so.1"
)
LICENSES=(
  "HIP-LICENSE.md opt/rocm-7.2.4/share/doc/hip/LICENSE.md"
  "comgr-LICENSE.txt opt/rocm-7.2.4/share/doc/amd_comgr/LICENSE.txt"
  "ROCr-LICENSE.md opt/rocm-7.2.4/share/doc/hsa-rocr/LICENSE.md"
  "rocprofiler-register-LICENSE.md opt/rocm-7.2.4/share/doc/rocprofiler-register/LICENSE.md"
  "libelf-copyright.txt usr/share/doc/libelf1/copyright"
  "libnuma-copyright.txt usr/share/doc/libnuma1/copyright"
)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/root"
for entry in "${PACKAGES[@]}"; do
  read -r url sha <<< "$entry"
  deb="$TMP/$(basename "$url")"
  curl -fsSL -o "$deb" "$url"
  echo "$sha  $deb" | sha256sum -c --quiet
  data="$(ar t "$deb" | grep '^data\.tar')"
  ar p "$deb" "$data" | tar -x -C "$TMP/root" -f - --use-compress-program="$(case "$data" in *.zst) echo zstd -d ;; *.xz) echo xz -d ;; *) echo gzip -d ;; esac)"
done

rm -rf "$OUT"
mkdir -p "$OUT/lib" "$OUT/licenses"
for entry in "${LIBRARIES[@]}"; do
  read -r soname path <<< "$entry"
  cp -L "$TMP/root/$path" "$OUT/lib/$soname"
done
for entry in "${LICENSES[@]}"; do
  read -r name path <<< "$entry"
  cp "$TMP/root/$path" "$OUT/licenses/$name"
done
printf 'ROCm runtime in %s (%s)\n' "$OUT" "$(du -sh "$OUT" | cut -f1)"
