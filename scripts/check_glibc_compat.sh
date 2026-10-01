#!/usr/bin/env bash
# Reject Linux release libraries that need a newer GLIBC than the target runtime.
# Usage: check_glibc_compat.sh MAX_GLIBC LIBRARY [LIBRARY ...]
set -euo pipefail

if [[ $# -lt 2 || ! "$1" =~ ^[0-9]+\.[0-9]+$ ]]; then
  printf 'usage: %s MAX_GLIBC LIBRARY [LIBRARY ...]\n' "$0" >&2
  exit 2
fi
max="GLIBC_$1"
shift
command -v readelf >/dev/null || { echo 'readelf is required to check GLIBC compatibility' >&2; exit 2; }

failed=0
for library in "$@"; do
  if [[ ! -f "$library" ]]; then
    printf 'missing library: %s\n' "$library" >&2
    failed=1
    continue
  fi
  if ! versions="$(readelf -W --version-info "$library" 2>&1)"; then
    printf 'cannot inspect %s: %s\n' "$library" "$versions" >&2
    failed=1
    continue
  fi
  required="$(printf '%s\n' "$versions" | grep -Eo 'GLIBC_[0-9]+\.[0-9]+' | sort -Vu | tail -1 || true)"
  if [[ -n "$required" && "$(printf '%s\n%s\n' "$max" "$required" | sort -Vu | tail -1)" != "$max" ]]; then
    printf '%s requires %s (target: at most %s)\n' "$library" "$required" "$max" >&2
    failed=1
  fi
done
exit "$failed"
