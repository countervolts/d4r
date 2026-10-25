#!/usr/bin/env bash
# Build a pinned OptiScaler-only upgrade for an existing d4r proxy installation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${1:-$ROOT/dist/optiscaler-v10}")"
NAME=d4r-optiscaler-v10-20261024-upgrade
mkdir -p "$OUT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
archive="$(python3 "$ROOT/scripts/fetch_optiscaler.py")"
7z x -y -o"$TMP/upstream" "$archive" >/dev/null
stage="$TMP/$NAME"
mkdir -p "$stage"
python3 "$ROOT/scripts/stage_optiscaler.py" "$TMP/upstream" "$stage"
cp "$ROOT/packaging/d4r-features.py" "$ROOT/packaging/d4r-check.sh" "$stage/d4r/"
cp "$ROOT/LICENSE" "$stage/d4r/licenses/OptiScaler-LICENSE-GPL-3.0.txt"
cp "$ROOT/packaging/OPTISCALER_UPGRADE.txt" "$stage/D4R_OPTISCALER_UPGRADE.txt"
python3 "$ROOT/scripts/check_redistributable.py" release clean "$stage"
python3 - "$stage" "$OUT/$NAME.zip" <<'PY'
from pathlib import Path
import sys
import zipfile
stage, target = map(Path, sys.argv[1:])
with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(stage.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(stage))
PY
(cd "$OUT" && sha256sum "$NAME.zip" > "$NAME.zip.sha256")
printf 'OptiScaler upgrade: %s/%s.zip\n' "$OUT" "$NAME"
