#!/usr/bin/env bash
set -euo pipefail
# One-command launch: runs the game configured in d4r.ini's [Launch] section with all d4r settings
# from the same file (see config/d4r.ini.default). Extra arguments go to the game.
#   D4R_CONFIG=/path/to/other.ini scripts/d4r_play.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mapfile -t LAUNCH < <(python3 "$ROOT/scripts/d4r_config.py" --launch) || exit 2
[[ ${#LAUNCH[@]} -eq 3 ]] || exit 2
exec "$ROOT/scripts/run_game_d4r_optiscaler_proton.sh" "${LAUNCH[0]}" "${LAUNCH[1]}" "${LAUNCH[2]}" "$@"
