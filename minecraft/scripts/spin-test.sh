#!/usr/bin/env bash
# Motion quality test: the dev client spins the camera (autoSpin deg/tick) and screenshots the upscaled frame at a fixed game tick.
# usage: scripts/spin-test.sh TAG key=value ...   (extra config keys, e.g. motionSign=-1 preset=native)
set -euo pipefail
cd "$(dirname "$0")/.."
tag="$1"; shift
mkdir -p run/config run/screenshots
{ printf 'enabled=true\npreset=quality\nlog=true\nautoWorld=D4R_Test\nautoSpin=${SPIN:-0}\nautoStrafe=${STRAFE:-false}\nrenderScale=1.1\nscreenshotTick=100\ntag=%s_\n' "$tag"; for kv in "$@"; do echo "$kv"; done; } > run/config/d4r-minecraft.properties
log=build/spin-$tag.log
./gradlew --no-daemon runClient > "$log" 2>&1 &
until ls run/screenshots/d4r_sr_${tag}_* >/dev/null 2>&1 || grep -q "BUILD FAILED" "$log"; do sleep 3; done
sleep 4
for p in $(pgrep -f "[d]evlaunchinjector") $(pgrep -f "[G]radleWrapperMain"); do kill $p 2>/dev/null || true; done
sleep 4
