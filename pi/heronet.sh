#!/bin/bash
# Starts HeroNet full screen on the Raspberry Pi. Used by the HeroNet desktop icon.
#
# Settings (change here, or run install_launcher.sh again with --camera / --lang):
CAMERA="heronet-cam.local"   # the ESP32-CAM: its name, or its IP address (e.g. 192.168.0.10)
LANGUAGE="en"                # labels: en (English) or es (español)

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR" || exit 1
LOG="$DIR/heronet.log"

say() {                       # a message box the kid (or you) can see, plus a line in the log
    echo "$(date '+%F %T') $1" >> "$LOG"
    if command -v zenity >/dev/null; then zenity --error --title="HeroNet" --text="$1" 2>/dev/null
    elif command -v notify-send >/dev/null; then notify-send "HeroNet" "$1"
    fi
}

if [ ! -f .venv/bin/activate ]; then
    say "HeroNet is not installed yet: the .venv folder is missing in $DIR."
    exit 1
fi
source .venv/bin/activate

MODELS=""
for m in hf/ncnn pi_models_v2 pi_models models_ncnn; do
    if [ -d "$m/det_ncnn_model" ]; then MODELS="$m"; break; fi
done
if [ -z "$MODELS" ]; then
    say "HeroNet can't find its models (hf/ncnn/det_ncnn_model)."
    exit 1
fi
POSE=()
[ -d "$MODELS/pose_ncnn_model" ] && POSE=(--pose "$MODELS/pose_ncnn_model")

echo "$(date '+%F %T') starting: camera $CAMERA, models $MODELS, language $LANGUAGE" >> "$LOG"
exec python3 scripts/live_detect.py "$CAMERA" --det "$MODELS/det_ncnn_model" "${POSE[@]}" \
    --kids --lang "$LANGUAGE" "$@" >> "$LOG" 2>&1
