#!/bin/bash
# Starts HeroNet full screen on the Raspberry Pi. Used by the HeroNet desktop icon.
#
# Settings (change here, or run install_launcher.sh again with --camera / --lang):
CAMERA="heronet-cam.local"   # the ESP32-CAM: its name, or its IP address (e.g. 192.168.0.10)
LANGUAGE="en"                # labels: en (English) or es (español)
SIZE="vga"                   # stream size: vga (640x480) or qvga (320x240; striped on some cameras)
EXTRA_OPTIONS=""             # e.g. "--pan-invert" if the left/right arrows move the camera the wrong way,
                             #      "--tilt-invert" for up/down, "--no-numbers" to hide percentages

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

echo "$(date '+%F %T') starting: camera $CAMERA ($SIZE), models $MODELS, language $LANGUAGE" >> "$LOG"
exec python3 scripts/live_detect.py "$CAMERA" --det "$MODELS/det_ncnn_model" "${POSE[@]}" \
    --size "$SIZE" --kids --lang "$LANGUAGE" $EXTRA_OPTIONS "$@" >> "$LOG" 2>&1
