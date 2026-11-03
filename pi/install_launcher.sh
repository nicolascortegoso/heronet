#!/bin/bash
# Puts a HeroNet icon on the Raspberry Pi desktop and in the applications menu.
#
#   pi/install_launcher.sh                      # English, camera at heronet-cam.local
#   pi/install_launcher.sh --lang es            # Spanish labels
#   pi/install_launcher.sh --camera 192.168.0.10
#   pi/install_launcher.sh --autostart          # also start HeroNet when the Pi starts
#   pi/install_launcher.sh --remove             # take the icon (and autostart) away again

set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
LANG_UI="" ; CAMERA="" ; AUTOSTART=0 ; REMOVE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --lang) LANG_UI="$2"; shift 2 ;;
        --camera) CAMERA="$2"; shift 2 ;;
        --autostart) AUTOSTART=1; shift ;;
        --remove) REMOVE=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
TARGETS=("$DESKTOP_DIR/HeroNet.desktop" "$HOME/.local/share/applications/HeroNet.desktop")
AUTO="$HOME/.config/autostart/HeroNet.desktop"

if [ "$REMOVE" = 1 ]; then
    rm -f "${TARGETS[@]}" "$AUTO"
    echo "HeroNet icon removed."
    exit 0
fi

chmod +x "$DIR/pi/heronet.sh"
[ -n "$LANG_UI" ] && sed -i "s/^LANGUAGE=\"[^\"]*\"/LANGUAGE=\"$LANG_UI\"/" "$DIR/pi/heronet.sh"
[ -n "$CAMERA" ] && sed -i "s/^CAMERA=\"[^\"]*\"/CAMERA=\"$CAMERA\"/" "$DIR/pi/heronet.sh"

ENTRY="[Desktop Entry]
Type=Application
Name=HeroNet
Comment=Who is that hero?
Exec=$DIR/pi/heronet.sh
Icon=$DIR/pi/heronet.png
Terminal=false
Categories=Game;Education;"

mkdir -p "$DESKTOP_DIR" "$HOME/.local/share/applications"
for t in "${TARGETS[@]}"; do
    printf '%s\n' "$ENTRY" > "$t"
    chmod +x "$t"
    gio set "$t" metadata::trusted true 2>/dev/null || true
done
echo "Icon added to the desktop and to the menu (Games)."

if [ "$AUTOSTART" = 1 ]; then
    mkdir -p "$HOME/.config/autostart"
    printf '%s\n' "$ENTRY" > "$AUTO"
    echo "HeroNet will also start by itself when the Pi starts."
fi
grep -E '^(CAMERA|LANGUAGE)=' "$DIR/pi/heronet.sh"
