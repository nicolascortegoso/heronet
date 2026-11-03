#!/usr/bin/env python3
"""
Saves frames from the ESP32-CAM live stream, exactly as the camera sends them.
No model needed. Two uses:

  Empty scene (backgrounds + "no figure" examples for training):
    python3 grab_frames.py <ESP32-IP> empty_scene --count 30 --interval 1
    -> while it runs, change the light a little, move the turntable, etc.

  Real test frames (to measure accuracy on the real camera with evaluate_live.py):
    python3 grab_frames.py <ESP32-IP> live_test/batman/both_up --count 3 --enter
    -> place the figure, press Enter for each frame (turn it between frames)

Folders are created as needed. For test frames use <character>/<arm_position> with the
same names as the models (the dataset's folder names).

Needs:  pip3 install opencv-python   (already installed with ultralytics)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_detect import StreamReader, stream_url, ROTATIONS  # noqa: E402
import cv2  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Save raw frames from the ESP32-CAM stream.")
    ap.add_argument("ip", help="ESP32-CAM IP address")
    ap.add_argument("out", help="folder to save into, e.g. empty_scene or live_test/batman/both_up")
    ap.add_argument("--count", type=int, default=10, help="number of frames to save (default 10)")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between frames (default 1)")
    ap.add_argument("--enter", action="store_true", help="save a frame each time you press Enter")
    ap.add_argument("--size", default="qvga", choices=["qqvga", "qvga", "hvga", "vga", "svga"],
                    help="stream size, the same as for live_detect.py (default qvga)")
    ap.add_argument("--quality", type=int, default=12, help="stream JPEG quality (default 12)")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="rotate frames, the same as for live_detect.py")
    ap.add_argument("--timeout", type=float, default=10)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    reader = StreamReader(stream_url(a.ip, a.size, a.quality), a.timeout)
    reader.start()
    frame, jpeg, last = reader.latest(0, timeout=a.timeout)
    if frame is None:
        reader.stop()
        sys.exit(f"No frames from {a.ip}: is the stream sketch running, and no other stream open?")
    print(f"Connected: {frame.shape[1]}x{frame.shape[0]} frames. Saving {a.count} to {a.out}/")

    saved = 0
    try:
        while saved < a.count:
            if a.enter:
                input(f"  [{saved + 1}/{a.count}] press Enter to save a frame (Ctrl+C to stop) ")
            elif saved:
                time.sleep(a.interval)
            frame, jpeg, last = reader.latest(last, timeout=a.timeout)
            if frame is None:
                print("  no new frame, retrying...")
                continue
            name = os.path.join(a.out, f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{saved:03d}")
            if ROTATIONS[a.rotate] is None:
                with open(name + ".jpg", "wb") as f:      # the camera's own JPEG, untouched
                    f.write(jpeg)
                path = name + ".jpg"
            else:
                path = name + ".png"                      # rotated: saved without extra compression
                cv2.imwrite(path, cv2.rotate(frame, ROTATIONS[a.rotate]))
            saved += 1
            print(f"  saved {path}")
    except KeyboardInterrupt:
        print()
    finally:
        reader.stop()
    print(f"{saved} frames in {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
