#!/usr/bin/env python3
"""
Exports the two trained models to NCNN (the fastest format on a Raspberry Pi) and checks
that the exported models give the same answers as the originals.

    <out>/det_ncnn_model/     character detector
    <out>/pose_ncnn_model/    arm-position classifier

Copy the whole <out> folder to the Raspberry Pi.

Usage (on the Mac, after training):
  python3 export_models.py --det runs/detect/character_live/weights/best.pt \\
      --pose runs/classify/pose_live/weights/best.pt --check live_test --out pi_models

Needs:  pip3 install ultralytics   (NCNN export tools are installed automatically on first use)
"""

import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_detect import Recognizer  # noqa: E402
import cv2  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".bmp")


def export(pt, imgsz, task, dst):
    from ultralytics import YOLO
    path = YOLO(pt, task=task).export(format="ncnn", imgsz=imgsz)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(str(path), dst)
    return dst


def check_images(folder, n):
    paths = []
    for root, _, files in os.walk(folder):
        paths += [os.path.join(root, f) for f in files if f.lower().endswith(EXT)]
    random.Random(0).shuffle(paths)
    return paths[:n]


def main():
    ap = argparse.ArgumentParser(description="Export the detector and classifier to NCNN and compare them.")
    ap.add_argument("--det", required=True, help="trained detector best.pt")
    ap.add_argument("--pose", help="trained classifier best.pt")
    ap.add_argument("--det-imgsz", type=int, default=320)
    ap.add_argument("--pose-imgsz", type=int, default=224)
    ap.add_argument("--check", metavar="FOLDER",
                    help="images to compare original and exported models on (e.g. live_test, or a val folder)")
    ap.add_argument("--check-count", type=int, default=60)
    ap.add_argument("--out", default="pi_models", help="folder for the exported models (default pi_models)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    det_dst = export(a.det, a.det_imgsz, "detect", os.path.join(a.out, "det_ncnn_model"))
    print(f"Detector   -> {det_dst}")
    pose_dst = None
    if a.pose:
        pose_dst = export(a.pose, a.pose_imgsz, "classify", os.path.join(a.out, "pose_ncnn_model"))
        print(f"Classifier -> {pose_dst}")

    if a.check:
        imgs = check_images(a.check, a.check_count)
        if not imgs:
            sys.exit(f"No images in {a.check}")
        orig = Recognizer(a.det, a.pose, a.det_imgsz, a.pose_imgsz, verbose=False, device="cpu")
        ncnn = Recognizer(det_dst, pose_dst, a.det_imgsz, a.pose_imgsz, verbose=False, device="cpu")
        orig.warmup()
        ncnn.warmup()
        same_c = same_p = 0
        t = {"orig": [0.0, 0.0], "ncnn": [0.0, 0.0]}
        for p in imgs:
            img = cv2.imread(p)
            r1, r2 = orig(img), ncnn(img)
            t["orig"][0] += orig.det_ms
            t["orig"][1] += orig.pose_ms
            t["ncnn"][0] += ncnn.det_ms
            t["ncnn"][1] += ncnn.pose_ms
            c1 = r1[0][1] if r1 else None
            c2 = r2[0][1] if r2 else None
            same_c += c1 == c2
            same_p += (r1[0][3] if r1 else None) == (r2[0][3] if r2 else None)
        n = len(imgs)
        print(f"\nSame answer on {n} images: character {same_c}/{n}" + (f", arm position {same_p}/{n}" if a.pose else ""))
        for k in ("orig", "ncnn"):
            print(f"  {k:<5} on this computer's CPU: detector {t[k][0] / n:.0f} ms"
                  + (f", classifier {t[k][1] / n:.0f} ms" if a.pose else ""))
        if same_c < n * 0.95 or (a.pose and same_p < n * 0.95):
            print("  WARNING: the exported models disagree with the originals more than expected.")
    print(f"\nCopy the folder '{a.out}' to the Raspberry Pi, e.g.:  scp -r {a.out} <user>@<pi-ip>:~/")


if __name__ == "__main__":
    main()
