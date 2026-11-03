#!/usr/bin/env python3
"""
Finds out WHERE the arm-position classifier goes wrong: runs it on the validation crops
made by make_pose_crops.py and groups the mistakes by viewing angle, by augmentation
(figure cut off at the edge, mirrored, rotated, blurred ...) and by character.
Also saves a sheet with the wrong crops, labelled "truth -> predicted".

Usage:
  python3 pose_errors.py pose_crops_multi aug_live --pose models/pose_multi.pt
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter, defaultdict

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Needs Pillow:  pip3 install pillow")

ANGLE_RE = re.compile(r"angle_(\d+(?:\.\d+)?)")


def main():
    ap = argparse.ArgumentParser(description="Where does the arm-position classifier make mistakes?")
    ap.add_argument("crops", help="folder made by make_pose_crops.py (has val/<arm_position>/)")
    ap.add_argument("aug", help="folder made by augment_dataset.py (has labels.csv)")
    ap.add_argument("--pose", required=True, help="trained classifier, e.g. models/pose_multi.pt")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="pose_errors.jpg", help="sheet with the wrong crops")
    a = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Needs Ultralytics:  pip3 install ultralytics")

    # crop file name -> the augmented image's labels (angle, augmentations ...)
    info = {}
    with open(os.path.join(a.aug, "labels.csv")) as f:
        for r in csv.DictReader(f):
            stem = r["path"].replace(os.sep, "/").split("/", 1)[1].replace("/", "__")[:-4]
            info[stem] = r

    val = os.path.join(a.crops, "val")
    items = [(os.path.join(val, c, f), c) for c in sorted(os.listdir(val)) if os.path.isdir(os.path.join(val, c))
             for f in sorted(os.listdir(os.path.join(val, c))) if f.endswith(".png")]
    if not items:
        sys.exit(f"No crops in {val}")
    model = YOLO(a.pose, task="classify")
    print(f"Classifying {len(items)} validation crops...")

    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))      # group -> value -> [wrong, total]
    wrong, conf = [], Counter()
    for i in range(0, len(items), 64):
        batch = items[i:i + 64]
        for (path, truth), r in zip(batch, model.predict([p for p, _ in batch], imgsz=a.imgsz,
                                                         device=a.device, verbose=False)):
            pred = model.names[int(r.probs.top1)]
            stem = os.path.basename(path)[:-4]
            row = info.get(stem, {})
            ops = row.get("augmentations", "")
            m = ANGLE_RE.search(stem)
            angle = m.group(1).lstrip("0") or "0" if m else "?"
            groups = {
                "viewing angle": f"{angle:>5}°",
                "cut off at the edge": "yes" if "cutoff" in ops else "no",
                "mirrored": "yes" if row.get("mirrored") == "1" else "no",
                "rotated": "yes" if "rot" in ops else "no",
                "blur / motion blur": "yes" if ("blur" in ops or "motion" in ops) else "no",
                "original photo (no augmentation)": "yes" if ops == "original" else "no",
                "character": row.get("character", "?"),
            }
            bad = pred != truth
            for g, v in groups.items():
                by[g][v][0] += bad
                by[g][v][1] += 1
            if bad:
                wrong.append((path, truth, pred))
                conf[(truth, pred)] += 1

    n, nw = len(items), len(wrong)
    print(f"\n{n - nw}/{n} correct ({(n - nw) / n:.1%})\n")
    for g, vals in by.items():
        print(f"{g}:")
        for v in sorted(vals, key=lambda x: (len(x), x)):
            w, t = vals[v]
            bar = "#" * round(40 * w / max(1, t))
            print(f"  {v:<14} wrong {w:>4}/{t:<5} {w / max(1, t):6.1%}  {bar}")
        print()
    print("Most common mistakes (truth -> predicted):")
    for (t, p), k in conf.most_common(10):
        print(f"  {t:<12} -> {p:<12} {k}")

    if wrong:
        cell, cols = 150, 10
        pick = wrong[:100]
        sheet = Image.new("RGB", (cols * cell, math.ceil(len(pick) / cols) * (cell + 28)), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for i, (p, t, pr) in enumerate(pick):
            with Image.open(p) as im:
                im = im.convert("RGB").resize((cell - 4, cell - 4))
            x, y = (i % cols) * cell, (i // cols) * (cell + 28)
            sheet.paste(im, (x + 2, y + 2))
            m = ANGLE_RE.search(os.path.basename(p))
            d.text((x + 3, y + cell - 1), f"{t} ->", fill=(0, 120, 0))
            d.text((x + 3, y + cell + 12), f"{pr}  ({m.group(1) if m else '?'}°)", fill=(200, 0, 0))
        sheet.save(a.out, quality=88)
        print(f"\nWrong crops (first {len(pick)}): {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
