#!/usr/bin/env python3
"""
Builds an image-classification dataset of arm positions from the output of
augment_dataset.py: every figure is cropped to a square around it (exactly like
live_detect.py crops the box found by the character detector) and sorted by class.

    <out>/train/<arm_position>/*.png
    <out>/val/<arm_position>/*.png
    <out>/preview_crops.jpg

Each class contains the photos of ALL characters, so the model learns what an arm
position looks like on any figure. Mirrored images were already relabelled
(left <-> right) by augment_dataset.py, so train with fliplr=0.

For use on the live stream add --random-bg (and --backgrounds <folder>) and --low-res,
the same as for the character detector.

Needs Pillow:  pip3 install pillow

Usage:
  python3 make_pose_crops.py <augmented folder> <crops folder> --random-bg --low-res

Then train (pip3 install ultralytics):
  yolo classify train data=<crops folder> model=yolo11n-cls.pt imgsz=224 epochs=50 fliplr=0
"""

import argparse
import csv
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("This tool needs Pillow:  pip3 install pillow")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_effects as fx  # noqa: E402


def class_name(row, scheme):
    return row["arm_position"] if scheme == "pose" else f"{row['character']}__{row['arm_position']}"


def crop_one(job):
    row, src, out, cls, opts, bg_photos = job
    split = "val" if row["split"] == "test" else "train"
    stem = row["path"].replace(os.sep, "/").split("/", 1)[1].replace("/", "__")[:-4]
    rng = fx.seeded_rng(opts.seed, row["path"])
    with Image.open(os.path.join(src, row["path"])) as im:
        im = im.convert("RGBA")
    box = im.split()[3].point(lambda v: 255 if v >= opts.alpha_min else 0).getbbox()
    if not box:
        return None, f"no figure found, skipped: {row['path']}"

    # Whole frame first (background + stream degradation), then the crop: same order as live.
    if opts.random_bg:
        bg = fx.random_background(im.size, rng, bg_photos, opts.white_fraction, opts.bg)
    else:
        bg = Image.new("RGB", im.size, opts.bg)
    img = fx.composite(im, bg)
    if opts.low_res and rng.random() < opts.low_res_prob:
        img, _ = fx.low_res(img, rng, box[3] - box[1], opts.fig_px)

    jittered = fx.jitter_box(box, rng, opts.jitter)
    margin = opts.margin + rng.uniform(-0.05, 0.05)
    crop = img.crop(fx.square_crop_box(jittered, img.size[0], img.size[1], max(0.0, margin)))
    crop = crop.resize((opts.size, opts.size), Image.BILINEAR)

    folder = os.path.join(out, split, cls)
    os.makedirs(folder, exist_ok=True)
    crop.save(os.path.join(folder, stem + ".png"), compress_level=6)
    return (split, cls, os.path.join(split, cls, stem + ".png")), None


def preview(out, done, n=48, cols=8, cell=140, seed=0):
    rng = random.Random(seed)
    pick = rng.sample(done, min(n, len(done)))
    sheet = Image.new("RGB", (cols * cell, math.ceil(len(pick) / cols) * (cell + 16)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (_, cls, rel) in enumerate(pick):
        with Image.open(os.path.join(out, rel)) as im:
            im = im.convert("RGB").resize((cell - 4, cell - 4))
        x, y = (i % cols) * cell, (i // cols) * (cell + 16)
        sheet.paste(im, (x + 2, y + 2))
        d.text((x + 3, y + cell), cls[:26], fill=(0, 0, 0))
    path = os.path.join(out, "preview_crops.jpg")
    sheet.save(path, quality=88)
    return path


def main():
    ap = argparse.ArgumentParser(description="Make a cropped arm-position classification dataset.")
    ap.add_argument("src", help="folder made by augment_dataset.py (contains labels.csv)")
    ap.add_argument("out", help="classification dataset folder to create")
    ap.add_argument("--classes", choices=["pose", "character_pose"], default="pose",
                    help="pose (default): one class per arm position, shared by all characters")
    ap.add_argument("--size", type=int, default=224, help="crop size in px (default 224)")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="space around the figure, fraction of its size (default 0.10, as in live_detect.py)")
    ap.add_argument("--jitter", type=float, default=0.05,
                    help="random box error, fraction of its size: imitates the detector (default 0.05)")
    ap.add_argument("--background", default="255,255,255", help="background colour R,G,B (default white)")
    ap.add_argument("--random-bg", action="store_true", help="random backgrounds instead of plain colour")
    ap.add_argument("--backgrounds", metavar="FOLDER", help="your own background photos for --random-bg")
    ap.add_argument("--white-fraction", type=float, default=0.2,
                    help="with --random-bg: share of crops that keep the plain background (default 0.2)")
    ap.add_argument("--low-res", action="store_true", help="degrade like small live stream frames")
    ap.add_argument("--low-res-prob", type=float, default=0.8, help="share of crops degraded (default 0.8)")
    ap.add_argument("--fig-px", default="80,240",
                    help="with --low-res: figure height range in stream pixels (default 80,240)")
    ap.add_argument("--alpha-min", type=int, default=64, help="alpha from which a pixel counts as figure")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="parallel processes")
    args = ap.parse_args()

    labels_csv = os.path.join(args.src, "labels.csv")
    if not os.path.exists(labels_csv):
        sys.exit(f"No labels.csv in {args.src}: run augment_dataset.py first.")
    rows = list(csv.DictReader(open(labels_csv)))
    args.bg = tuple(int(v) for v in args.background.split(","))
    args.fig_px = tuple(float(v) for v in args.fig_px.split(","))
    bg_photos = fx.load_backgrounds(args.backgrounds) if args.random_bg else []
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    classes = sorted({class_name(r, args.classes) for r in rows})
    print(f"{len(rows)} images -> {args.size}x{args.size} crops, {len(classes)} classes ({args.classes})")
    jobs = [(r, args.src, out, class_name(r, args.classes), args, bg_photos) for r in rows]
    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (res, err) in enumerate(ex.map(crop_one, jobs, chunksize=16), 1):
            if err:
                print("  " + err)
            else:
                done.append(res)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} crops done")

    print(f"\n{'class':<28}{'train':>7}{'val':>6}")
    for c in classes:
        tr = sum(1 for s, k, _ in done if k == c and s == "train")
        va = sum(1 for s, k, _ in done if k == c and s == "val")
        print(f"{c:<28}{tr:>7}{va:>6}")
    print(f"\nDataset: {out}\nPreview: {preview(out, done)}")
    print(f"\nTrain with:  yolo classify train data={out} model=yolo11n-cls.pt imgsz={args.size} epochs=50 fliplr=0")


if __name__ == "__main__":
    main()
