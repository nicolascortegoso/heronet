#!/usr/bin/env python3
"""
Turns the output of augment_dataset.py (transparent PNGs + labels.csv) into a YOLO
detection dataset. Bounding boxes come from the figures' non-transparent pixels, so
no manual annotation is needed.

    <out>/data.yaml                 class names and paths, ready for Ultralytics YOLO
    <out>/classes.txt               one class name per line (class id = line number - 1)
    <out>/images/train/*.png        <out>/images/val/*.png   (or .jpg with --format jpg)
    <out>/labels/train/*.txt        <out>/labels/val/*.txt   (class cx cy w h, normalised)
    <out>/preview_boxes.jpg         random samples with their boxes drawn, to check by eye

Classes (--classes):
    character_pose  (default)  one class per character + arm position, e.g. batman__left_above
    character                  one class per character
    pose                       one class per arm position

For a model that runs on the live camera stream (no background removal), use:
    --random-bg          figures on random backgrounds instead of plain white
                         (add --backgrounds <folder> with photos of your rig / desk: best)
    --low-res            images degraded like small stream frames (QVGA/VGA + JPEG)
    --negatives <folder> frames of the empty scene (grab_frames.py): added with no boxes,
                         so the detector learns "no figure here"

The "test" split of augment_dataset.py becomes YOLO's "val" split.

Needs Pillow:  pip3 install pillow

Usage:
  python3 make_yolo_dataset.py <augmented folder> <yolo folder> --classes character
  python3 make_yolo_dataset.py <augmented folder> <yolo folder> --classes character --random-bg --low-res

Then train, for example (pip3 install ultralytics):
  yolo detect train data=<yolo folder>/data.yaml model=yolo11n.pt imgsz=320 epochs=60
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
    if scheme == "character":
        return row["character"]
    if scheme == "pose":
        return row["arm_position"]
    return f"{row['character']}__{row['arm_position']}"


def convert(job):
    """One image: put it on a background, optionally degrade it, save it, write the YOLO label."""
    row, src, out, cls_id, opts, bg_photos = job
    split = "val" if row["split"] == "test" else "train"
    stem = row["path"].replace(os.sep, "/").split("/", 1)[1].replace("/", "__")[:-4]   # unique name
    rng = fx.seeded_rng(opts.seed, row["path"])
    with Image.open(os.path.join(src, row["path"])) as im:
        im = im.convert("RGBA")
    w, h = im.size
    box = im.split()[3].point(lambda v: 255 if v >= opts.alpha_min else 0).getbbox()
    if not box:
        return None, f"no figure found, skipped: {row['path']}"
    x0, y0, x1, y1 = box

    if opts.random_bg:
        bg = fx.random_background(im.size, rng, bg_photos, opts.white_fraction, opts.bg)
    else:
        bg = Image.new("RGB", im.size, opts.bg)
    img = fx.composite(im, bg)
    if opts.low_res and rng.random() < opts.low_res_prob:
        img, _ = fx.low_res(img, rng, y1 - y0, opts.fig_px)

    path = os.path.join(out, "images", split, f"{stem}.{opts.format}")
    if opts.format == "png":
        img.save(path, compress_level=6)
    else:
        img.save(path, quality=opts.quality)
    cx, cy, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
    with open(os.path.join(out, "labels", split, stem + ".txt"), "w") as f:
        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return (split, f"{stem}.{opts.format}", cls_id), None


def make_negative(job):
    """Variants of one empty-scene frame, saved with an empty label file (= no figure)."""
    path, idx, out, opts = job
    rng = fx.seeded_rng(opts.seed, "neg:" + path)
    split = "val" if rng.random() < opts.negative_val else "train"
    with Image.open(path) as im:
        im = im.convert("RGB")
    results = []
    for k in range(opts.negative_copies):
        img = im
        if k:                                               # copy 0 = the frame as it is
            w, h = img.size
            f = rng.uniform(0.8, 1.0)
            cw, ch = int(w * f), int(h * f)
            x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BILINEAR)
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = fx.ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
            img = fx.ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
            img, _ = fx.low_res(img, rng, img.size[1], (img.size[1] * 0.6, img.size[1]))
        stem = f"negative_{idx:04d}_{k:02d}"
        dst = os.path.join(out, "images", split, f"{stem}.{opts.format}")
        if opts.format == "png":
            img.save(dst, compress_level=6)
        else:
            img.save(dst, quality=opts.quality)
        open(os.path.join(out, "labels", split, stem + ".txt"), "w").close()
        results.append(split)
    return results


def preview(out, done, names, n=24, cols=6, cell=200, seed=0):
    rng = random.Random(seed)
    pick = rng.sample(done, min(n, len(done)))
    sheet = Image.new("RGB", (cols * cell, math.ceil(len(pick) / cols) * (cell + 18)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (split, fname, cls_id) in enumerate(pick):
        stem = fname.rsplit(".", 1)[0]
        with Image.open(os.path.join(out, "images", split, fname)) as im:
            im = im.convert("RGB")
            w, h = im.size
            with open(os.path.join(out, "labels", split, stem + ".txt")) as f:
                _, cx, cy, bw, bh = f.read().split()
            cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
            dd = ImageDraw.Draw(im)
            dd.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                         outline=(255, 0, 0), width=max(2, w // 150))
            im = im.resize((cell - 6, cell - 6))
        x, y = (i % cols) * cell, (i // cols) * (cell + 18)
        sheet.paste(im, (x + 3, y + 3))
        d.text((x + 4, y + cell - 2), names[cls_id][:34], fill=(0, 0, 0))
    path = os.path.join(out, "preview_boxes.jpg")
    sheet.save(path, quality=88)
    return path


def main():
    ap = argparse.ArgumentParser(description="Convert augment_dataset.py output into a YOLO dataset.")
    ap.add_argument("src", help="folder made by augment_dataset.py (contains labels.csv)")
    ap.add_argument("out", help="YOLO dataset folder to create")
    ap.add_argument("--classes", choices=["character_pose", "character", "pose"], default="character_pose",
                    help="what each box is labeled with (default: character_pose)")
    ap.add_argument("--background", default="255,255,255", help="background colour R,G,B (default white)")
    ap.add_argument("--random-bg", action="store_true",
                    help="random backgrounds instead of plain colour (for models used on live frames)")
    ap.add_argument("--backgrounds", metavar="FOLDER",
                    help="your own background photos to use with --random-bg (e.g. the empty rig, your desk)")
    ap.add_argument("--white-fraction", type=float, default=0.2,
                    help="with --random-bg: share of images that keep the plain background (default 0.2)")
    ap.add_argument("--low-res", action="store_true",
                    help="degrade images like small live stream frames (shrink + enlarge + JPEG)")
    ap.add_argument("--low-res-prob", type=float, default=0.8, help="share of images degraded (default 0.8)")
    ap.add_argument("--fig-px", default="80,240",
                    help="with --low-res: figure height range in stream pixels (default 80,240 = QVGA..VGA)")
    ap.add_argument("--negatives", metavar="FOLDER",
                    help="frames of the empty scene, added as images without any figure")
    ap.add_argument("--negative-copies", type=int, default=4,
                    help="variants per empty-scene frame (default 4)")
    ap.add_argument("--negative-val", type=float, default=0.2,
                    help="share of empty-scene frames used for validation (default 0.2)")
    ap.add_argument("--format", choices=["png", "jpg"], default="png", help="image format (default png)")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality, only with --format jpg (default 92)")
    ap.add_argument("--alpha-min", type=int, default=64,
                    help="transparency level (0-255) from which a pixel counts as figure (default 64)")
    ap.add_argument("--seed", type=int, default=42, help="random seed: same seed = same dataset")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="parallel processes")
    args = ap.parse_args()

    labels_csv = os.path.join(args.src, "labels.csv")
    if not os.path.exists(labels_csv):
        sys.exit(f"No labels.csv in {args.src}: run augment_dataset.py first.")
    rows = list(csv.DictReader(open(labels_csv)))
    args.bg = tuple(int(v) for v in args.background.split(","))
    args.fig_px = tuple(float(v) for v in args.fig_px.split(","))
    bg_photos = fx.load_backgrounds(args.backgrounds) if args.random_bg else []
    if args.backgrounds and not args.random_bg:
        print("Note: --backgrounds is only used together with --random-bg")

    names = sorted({class_name(r, args.classes) for r in rows})
    ids = {n: i for i, n in enumerate(names)}
    out = os.path.abspath(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    print(f"{len(rows)} images, {len(names)} classes ({args.classes})")
    if args.random_bg:
        print(f"Backgrounds: random ({len(bg_photos)} photos of yours + generated, "
              f"{args.white_fraction:.0%} plain)")
    if args.low_res:
        print(f"Low-res: {args.low_res_prob:.0%} of images, figure {args.fig_px[0]:.0f}-{args.fig_px[1]:.0f} px tall")
    jobs = [(r, args.src, out, ids[class_name(r, args.classes)], args, bg_photos) for r in rows]
    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (res, err) in enumerate(ex.map(convert, jobs, chunksize=16), 1):
            if err:
                print("  " + err)
            else:
                done.append(res)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} images done")

    neg = {"train": 0, "val": 0}
    if args.negatives:
        neg_paths = fx.load_backgrounds(args.negatives)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for splits in ex.map(make_negative, [(p, i, out, args) for i, p in enumerate(neg_paths)]):
                for s in splits:
                    neg[s] += 1
        print(f"  {len(neg_paths)} empty-scene frames -> {neg['train']} train + {neg['val']} val images without figures")

    with open(os.path.join(out, "classes.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {out}\ntrain: images/train\nval: images/val\n\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: \"{n}\"\n")          # quoted: names may contain spaces

    counts = {"train": {}, "val": {}}
    for split, _, c in done:
        counts[split][c] = counts[split].get(c, 0) + 1
    tr, va = sum(counts["train"].values()), sum(counts["val"].values())
    print(f"\nTrain: {tr} images, val: {va} images" + (f" (+ {neg['train']} / {neg['val']} empty)" if args.negatives else ""))
    missing_val = [names[i] for i in range(len(names)) if i not in counts["val"]]
    if missing_val:
        print(f"  Note: {len(missing_val)} classes have no val images (few photos per class): "
              + ", ".join(missing_val[:8]) + (" ..." if len(missing_val) > 8 else ""))
    lo = min(counts["train"].get(i, 0) for i in range(len(names)))
    hi = max(counts["train"].get(i, 0) for i in range(len(names)))
    print(f"  Train images per class: {lo} to {hi}")
    print(f"\nDataset: {out}\nConfig:  {os.path.join(out, 'data.yaml')}")
    print(f"Preview: {preview(out, done, names)}")
    flip = " fliplr=0" if args.classes != "character" else ""
    print("\nTrain with:  yolo detect train data=" + os.path.join(out, "data.yaml")
          + f" model=yolo11n.pt imgsz=320 epochs=60{flip}")


if __name__ == "__main__":
    main()
