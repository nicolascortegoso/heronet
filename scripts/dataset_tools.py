#!/usr/bin/env python3
"""
Checks and summarises a photo dataset organised as

    <root>/<character>/<arm_position>/angle_045.png  (or angle_045_tilt_10.png, angle_045_tilt_00_nobg.png)

What it does:
  1. Check   - counts photos per character and arm position, lists missing angles,
               unexpected files and images that can't be opened, and flags cutouts
               that look suspicious (the figure covers far more or less of the photo
               than is usual for that character).
  2. Labels  - writes <root>/labels.csv with one row per photo:
               path, character, arm_position, angle, tilt, width, height, coverage
  3. Sheets  - writes one overview image per character to <root>/_contact_sheets/:
               a grid with one row per arm position and one column per angle.

Needs Pillow:  pip3 install pillow

Usage:
  python3 dataset_tools.py <root>                 # check + labels.csv + contact sheets
  python3 dataset_tools.py <root> --no-sheets     # skip the contact sheets
  python3 dataset_tools.py <root> --thumb 160     # bigger thumbnails (default 120 px)
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("This tool needs Pillow:  pip3 install pillow")

NAME_RE = re.compile(r"^angle_(\d+(?:\.\d+)?)(?:_tilt_(-?\d+))?(_nobg)?\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)
SKIP_DIRS = {"_contact_sheets"}


def angle_key(a):
    return float(a)


def scan(root):
    """Returns (photos, problems). photos: list of dicts; problems: list of strings."""
    photos, problems = [], []
    for character in sorted(os.listdir(root)):
        cdir = os.path.join(root, character)
        if not os.path.isdir(cdir) or character.startswith(".") or character in SKIP_DIRS:
            continue
        for pose in sorted(os.listdir(cdir)):
            pdir = os.path.join(cdir, pose)
            if pose.startswith("."):
                continue
            if not os.path.isdir(pdir):
                problems.append(f"unexpected file (not in an arm-position folder): {character}/{pose}")
                continue
            found = {}
            for f in sorted(os.listdir(pdir)):
                if f.startswith("."):
                    continue
                m = NAME_RE.match(f)
                if not m:
                    problems.append(f"unexpected file name: {character}/{pose}/{f}")
                    continue
                angle = m.group(1)
                angle = str(int(float(angle))) if float(angle) == int(float(angle)) else angle
                tilt = m.group(2) if m.group(2) is not None else ""
                # If both the original and its _nobg cutout exist, count the photo once (the cutout).
                if (angle, tilt) in found and not m.group(3):
                    continue
                found[(angle, tilt)] = {
                    "path": os.path.join(character, pose, f),
                    "character": character,
                    "arm_position": pose,
                    "angle": angle,
                    "tilt": tilt,
                }
            photos += [found[k] for k in sorted(found, key=lambda k: (float(k[0]), k[1]))]
    return photos, problems


def measure(root, photo):
    """Adds width, height and coverage (fraction of non-transparent pixels)."""
    with Image.open(os.path.join(root, photo["path"])) as im:
        im.load()
        photo["width"], photo["height"] = im.size
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            alpha = im.convert("RGBA").split()[3]
            hist = alpha.histogram()
            opaque = sum(hist[128:])
            photo["coverage"] = round(opaque / (im.size[0] * im.size[1]), 4)
            box = alpha.point(lambda v: 255 if v >= 64 else 0).getbbox()
            w, h = im.size
            if box:
                photo["edges"] = [n for n, hit in (("bottom", box[3] >= h - 2), ("top", box[1] <= 1),
                                                   ("left", box[0] <= 1), ("right", box[2] >= w - 2)) if hit]
        else:
            photo["coverage"] = ""          # no transparency: not a cutout


def check(photos, problems):
    """Prints a report; returns the number of issues found."""
    by_char = defaultdict(lambda: defaultdict(list))
    for p in photos:
        by_char[p["character"]][p["arm_position"]].append(p)

    all_poses = sorted({p["arm_position"] for p in photos})
    all_views = sorted({(p["angle"], p["tilt"]) for p in photos}, key=lambda v: (angle_key(v[0]), v[1]))
    issues = list(problems)

    print(f"\nFound {len(photos)} photos: {len(by_char)} characters, "
          f"{len(all_poses)} arm positions, {len(all_views)} views (angle/tilt) per pose.")
    expected = len(by_char) * len(all_poses) * len(all_views)
    print(f"Expected for a complete set: {len(by_char)} x {len(all_poses)} x {len(all_views)} = {expected}\n")

    print(f"{'character':<24}{'poses':>6}{'photos':>8}")
    for character, poses in by_char.items():
        n = sum(len(v) for v in poses.values())
        print(f"{character:<24}{len(poses):>6}{n:>8}")
        for pose in all_poses:
            if pose not in poses:
                issues.append(f"missing arm position: {character}/{pose}")
                continue
            have = {(p["angle"], p["tilt"]) for p in poses[pose]}
            for view in all_views:
                if view not in have:
                    a, t = view
                    issues.append(f"missing photo: {character}/{pose}/angle {a}" + (f" tilt {t}" if t else ""))

    # Suspicious cutouts: coverage far from the character's typical value.
    for character, poses in by_char.items():
        covs = sorted(p["coverage"] for v in poses.values() for p in v if p.get("coverage") not in ("", None))
        if len(covs) < 5:
            continue
        median = covs[len(covs) // 2]
        for v in poses.values():
            for p in v:
                c = p.get("coverage")
                if c in ("", None):
                    continue
                if c < 0.01:
                    issues.append(f"almost empty cutout ({c:.1%} of the photo): {p['path']}")
                elif c > 0.9:
                    issues.append(f"background probably not removed ({c:.0%} opaque): {p['path']}")
                elif median > 0 and (c > 2.0 * median or c < 0.5 * median):
                    issues.append(f"unusual cutout size ({c:.1%} vs typical {median:.1%}): {p['path']}")

    for p in photos:
        if p.get("edges"):
            issues.append(f"cutout touches the {'/'.join(p['edges'])} edge (turntable left in?): {p['path']}")

    print()
    if issues:
        print(f"{len(issues)} issue(s):")
        for i in issues:
            print("  - " + i)
    else:
        print("No issues found: the dataset is complete.")
    return len(issues)


def write_labels(root, photos):
    path = os.path.join(root, "labels.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "character", "arm_position", "angle", "tilt",
                                          "width", "height", "coverage"])
        w.writeheader()
        for p in photos:
            w.writerow({k: p.get(k, "") for k in w.fieldnames})
    print(f"\nLabels written to {path}")


def _font(size):
    for name in ("Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def contact_sheets(root, photos, thumb):
    outdir = os.path.join(root, "_contact_sheets")
    os.makedirs(outdir, exist_ok=True)
    all_poses = sorted({p["arm_position"] for p in photos})
    all_views = sorted({(p["angle"], p["tilt"]) for p in photos}, key=lambda v: (angle_key(v[0]), v[1]))
    font, small = _font(max(12, thumb // 7)), _font(max(10, thumb // 9))
    label_w, head_h, pad = max(160, thumb + 40), max(28, thumb // 4), 6

    by_key = {(p["character"], p["arm_position"], p["angle"], p["tilt"]): p for p in photos}
    for character in sorted({p["character"] for p in photos}):
        cell_w, cell_h = thumb + pad, int(thumb * 4 / 3) + pad      # photos are portrait (3:4)
        sheet = Image.new("RGB", (label_w + cell_w * len(all_views), head_h * 2 + cell_h * len(all_poses)),
                          (235, 235, 235))
        d = ImageDraw.Draw(sheet)
        d.text((8, 6), character, fill=(0, 0, 0), font=font)
        for j, (a, t) in enumerate(all_views):
            d.text((label_w + j * cell_w + 4, head_h + 4), f"{a}°" + (f" t{t}" if t else ""),
                   fill=(60, 60, 60), font=small)
        for i, pose in enumerate(all_poses):
            y = head_h * 2 + i * cell_h
            d.text((8, y + cell_h // 2 - 8), pose, fill=(0, 0, 0), font=small)
            for j, (a, t) in enumerate(all_views):
                x = label_w + j * cell_w
                p = by_key.get((character, pose, a, t))
                if not p:
                    d.rectangle([x + 2, y + 2, x + cell_w - 4, y + cell_h - 4], outline=(220, 0, 0), width=2)
                    d.text((x + 8, y + cell_h // 2 - 8), "missing", fill=(220, 0, 0), font=small)
                    continue
                try:
                    with Image.open(os.path.join(root, p["path"])) as im:
                        im = im.convert("RGBA")
                        im.thumbnail((cell_w - pad, cell_h - pad))
                        tile = Image.new("RGBA", im.size, (255, 255, 255, 255))
                        tile.alpha_composite(im)
                        sheet.paste(tile.convert("RGB"), (x + (cell_w - im.size[0]) // 2,
                                                          y + (cell_h - im.size[1]) // 2))
                except OSError:
                    d.text((x + 8, y + cell_h // 2 - 8), "unreadable", fill=(220, 0, 0), font=small)
        out = os.path.join(outdir, f"{character}.jpg")
        sheet.save(out, quality=90)
        print(f"  contact sheet: {out}")


def main():
    ap = argparse.ArgumentParser(description="Check a character/arm_position/angle photo dataset, "
                                             "write labels.csv and contact sheets.")
    ap.add_argument("root", help="dataset folder (contains one folder per character)")
    ap.add_argument("--no-sheets", action="store_true", help="don't make contact sheets")
    ap.add_argument("--thumb", type=int, default=120, help="thumbnail width in the contact sheets (px)")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Not a folder: {args.root}")
    photos, problems = scan(args.root)
    if not photos:
        sys.exit("No photos found. Expected <root>/<character>/<arm_position>/angle_045.png")

    readable = []
    for p in photos:
        try:
            measure(args.root, p)
            readable.append(p)
        except OSError as e:
            problems.append(f"can't open image: {p['path']} ({e})")

    check(readable, problems)
    write_labels(args.root, readable)
    if not args.no_sheets:
        print("\nMaking contact sheets...")
        contact_sheets(args.root, readable, args.thumb)


if __name__ == "__main__":
    main()
