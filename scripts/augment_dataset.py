#!/usr/bin/env python3
"""
Augments a dataset of transparent cutouts organised as

    <src>/<character>/<arm_position>/angle_045.png   (also angle_045_tilt_00_nobg.png etc.)

into a new folder (the originals are never changed):

    <out>/train/<character>/<arm_position>/angle_045_a03.png
    <out>/test/ <character>/<arm_position>/angle_045_a03.png
    <out>/labels.csv      one row per image: path, split, character, arm_position, angle,
                          source, mirrored, augmentations
    <out>/preview.jpg     a grid of random samples, to check the results by eye

Every output image is a square transparent PNG (default 512 x 512) with the figure
placed on it. Augmentations (each applied with some probability):
  - scale and position (sometimes partly cut off at an edge)
  - rotation (small) and perspective
  - mirroring, WITH label correction: left/right arm positions swap and the angle
    becomes 360 - angle
  - brightness, contrast, saturation, small hue shift, coloured light tint
  - blur (focus or motion), sensor noise, JPEG compression artefacts
  - optional: random erased patches (--erase)

Train/test split: done on the original photos BEFORE augmenting (random, per
character and arm position), so variants of a test photo never end up in training.
If <src> already contains train/ and test/ folders (like the published dataset),
that split is used instead.

Needs Pillow:  pip3 install pillow

Usage:
  python3 augment_dataset.py <src> <out>
  python3 augment_dataset.py <src> <out> --per-image 10 --size 512 --test-fraction 0.2 --seed 42
  python3 augment_dataset.py <src> <out> --mirror-pairs "4_left:5_right,6_left:7_right"
"""

import argparse
import csv
import io
import math
import os
import random
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

try:
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter
except ImportError:
    sys.exit("This tool needs Pillow:  pip3 install pillow")

NAME_RE = re.compile(r"^angle_(\d+(?:\.\d+)?)(?:_tilt_(-?\d+))?(_nobg)?\.png$", re.IGNORECASE)


# ----------------------------------------------------------------------------- labels

def scan(src, prefix=""):
    """prefix: sub-folder of src to scan (e.g. "train"); it is kept in the source paths."""
    photos = []
    base = os.path.join(src, prefix) if prefix else src
    for character in sorted(os.listdir(base)):
        cdir = os.path.join(base, character)
        if not os.path.isdir(cdir) or character.startswith((".", "_")):
            continue
        for pose in sorted(os.listdir(cdir)):
            pdir = os.path.join(cdir, pose)
            if not os.path.isdir(pdir) or pose.startswith("."):
                continue
            found = {}
            for f in sorted(os.listdir(pdir)):
                m = NAME_RE.match(f)
                if m:
                    a = float(m.group(1))
                    key = (a, m.group(2) or "")
                    # If both angle_045.png and angle_045_nobg.png exist, use the _nobg cutout.
                    if key in found and not m.group(3):
                        continue
                    found[key] = {"source": os.path.join(prefix, character, pose, f), "character": character,
                                  "arm_position": pose, "angle": int(a) if a == int(a) else a,
                                  "tilt": m.group(2) or ""}
            photos += [found[k] for k in sorted(found)]
    return photos


def swap_left_right(name):
    """'4_left_arm_extended' -> '4_right_arm_extended' (keeps the case of each word)."""
    def rep(m):
        w = m.group(0)
        new = {"left": "right", "right": "left"}[w.lower()]
        return new.upper() if w.isupper() else new.capitalize() if w[0].isupper() else new
    return re.sub(r"left|right", rep, name, flags=re.IGNORECASE)


def build_mirror_map(poses, pairs_arg):
    """Arm position -> its mirror image. Poses without left/right map to themselves.
    Returns (map, warnings). A pose whose mirror can't be determined maps to None."""
    mapping, warnings = {}, []
    if pairs_arg:
        for pair in pairs_arg.split(","):
            a, b = [x.strip() for x in pair.split(":")]
            for x in (a, b):
                if x not in poses:
                    sys.exit(f"--mirror-pairs: no arm-position folder called '{x}'")
            mapping[a], mapping[b] = b, a
        for p in poses:
            mapping.setdefault(p, p)
        return mapping, warnings
    for p in poses:
        if not re.search(r"left|right", p, re.IGNORECASE):
            mapping[p] = p                                  # symmetric pose, e.g. both_arms_down
            continue
        swapped = swap_left_right(p)
        # Match ignoring numbering differences, e.g. '4_left_extended' vs '5_right_extended'.
        strip = lambda s: re.sub(r"^\d+[_\- ]*", "", s).lower()
        match = [q for q in poses if q == swapped or strip(q) == strip(swapped)]
        if match:
            mapping[p] = match[0]
        else:
            mapping[p] = None
            warnings.append(f"no mirror folder found for '{p}' (expected something like '{swapped}'): "
                            "these photos won't be mirrored")
    return mapping, warnings


def split_photos(photos, test_fraction, seed):
    """Random split of the ORIGINAL photos, separately for every character + arm position,
    so each combination (e.g. a YOLO class like batman__6_left_up) has photos in both splits.
    At least one photo per combination goes to test when test_fraction > 0."""
    rng = random.Random(seed)
    groups = {}
    for p in photos:
        groups.setdefault((p["character"], p["arm_position"]), []).append(p)
    for group in groups.values():
        order = group[:]
        rng.shuffle(order)
        n_test = round(len(order) * test_fraction)
        if test_fraction > 0 and len(order) > 1:
            n_test = min(max(1, n_test), len(order) - 1)
        for i, p in enumerate(order):
            p["split"] = "test" if i < n_test else "train"


# ----------------------------------------------------------------------------- image ops

def _perspective_coeffs(src_pts, dst_pts):
    """Coefficients for Image.transform(PERSPECTIVE) mapping dst_pts (output) -> src_pts (input)."""
    a, b = [], []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    n = 8                                                   # Gaussian elimination
    m = [row[:] + [bv] for row, bv in zip(a, b)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c and m[c][c]:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _with_rgb(img, fn):
    """Applies fn to the RGB part of an RGBA image, keeping its transparency."""
    alpha = img.split()[3]
    out = fn(img.convert("RGB")).convert("RGBA")
    out.putalpha(alpha)
    return out


def _hue_shift(rgb, degrees):
    h, s, v = rgb.convert("HSV").split()
    shift = int(round(degrees / 360 * 255))
    h = h.point(lambda x: (x + shift) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def _motion_kernel(rng):
    k = [0] * 25
    kind = rng.choice(["h", "v", "d1", "d2"])
    for i in range(5):
        idx = {"h": 2 * 5 + i, "v": i * 5 + 2, "d1": i * 5 + i, "d2": i * 5 + (4 - i)}[kind]
        k[idx] = 1
    return ImageFilter.Kernel((5, 5), k, scale=5)


def augment(cut, size, rng, opts):
    """One random augmentation of a cropped RGBA cutout. Returns (image, mirrored, list_of_ops)."""
    ops = []
    img = cut

    # Scale: the figure's longer side becomes 55-95% of the canvas (rotation may enlarge it a bit).
    target = size * rng.uniform(0.55, 0.95)
    f = target / max(img.size)
    img = img.resize((max(1, round(img.size[0] * f)), max(1, round(img.size[1] * f))), Image.LANCZOS)
    ops.append(f"scale{f * max(cut.size) / size:.2f}")

    # Colour and light (only the figure's pixels; transparency unchanged).
    if rng.random() < 0.8:
        b, c, s = rng.uniform(0.75, 1.25), rng.uniform(0.75, 1.25), rng.uniform(0.7, 1.3)
        img = _with_rgb(img, lambda x: ImageEnhance.Color(ImageEnhance.Contrast(
            ImageEnhance.Brightness(x).enhance(b)).enhance(c)).enhance(s))
        ops.append(f"color(b{b:.2f},c{c:.2f},s{s:.2f})")
    if rng.random() < 0.3:
        deg = rng.uniform(-opts.max_hue, opts.max_hue)
        img = _with_rgb(img, lambda x: _hue_shift(x, deg))
        ops.append(f"hue{deg:+.0f}")
    if rng.random() < 0.25:
        tint = tuple(rng.randint(200, 255) for _ in range(3))
        img = _with_rgb(img, lambda x: ImageChops.multiply(x, Image.new("RGB", x.size, tint)))
        ops.append("tint")

    # Mirroring.
    mirrored = False
    if opts.mirror and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirrored = True
        ops.append("mirror")

    # Rotation.
    if rng.random() < 0.7:
        deg = rng.uniform(-opts.max_rotation, opts.max_rotation)
        img = img.rotate(deg, resample=Image.BICUBIC, expand=True)
        ops.append(f"rot{deg:+.0f}")

    # Perspective: move the corners by up to 8% of the image size.
    if rng.random() < 0.4:
        w, h = img.size
        j = lambda d: rng.uniform(-0.08, 0.08) * d
        dst = [(0, 0), (w, 0), (w, h), (0, h)]
        src = [(j(w), j(h)), (w + j(w), j(h)), (w + j(w), h + j(h)), (j(w), h + j(h))]
        img = img.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(src, dst), Image.BICUBIC)
        ops.append("persp")

    # Keep it inside the canvas.
    if max(img.size) > size:
        f = size / max(img.size)
        img = img.resize((max(1, int(img.size[0] * f)), max(1, int(img.size[1] * f))), Image.LANCZOS)

    # Position: anywhere; sometimes partly outside (cut off by up to 20%).
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    w, h = img.size
    if rng.random() < opts.cutoff:
        x = rng.randint(-int(0.2 * w), size - int(0.8 * w))
        y = rng.randint(-int(0.2 * h), size - int(0.8 * h))
        ops.append("cutoff")
    else:
        x, y = rng.randint(0, size - w), rng.randint(0, size - h)
    canvas.paste(img, (x, y))      # canvas is empty, so a plain paste copies colour and transparency exactly
    img = canvas

    # Camera effects (figure only).
    if rng.random() < 0.3:
        r = rng.uniform(0.5, 1.8)
        img = _with_rgb(img, lambda x: x.filter(ImageFilter.GaussianBlur(r)))
        ops.append(f"blur{r:.1f}")
    elif rng.random() < 0.15:
        k = _motion_kernel(rng)
        img = _with_rgb(img, lambda x: x.filter(k))
        ops.append("motion")
    if rng.random() < 0.3:
        sigma = rng.uniform(4, 14)
        def noise(x):
            n = Image.effect_noise(x.size, sigma).convert("RGB")
            return ImageChops.add(x, n, scale=1.0, offset=-128)
        img = _with_rgb(img, noise)
        ops.append(f"noise{sigma:.0f}")
    if rng.random() < 0.3:
        q = rng.randint(30, 75)
        def jpeg(x):
            buf = io.BytesIO()
            x.save(buf, "JPEG", quality=q)
            return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        img = _with_rgb(img, jpeg)
        ops.append(f"jpeg{q}")

    # Optional random erasing: transparent holes in the figure.
    if opts.erase and rng.random() < opts.erase:
        bbox = img.split()[3].getbbox()
        if bbox:
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ew, eh = int(bw * rng.uniform(0.15, 0.35)), int(bh * rng.uniform(0.1, 0.25))
            ex, ey = rng.randint(bbox[0], bbox[2] - ew), rng.randint(bbox[1], bbox[3] - eh)
            alpha = img.split()[3]
            alpha.paste(0, (ex, ey, ex + ew, ey + eh))
            img.putalpha(alpha)
            ops.append("erase")

    return img, mirrored, ops


# ----------------------------------------------------------------------------- per photo

def out_name(angle, tilt, k, mirrored):
    a = f"{angle:03d}" if isinstance(angle, int) else f"{angle:05.1f}"
    t = f"_tilt_{tilt}" if tilt != "" else ""
    return f"angle_{a}{t}_a{k:02d}" + ("_m" if mirrored else "") + ".png"


def process_photo(job):
    photo, src, out, opts, mirror_map, seed = job
    rng = random.Random(seed ^ zlib.crc32(photo["source"].encode()))
    with Image.open(os.path.join(src, photo["source"])) as im:
        im = im.convert("RGBA")
    w, h = im.size
    alpha = im.split()[3]
    # Remove the faint haze some cutouts have around the figure.
    alpha = alpha.point(lambda v: 0 if v < opts.alpha_clean else v)
    # Remove everything below the cut line (e.g. the turntable base under the feet).
    if opts.cut_below:
        y_cut = int(opts.cut_below * h) if opts.cut_below <= 1 else int(opts.cut_below)
        alpha.paste(0, (0, y_cut, w, h))
    im.putalpha(alpha)
    bbox = alpha.point(lambda v: 255 if v >= 64 else 0).getbbox()
    if not bbox:
        return [], f"empty cutout, skipped: {photo['source']}"
    warning = None
    edges = [n for n, hit in (("bottom", bbox[3] >= h - 2), ("top", bbox[1] <= 1),
                              ("left", bbox[0] <= 1), ("right", bbox[2] >= w - 2)) if hit]
    if edges:
        warning = (f"cutout touches the {'/'.join(edges)} edge (turntable left in?), "
                   f"its boxes will be too big: {photo['source']}")
    cut = im.crop(alpha.getbbox())

    can_mirror = mirror_map.get(photo["arm_position"]) is not None
    rows = []
    variants = [(0, None)] if opts.include_original else []
    variants += [(k, True) for k in range(1, opts.per_image + 1)]
    for k, do_aug in variants:
        if do_aug:
            o = opts if can_mirror else argparse.Namespace(**{**vars(opts), "mirror": False})
            img, mirrored, ops = augment(cut, opts.size, rng, o)
        else:                                               # original, just centred and resized
            f = opts.size * 0.85 / max(cut.size)
            img0 = cut.resize((max(1, round(cut.size[0] * f)), max(1, round(cut.size[1] * f))), Image.LANCZOS)
            img = Image.new("RGBA", (opts.size, opts.size), (0, 0, 0, 0))
            img.alpha_composite(img0, ((opts.size - img0.size[0]) // 2, (opts.size - img0.size[1]) // 2))
            mirrored, ops = False, ["original"]
        pose = mirror_map[photo["arm_position"]] if mirrored else photo["arm_position"]
        angle = photo["angle"]
        if mirrored:
            angle = (360 - angle) % 360
            angle = int(angle) if angle == int(angle) else round(angle, 1)
        rel = os.path.join(photo["split"], photo["character"], pose, out_name(angle, photo["tilt"], k, mirrored))
        # Two different source photos can't collide: a mirrored copy has the "_m" suffix.
        os.makedirs(os.path.join(out, os.path.dirname(rel)), exist_ok=True)
        img.save(os.path.join(out, rel), optimize=False, compress_level=6)
        rows.append({"path": rel, "split": photo["split"], "character": photo["character"],
                     "arm_position": pose, "angle": angle, "tilt": photo["tilt"],
                     "source": photo["source"], "mirrored": int(mirrored), "augmentations": " ".join(ops)})
    return rows, warning


def preview(out, rows, n=48, cols=8, cell=160, seed=0):
    rng = random.Random(seed)
    pick = rng.sample(rows, min(n, len(rows)))
    rows_n = math.ceil(len(pick) / cols)
    sheet = Image.new("RGB", (cols * cell, rows_n * cell), (255, 255, 255))
    checker = Image.new("RGB", (cell, cell), (255, 255, 255))
    for yy in range(0, cell, 16):
        for xx in range(0, cell, 16):
            if (xx // 16 + yy // 16) % 2:
                checker.paste((222, 222, 222), (xx, yy, xx + 16, yy + 16))
    for i, r in enumerate(pick):
        with Image.open(os.path.join(out, r["path"])) as im:
            im = im.convert("RGBA").resize((cell - 4, cell - 4), Image.LANCZOS)
        tile = checker.copy().convert("RGBA")
        tile.alpha_composite(im, (2, 2))
        sheet.paste(tile.convert("RGB"), ((i % cols) * cell, (i // cols) * cell))
    path = os.path.join(out, "preview.jpg")
    sheet.save(path, quality=88)
    return path


def main():
    ap = argparse.ArgumentParser(description="Augment a character/arm_position/angle cutout dataset "
                                             "into transparent PNGs with a train/test split.")
    ap.add_argument("src", help="dataset folder: <src>/<character>/<arm_position>/angle_045.png")
    ap.add_argument("out", help="output folder (created; must not be inside src)")
    ap.add_argument("--per-image", type=int, default=10, help="augmented variants per photo (default 10)")
    ap.add_argument("--size", type=int, default=512, help="output image size in px, square (default 512)")
    ap.add_argument("--test-fraction", type=float, default=0.2, help="share of photos for testing (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="random seed: same seed = same results")
    ap.add_argument("--no-originals", dest="include_original", action="store_false",
                    help="don't include the un-augmented photo (centred and resized) for each source")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false", help="never mirror images")
    ap.add_argument("--mirror-pairs", metavar="A:B,C:D",
                    help="arm-position folders that are mirror images of each other, if their names "
                         "don't contain 'left'/'right'")
    ap.add_argument("--max-rotation", type=float, default=12, help="max rotation in degrees (default 12)")
    ap.add_argument("--max-hue", type=float, default=8, help="max hue shift in degrees (default 8)")
    ap.add_argument("--cutoff", type=float, default=0.15,
                    help="probability that the figure is partly cut off at an edge (default 0.15)")
    ap.add_argument("--cut-below", type=float, default=0, metavar="Y",
                    help="remove everything below this height before augmenting, e.g. the turntable "
                         "base under the feet: a fraction of the photo height (0.80) or pixels (1280)")
    ap.add_argument("--alpha-clean", type=int, default=32,
                    help="pixels more transparent than this (0-255) are removed: cleans faint haze "
                         "around cutouts (default 32)")
    ap.add_argument("--erase", type=float, default=0.0,
                    help="probability of a random transparent hole in the figure (default 0 = off)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="parallel processes")
    opts = ap.parse_args()

    src, out = os.path.abspath(opts.src), os.path.abspath(opts.out)
    if out == src or out.startswith(src + os.sep):
        sys.exit("The output folder must not be inside the source folder.")
    fixed_split = all(os.path.isdir(os.path.join(src, s)) for s in ("train", "test"))
    if fixed_split:
        photos = []
        for s in ("train", "test"):
            for p in scan(src, s):
                p["split"] = s
                photos.append(p)
    else:
        photos = scan(src)
    if not photos:
        sys.exit("No photos found. Expected <src>/<character>/<arm_position>/angle_045.png")

    poses = sorted({p["arm_position"] for p in photos})
    mirror_map, warnings = build_mirror_map(poses, opts.mirror_pairs) if opts.mirror else ({p: None for p in poses}, [])
    if fixed_split:
        print("Using the train/ and test/ folders as the split.")
    else:
        split_photos(photos, opts.test_fraction, opts.seed)

    n_chars = len({p["character"] for p in photos})
    per = opts.per_image + (1 if opts.include_original else 0)
    print(f"{len(photos)} photos ({n_chars} characters, {len(poses)} arm positions) -> "
          f"{len(photos) * per} images of {opts.size}x{opts.size} px")
    print(f"Split: {sum(p['split'] == 'train' for p in photos)} train / "
          f"{sum(p['split'] == 'test' for p in photos)} test source photos")
    if opts.mirror:
        print("Mirroring:")
        for p in poses:
            m = mirror_map[p]
            print(f"  {p:<32} -> " + ("(not mirrored)" if m is None else m if m != p else "itself (symmetric)"))
    for w in warnings:
        print("  WARNING: " + w)

    os.makedirs(out, exist_ok=True)
    jobs = [(p, src, out, opts, mirror_map, opts.seed) for p in photos]
    rows, done, warned = [], 0, []
    with ProcessPoolExecutor(max_workers=opts.workers) as ex:
        for r, err in ex.map(process_photo, jobs, chunksize=4):
            rows += r
            done += 1
            if err:
                print("  " + err)
                warned.append(err)
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} photos done")

    rows.sort(key=lambda r: r["path"])
    with open(os.path.join(out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "character", "arm_position", "angle", "tilt",
                                          "source", "mirrored", "augmentations"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} images written to {out}")
    print(f"Labels: {os.path.join(out, 'labels.csv')}")
    print(f"Preview: {preview(out, rows)}")
    edge = [w for w in warned if "touches" in w]
    if edge:
        print(f"\n{len(edge)} cutout(s) touch an edge of the photo: probably part of the turntable is "
              "still in them, so their boxes include it. Fix with --cut-below (e.g. 0.80) or redo those "
              "cutouts. They are listed above.")


if __name__ == "__main__":
    main()
