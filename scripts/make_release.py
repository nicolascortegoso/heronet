#!/usr/bin/env python3
"""
Builds a shareable, Hugging Face-ready copy of the photo dataset (full resolution):

    <out>/README.md                    dataset card (licence CC BY 4.0, description, how to use, citation)
    <out>/LICENSE
    <out>/preview.jpg                  one photo of every character x arm position
    <out>/data/train/<character>/<arm_position>/angle_045_tilt_00.png
    <out>/data/test/<character>/<arm_position>/angle_045_tilt_00.png
    <out>/data/{train,test}/metadata.jsonl   labels + bounding box per image (read by Hugging Face)
    <out>/annotations/labels.csv       every label in one table
    <out>/annotations/coco_train.json, coco_test.json   COCO-format boxes (category = character)
    <out>/scripts/                     the capture, augmentation and training scripts

Arm positions: describe each one once in a small CSV (--poses) to give it a clear name and
per-arm labels (left_arm, right_arm). Make the template with --make-poses-template, fill it in,
then pass it with --poses. Characters can be renamed with --rename-characters.

The photos are cleaned exactly like for training (faint haze removed, optional
--cut-below for the turntable base), names are standardised (lowercase, underscores)
and the train/test split is the same as augment_dataset.py makes with the same
--seed and --test-fraction, so published results match yours.

Your source folder is never changed.

Usage:
  python3 make_release.py <photo folder> --make-poses-template poses.csv
  (fill in poses.csv)
  python3 make_release.py <photo folder> <release folder> --cut-below 0.80 --poses poses.csv \
      --left-right figure --rename-characters wonderwoman=wonder_woman

Then upload (pip3 install -U huggingface_hub):
  hf auth login
  hf upload nicolashr/playmobil-superhero-poses <release folder> . --repo-type dataset
"""

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("This tool needs Pillow:  pip3 install pillow")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from augment_dataset import scan, build_mirror_map, split_photos  # noqa: E402

SCRIPTS = ["turntable.py", "esp32cam_turntable.ino", "dataset_tools.py", "rename_poses.py", "augment_dataset.py",
           "make_yolo_dataset.py", "make_pose_crops.py", "train_effects.py", "grab_frames.py", "live_detect.py",
           "evaluate_live.py", "export_models.py", "make_release.py"]


def std_name(name):
    """'Left Above' -> 'left_above'"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def file_name(angle, tilt):
    a = f"{angle:03d}" if isinstance(angle, int) else f"{angle:05.1f}"
    return f"angle_{a}" + (f"_tilt_{tilt}" if tilt != "" else "") + ".png"


def clean(path, alpha_clean, cut_below):
    """Same cleanup as augment_dataset.py; colour under fully transparent pixels set to 0."""
    with Image.open(path) as im:
        im = im.convert("RGBA")
    w, h = im.size
    alpha = im.split()[3].point(lambda v: 0 if v < alpha_clean else v)
    if cut_below:
        y_cut = int(cut_below * h) if cut_below <= 1 else int(cut_below)
        alpha.paste(0, (0, y_cut, w, h))
    clear = alpha.point(lambda v: 255 if v == 0 else 0)
    im.paste((0, 0, 0, 0), (0, 0, w, h), clear)
    im.putalpha(alpha)
    box = alpha.point(lambda v: 255 if v >= 64 else 0).getbbox()
    return im, box


def font(size):
    for name in ("Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_preview(out, rows, characters, poses, cell=150):
    """Grid: one row per character, one column per arm position, the most frontal photo."""
    label_w, head = 130, 40
    sheet = Image.new("RGB", (label_w + cell * len(poses), head + cell * len(characters)), (240, 240, 240))
    d = ImageDraw.Draw(sheet)
    small = font(13)
    for j, p in enumerate(poses):
        d.text((label_w + j * cell + 6, 12), p.replace("_", " "), fill=(40, 40, 40), font=small)
    for i, c in enumerate(characters):
        d.text((8, head + i * cell + cell // 2 - 8), c, fill=(0, 0, 0), font=small)
        for j, p in enumerate(poses):
            cand = [r for r in rows if r["character"] == c and r["arm_position"] == p]
            if not cand:
                continue
            r = min(cand, key=lambda r: (min(float(r["angle"]), 360 - float(r["angle"])), str(r["tilt"])))
            with Image.open(os.path.join(out, r["path"])) as im:
                im = im.convert("RGBA")
                im = im.crop(im.split()[3].getbbox() or (0, 0) + im.size)
                im.thumbnail((cell - 12, cell - 12))
            tile = Image.new("RGBA", (cell - 4, cell - 4), (255, 255, 255, 255))
            tile.alpha_composite(im, ((tile.size[0] - im.size[0]) // 2, (tile.size[1] - im.size[1]) // 2))
            sheet.paste(tile.convert("RGB"), (label_w + j * cell + 2, head + i * cell + 2))
    sheet.save(os.path.join(out, "preview.jpg"), quality=90)


def coco(rows, split, characters, poses):
    cid = {c: i + 1 for i, c in enumerate(characters)}
    pid = {p: i for i, p in enumerate(poses)}
    images, anns = [], []
    for i, r in enumerate([r for r in rows if r["split"] == split], 1):
        images.append({"id": i, "file_name": r["path"], "width": r["width"], "height": r["height"],
                       "character": r["character"], "arm_position": r["arm_position"],
                       "left_arm": r["left_arm"], "right_arm": r["right_arm"],
                       "angle": r["angle"], "tilt": r["tilt"]})
        x, y, w, h = r["bbox"]
        anns.append({"id": i, "image_id": i, "category_id": cid[r["character"]], "bbox": [x, y, w, h],
                     "area": w * h, "iscrowd": 0, "arm_position": r["arm_position"],
                     "arm_position_id": pid[r["arm_position"]],
                     "left_arm": r["left_arm"], "right_arm": r["right_arm"]})
    return {"info": {"description": f"Playmobil superhero figures, {split} split", "version": "1.0"},
            "licenses": [{"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
            "categories": [{"id": cid[c], "name": c, "supercategory": "figure"} for c in characters],
            "arm_positions": [{"id": pid[p], "name": p} for p in poses],
            "images": images, "annotations": anns}


def copy_scripts(out):
    dst = os.path.join(out, "scripts")
    os.makedirs(dst, exist_ok=True)
    copied = []
    for name in SCRIPTS:
        cands = [os.path.join(HERE, name),                      # scripts next to this one
                 os.path.join(HERE, "..", "firmware", os.path.splitext(name)[0], name)]   # repo layout
        src = next((c for c in cands if os.path.exists(c)), None)
        if not src:
            continue
        text = open(src, encoding="utf-8").read()
        if name.endswith(".ino"):       # never publish your Wi-Fi name and password
            text = re.sub(r'(WIFI_SSID\s*=\s*)"[^"]*"', r'\1"YOUR_SSID"', text)
            text = re.sub(r'(WIFI_PASS\s*=\s*)"[^"]*"', r'\1"YOUR_PASSWORD"', text)
        with open(os.path.join(dst, name), "w", encoding="utf-8") as f:
            f.write(text)
        copied.append(name)
    return copied


LICENSE_TEXT = """Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) {year} {author}

The images and annotations in this dataset are licensed under the Creative Commons
Attribution 4.0 International License. You are free to share and adapt them for any
purpose, even commercially, as long as you give appropriate credit, provide a link to
the license, and indicate if changes were made.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode
Summary:           https://creativecommons.org/licenses/by/4.0/

Playmobil is a trademark of geobra Brandstätter Stiftung & Co. KG. The characters are
trademarks of their respective owners ({owners}). This dataset is not affiliated with,
endorsed or sponsored by any of them. The license covers the photographs and annotations only.
"""


# ----------------------------------------------------------------------------- arm-position table

POSE_COLS = ["folder", "name", "left_arm", "right_arm", "description"]
KNOWN = {"hands_down": ("down", "down"), "both_raised": ("up", "up"), "both_extended": ("side", "side")}


def make_template(src, path):
    """Writes the arm-position table with one row per folder, to fill in by hand."""
    if os.path.exists(path):
        sys.exit(f"{path} already exists: delete it or choose another name.")
    poses = sorted({p["arm_position"] for p in scan(os.path.abspath(src))})
    if not poses:
        sys.exit("No photos found.")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POSE_COLS)
        w.writeheader()
        for p in poses:
            la, ra = KNOWN.get(std_name(p), ("", ""))
            w.writerow({"folder": p, "name": std_name(p), "left_arm": la, "right_arm": ra, "description": ""})
    print(f"Template written: {path}  ({len(poses)} arm positions)")
    print("Fill in for every row:")
    print("  name         the new label, lowercase with underscores (keep 'left'/'right' in it when it")
    print("               concerns one arm, e.g. left_arm_up, so mirror pairs are found automatically)")
    print("  left_arm     what the left arm does, one word from a small set you choose, e.g. down, side, up, forward")
    print("  right_arm    the same for the right arm")
    print("  description  a few words for the README, e.g. 'left arm straight up, right arm down'")
    print("Open it in a text editor or a spreadsheet (save as CSV).")


def read_poses(path, raw_poses):
    """Returns ({folder: new name}, {new name: {left_arm, right_arm, description}})."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in POSE_COLS if rows and c not in rows[0]]
    if not rows or missing:
        sys.exit(f"{path}: expected the columns {', '.join(POSE_COLS)}")
    names, info = {}, {}
    for r in rows:
        folder = (r["folder"] or "").strip()
        if folder not in raw_poses:
            sys.exit(f"{path}: no arm-position folder called '{folder}'")
        name = std_name(r["name"] or folder)
        la, ra = std_name(r["left_arm"] or ""), std_name(r["right_arm"] or "")
        if not la or not ra:
            sys.exit(f"{path}: fill in left_arm and right_arm for '{folder}'")
        names[folder] = name
        info[name] = {"left_arm": la, "right_arm": ra, "description": (r["description"] or "").strip()}
    absent = [p for p in raw_poses if p not in names]
    if absent:
        sys.exit(f"{path}: missing rows for {', '.join(absent)}")
    if len(set(names.values())) != len(names):
        sys.exit(f"{path}: two folders get the same name")
    combos = {}
    for n, i in info.items():
        combos.setdefault((i["left_arm"], i["right_arm"]), []).append(n)
    same = [v for v in combos.values() if len(v) > 1]
    if same:
        print("  NOTE: these arm positions have identical left_arm/right_arm labels, so the per-arm "
              "columns can't tell them apart: " + "; ".join(" = ".join(v) for v in same))
    return names, info


def check_mirror(mirror, info, poses):
    """Mirror pairs from the arm labels (left and right arm swapped), compared with the name-based ones."""
    by_arms = {}
    for p in poses:
        by_arms.setdefault((info[p]["left_arm"], info[p]["right_arm"]), []).append(p)
    result, notes = {}, []
    for p in poses:
        cands = by_arms.get((info[p]["right_arm"], info[p]["left_arm"]), [])
        if len(cands) == 1:
            result[p] = cands[0]
        elif mirror.get(p) in cands:
            result[p] = mirror[p]
        else:
            result[p] = None
            notes.append(f"'{p}' has no mirror position in this dataset (no pose with its arms swapped): "
                         "its photos won't be mirrored")
        if mirror.get(p) and result[p] and mirror[p] != result[p]:
            notes.append(f"'{p}': folder names suggested mirror '{mirror[p]}', the arm labels say '{result[p]}' "
                         "(using the arm labels)")
    return result, notes


def readme(a, stats):
    chars, poses, mirror = stats["characters"], stats["poses"], stats["mirror"]
    pinfo = stats["pinfo"]
    pose_rows = "\n".join(
        f"| `{p}` | {pinfo[p]['description'] or '—'} | {pinfo[p]['left_arm'] or '—'} | {pinfo[p]['right_arm'] or '—'} | "
        + ("symmetric" if mirror[p] == p else f"`{mirror[p]}`" if mirror[p] else "—")
        + f" | {stats['per_pose'][p]} |" for p in poses)
    if stats["lr"] == "figure":
        lr_text = "`left` and `right` always mean the **figure's own** left and right arm (not the viewer's left and right)."
    elif stats["lr"] == "camera":
        lr_text = ("`left` and `right` always mean the arm on the **left / right side of the photo** as seen by the "
                   "camera in the front view (not the figure's own left and right).")
    else:
        lr_text = ("<!-- TODO: say whether left/right are the figure's own left and right or the viewer's "
                   "(run make_release.py with --left-right figure or --left-right camera). -->")
    states = sorted({v for p in poses for v in (pinfo[p]["left_arm"], pinfo[p]["right_arm"]) if v})
    states_text = (f"Arm states used in `left_arm` / `right_arm`: {', '.join(f'`{v}`' for v in states)}.\n"
                   if states else "")
    mp = stats.get("mirror_pairs_arg")
    mirror_hint = (f" With these names, pass `--mirror-pairs \"{mp}\"` to `augment_dataset.py`." if mp else "")
    char_rows = "\n".join(f"| `{c}` | {stats['per_char'][c]} |" for c in chars)
    angles = ", ".join(f"{x}°" for x in stats["angles"])
    tilts = ", ".join(f"{t}" for t in stats["tilts"]) if stats["tilts"] != [""] else "0 (not varied)"
    slug = a.repo_id.split("/")[-1].replace("-", "_")
    return f"""---
license: cc-by-4.0
pretty_name: "{a.title.replace('"', "'")}"
task_categories:
- object-detection
- image-classification
tags:
- playmobil
- toys
- figures
- pose
- multi-view
- turntable
- transparent-background
- esp32
size_categories:
- {stats['size_category']}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/**
  - split: test
    path: data/test/**
---

# {a.title}

{stats['total']} photos of {len(chars)} Playmobil superhero figures ({a.owners} characters), each in {len(poses)} arm positions,
photographed from {len(stats['angles'])} viewing angles on an automated turntable. Every photo is a
**transparent cutout** (background removed) at full camera resolution, labelled with the
**character**, **arm position**, **viewing angle** and a **bounding box**.

![preview](preview.jpg)

## Contents

| | |
|---|---|
| Images | {stats['total']} transparent PNG cutouts ({stats['resolution']}) |
| Characters | {len(chars)} |
| Arm positions | {len(poses)} |
| Viewing angles | {angles} |
| Camera tilt | {tilts} |
| Split | {stats['n_train']} train / {stats['n_test']} test |
| Licence | CC BY 4.0 |

### Characters

| character | images |
|---|---|
{char_rows}

### Arm positions

{lr_text} Each arm position is also split into what each arm does (`left_arm`, `right_arm`), so
models can learn the arms separately and recognise combinations that were not photographed.
{states_text}
The mirror column says which arm position a horizontally mirrored photo shows, for label-correct
flip augmentation (the viewing angle of a mirrored photo becomes 360° − angle).{mirror_hint}

| arm position | description | left arm | right arm | mirror image | images |
|---|---|---|---|---|---|
{pose_rows}

### Viewing angle

The angle the turntable had turned the figure when the photo was taken.
<!-- TODO: check and describe your convention, e.g. "0° = figure facing the camera, angles increase
clockwise seen from above". -->

## Structure

```
data/
  train/<character>/<arm_position>/angle_045_tilt_00.png
  train/metadata.jsonl
  test/<character>/<arm_position>/angle_045_tilt_00.png
  test/metadata.jsonl
annotations/
  labels.csv           all labels in one table
  coco_train.json      COCO format, category = character, arm position as an extra field
  coco_test.json
scripts/               capture, augmentation and training code
```

Each line of `metadata.jsonl`:

```json
{{"file_name": "{stats['example_file']}", "character": "{stats['example_char']}",
 "arm_position": "{stats['example_pose']}", "left_arm": "...", "right_arm": "...", "angle": 45, "tilt": 0, "width": 1200, "height": 1600, "mirror_arm_position": "...",
 "objects": {{"bbox": [[x, y, w, h]], "categories": [1]}}}}
```

Bounding boxes are in pixels, `[x, y, width, height]` from the top-left corner, computed from the
cutout's opaque pixels (alpha ≥ 64). `categories` holds the character index in the order of the
character table above (starting at 0).

The split is random **per character and arm position** (20% of the views of each combination are in
`test`), so every combination appears in both splits but no view is shared between them.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{a.repo_id}")
example = ds["train"][0]
example["image"]          # PIL image, RGBA (transparent background)
example["character"], example["arm_position"], example["angle"]
```

The images keep their transparency: composite them onto any background you like, e.g.

```python
from PIL import Image
img = example["image"]
white = Image.new("RGBA", img.size, (255, 255, 255, 255))
white.alpha_composite(img)
```

The `scripts/` folder contains the full pipeline used to train real-time models on this data:

1. `augment_dataset.py data aug` — augmentation with label-correct mirroring
   (uses the train/test split of this dataset).
2. `make_yolo_dataset.py aug yolo_char --classes character --random-bg --low-res` — YOLO detection dataset.
3. `make_pose_crops.py aug pose_crops --random-bg --low-res` — cropped arm-position classification dataset.
4. `live_detect.py` — runs both models on an ESP32-CAM video stream.

## How it was made

- **Camera:** ESP32-CAM (AI-Thinker, OV2640 sensor), JPEG photos sent over Wi-Fi to a computer.
- **Turntable:** an SG90 servo turning the figure, controlled by the same ESP32-CAM (`scripts/esp32cam_turntable.ino`,
  `scripts/turntable.py`).
- **Background removal:** [rembg](https://github.com/danielgatis/rembg) with the u2net model; only the cutouts
  were kept. Faint semi-transparent haze (alpha < {a.alpha_clean}) was removed{stats['cut_note']}.

## Limitations

- {len(stats['angles'])} viewing angles per arm position, from one camera height and one lighting setup.
- Cutouts were made automatically and may contain small errors at the edges.
- Original photos with their backgrounds are not included.
- Some arm positions look alike from behind (e.g. one figure's `left` seen from the back resembles `right`
  seen from the front).

## Licence and trademarks

The photographs and annotations are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
you may use them for any purpose, including commercially, with attribution.

Playmobil is a trademark of geobra Brandstätter Stiftung & Co. KG; the characters are trademarks of their
respective owners ({a.owners}). This dataset shows commercially available toy figures and is not affiliated
with, endorsed or sponsored by any of them.

## Citation

```bibtex
@misc{{{slug}_{a.year},
  author       = {{{a.author}}},
  title        = {{{a.title}}},
  year         = {{{a.year}}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{a.repo_id}}}}}
}}
```
"""


def main():
    ap = argparse.ArgumentParser(description="Make a Hugging Face-ready release of the photo dataset.")
    ap.add_argument("src", help="photo folder: <src>/<character>/<arm_position>/angle_045_tilt_00_nobg.png")
    ap.add_argument("out", nargs="?", help="release folder to create (must not exist or be empty)")
    ap.add_argument("--repo-id", default="nicolashr/playmobil-superhero-poses",
                    help="Hugging Face dataset id, user/name (used in the README)")
    ap.add_argument("--title", default="Playmobil Superhero Figures: Arm Positions and Viewing Angles")
    ap.add_argument("--owners", default="DC Comics and Marvel",
                    help="who owns the characters, for the trademark notice (default: DC Comics and Marvel)")
    ap.add_argument("--author", default="N. Cortegoso Vissio")
    ap.add_argument("--year", type=int, default=datetime.date.today().year)
    ap.add_argument("--cut-below", type=float, default=0, metavar="Y",
                    help="remove everything below this height (fraction or pixels), as in augment_dataset.py")
    ap.add_argument("--alpha-clean", type=int, default=32, help="remove alpha below this (default 32)")
    ap.add_argument("--test-fraction", type=float, default=0.2, help="as in augment_dataset.py (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="as in augment_dataset.py (default 42)")
    ap.add_argument("--poses", metavar="CSV",
                    help="arm-position table: folder,name,left_arm,right_arm,description (see --make-poses-template)")
    ap.add_argument("--make-poses-template", metavar="CSV",
                    help="write a template of that table for your photo folder, then stop")
    ap.add_argument("--left-right", choices=["figure", "camera"],
                    help="are left/right the figure's own (figure) or as seen in the photo (camera)?")
    ap.add_argument("--rename-characters", metavar="OLD=NEW,...", default="",
                    help="e.g. wonderwoman=wonder_woman")
    a = ap.parse_args()
    if a.make_poses_template:
        make_template(a.src, a.make_poses_template)
        return

    if not a.out:
        sys.exit("Give the release folder to create (or --make-poses-template poses.csv).")
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    if out == src or out.startswith(src + os.sep):
        sys.exit("The release folder must not be inside the photo folder.")
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"{out} already exists and is not empty: delete it or choose another name.")
    photos = scan(src)
    if not photos:
        sys.exit("No photos found. Expected <src>/<character>/<arm_position>/angle_045.png")

    raw_poses = sorted({p["arm_position"] for p in photos})
    mirror_raw, warnings = build_mirror_map(raw_poses, None)
    split_photos(photos, a.test_fraction, a.seed)            # identical to augment_dataset.py

    renames = dict(x.split("=", 1) for x in a.rename_characters.split(",") if x.strip())
    std_c = {c: std_name(renames.get(c, renames.get(std_name(c), c))) for c in {p["character"] for p in photos}}
    for old in renames:
        if old not in std_c and old not in {std_name(c) for c in std_c}:
            sys.exit(f"--rename-characters: no character folder called '{old}'")
    if a.poses:
        std_p, pinfo = read_poses(a.poses, raw_poses)
    else:
        std_p = {p: std_name(p) for p in raw_poses}
        pinfo = {std_p[p]: {"left_arm": "", "right_arm": "", "description": ""} for p in raw_poses}
    for kind, m in (("character", std_c), ("arm position", std_p)):
        if len(set(m.values())) != len(m):
            sys.exit(f"Two {kind} folders get the same standardised name: {m}")
    characters, poses = sorted(std_c.values()), sorted(std_p.values())
    mirror = {std_p[p]: (std_p[m] if m else None) for p, m in mirror_raw.items()}
    mirror_pairs_arg = None
    if a.poses:
        mirror, mirror_problems = check_mirror(mirror, pinfo, poses)
        for m in mirror_problems:
            print("  NOTE: " + m)
        auto, _ = build_mirror_map(poses, None)        # what augment_dataset.py will guess from the new names
        if any(auto[p] != mirror[p] for p in poses):
            pairs = sorted({tuple(sorted((p, q))) for p, q in mirror.items() if q and q != p})
            mirror_pairs_arg = ",".join(f"{p}:{q}" for p, q in pairs)
            print("  NOTE: the new names don't say left/right clearly enough for augment_dataset.py to find the "
                  f"mirror pairs by itself; the README tells users to pass --mirror-pairs \"{mirror_pairs_arg}\"")
    char_idx = {c: i for i, c in enumerate(characters)}

    print(f"{len(photos)} photos: {len(characters)} characters, {len(poses)} arm positions")
    for w in warnings:
        print("  WARNING: " + w)
    renamed = [f"{k} -> {v}" for k, v in list(std_c.items()) + list(std_p.items()) if k != v]
    if renamed:
        print("Renamed folders: " + ", ".join(renamed))

    rows, problems = [], []
    for i, p in enumerate(photos, 1):
        c, pose = std_c[p["character"]], std_p[p["arm_position"]]
        rel = os.path.join("data", p["split"], c, pose, file_name(p["angle"], p["tilt"]))
        im, box = clean(os.path.join(src, p["source"]), a.alpha_clean, a.cut_below)
        if not box:
            problems.append(f"empty after cleaning, left out: {p['source']}")
            continue
        w, h = im.size
        if box[3] >= h - 2 or box[1] <= 1 or box[0] <= 1 or box[2] >= w - 2:
            problems.append(f"figure touches the photo edge (turntable left in? try --cut-below): {p['source']}")
        os.makedirs(os.path.join(out, os.path.dirname(rel)), exist_ok=True)
        im.save(os.path.join(out, rel), optimize=True)
        tilt = int(p["tilt"]) if p["tilt"] != "" else 0
        rows.append({"path": rel, "split": p["split"], "character": c, "arm_position": pose,
                     "left_arm": pinfo[pose]["left_arm"], "right_arm": pinfo[pose]["right_arm"],
                     "angle": p["angle"], "tilt": tilt, "width": w, "height": h,
                     "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                     "mirror_arm_position": mirror[pose], "source_file": p["source"].replace(os.sep, "/")})
        if i % 50 == 0 or i == len(photos):
            print(f"  {i}/{len(photos)} photos")

    for split in ("train", "test"):
        with open(os.path.join(out, "data", split, "metadata.jsonl"), "w") as f:
            for r in rows:
                if r["split"] != split:
                    continue
                f.write(json.dumps({
                    "file_name": r["path"].split(os.sep, 2)[2].replace(os.sep, "/"),
                    "character": r["character"], "arm_position": r["arm_position"],
                    "left_arm": r["left_arm"], "right_arm": r["right_arm"],
                    "angle": r["angle"], "tilt": r["tilt"], "width": r["width"], "height": r["height"],
                    "mirror_arm_position": r["mirror_arm_position"],
                    "objects": {"bbox": [r["bbox"]], "categories": [char_idx[r["character"]]]},
                }) + "\n")

    ann = os.path.join(out, "annotations")
    os.makedirs(ann, exist_ok=True)
    with open(os.path.join(ann, "labels.csv"), "w", newline="") as f:
        cols = ["path", "split", "character", "arm_position", "left_arm", "right_arm", "angle", "tilt", "width", "height",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h", "mirror_arm_position"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            x, y, bw, bh = r["bbox"]
            wr.writerow({**{k: r[k] for k in cols if k in r}, "path": r["path"].replace(os.sep, "/"),
                         "bbox_x": x, "bbox_y": y, "bbox_w": bw, "bbox_h": bh,
                         "mirror_arm_position": r["mirror_arm_position"] or ""})
    for split in ("train", "test"):
        data = coco(rows, split, characters, poses)
        for im in data["images"]:
            im["file_name"] = im["file_name"].replace(os.sep, "/")
        with open(os.path.join(ann, f"coco_{split}.json"), "w") as f:
            json.dump(data, f)

    make_preview(out, rows, characters, poses)
    copied = copy_scripts(out)

    sizes = sorted({(r["width"], r["height"]) for r in rows})
    n = len(rows)
    stats = {
        "characters": characters, "poses": poses, "mirror": mirror, "total": n,
        "pinfo": pinfo, "lr": a.left_right, "mirror_pairs_arg": mirror_pairs_arg,
        "example_file": rows[0]["path"].split(os.sep, 2)[2].replace(os.sep, "/"),
        "example_char": rows[0]["character"], "example_pose": rows[0]["arm_position"],
        "per_char": {c: sum(r["character"] == c for r in rows) for c in characters},
        "per_pose": {p: sum(r["arm_position"] == p for r in rows) for p in poses},
        "angles": sorted({r["angle"] for r in rows}),
        "tilts": sorted({str(r["tilt"]) for r in rows}) if any(p["tilt"] != "" for p in photos) else [""],
        "resolution": " / ".join(f"{w}×{h} px" for w, h in sizes),
        "n_train": sum(r["split"] == "train" for r in rows), "n_test": sum(r["split"] == "test" for r in rows),
        "size_category": "n<1K" if n < 1000 else "1K<n<10K" if n < 10000 else "10K<n<100K",
        "cut_note": (f" and everything below {a.cut_below:.0%} of the photo height (the turntable base) was cut off"
                     if 0 < a.cut_below <= 1 else
                     f" and everything below {a.cut_below:.0f} px (the turntable base) was cut off" if a.cut_below else ""),
    }
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme(a, stats))
    with open(os.path.join(out, "LICENSE"), "w", encoding="utf-8") as f:
        f.write(LICENSE_TEXT.format(year=a.year, author=a.author, owners=a.owners))

    total_mb = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out) for fn in fns) / 1e6
    print(f"\nRelease: {out}  ({n} images, {total_mb:.0f} MB)")
    print(f"  {stats['n_train']} train / {stats['n_test']} test, resolution {stats['resolution']}")
    print(f"  scripts included: {', '.join(copied) or 'none found next to make_release.py'}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  - " + p)
    print("\nBefore uploading:")
    todo = ["viewing-angle convention"] + ([] if a.left_right else ["left/right convention (or use --left-right)"])
    print(f"  1. Open {os.path.join(out, 'README.md')} and fill in the TODOs: {'; '.join(todo)}.")
    if not a.poses:
        print("     Tip: describe the arm positions with --make-poses-template / --poses for clearer labels.")
    print(f"  2. Look at {os.path.join(out, 'preview.jpg')}.")
    if a.repo_id.startswith("YOUR_HF_USERNAME"):
        print("  3. Run again with --repo-id <your-hf-username>/playmobil-superhero-poses (or edit the README).")
    print("\nUpload:  pip3 install -U huggingface_hub")
    print("         hf auth login")
    print(f"         hf upload {a.repo_id} {out} . --repo-type dataset")


if __name__ == "__main__":
    main()
