#!/usr/bin/env python3
"""
Measures how well the models work on REAL camera frames (saved with grab_frames.py):

    <folder>/<character>/<arm_position>/*.jpg

Reports character accuracy, arm-position accuracy, both together, frames where no
figure was found, the most common mistakes and the speed. Every wrong frame is saved,
annotated, to <out>/ so you can see what went wrong.

Several figures per frame (--multi): frames in folders named after the figures they show,
    <folder>/batman+joker/*.jpg     <folder>/batman+batman+robin/*.jpg
Reports how many frames got exactly the right characters, and figure-level recall/precision.

Usage:
  python3 evaluate_live.py live_test --det <detector> --pose <classifier>
  python3 evaluate_live.py live_test_multi --multi --det <detector> --pose <classifier>
  python3 evaluate_live.py live_test --det det_ncnn_model --pose pose_ncnn_model   (on the Pi)
"""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_detect import Recognizer, draw  # noqa: E402
import cv2  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".bmp")


def evaluate_multi(a, rec):
    items = []
    for d in sorted(os.listdir(a.folder)):
        ddir = os.path.join(a.folder, d)
        if os.path.isdir(ddir) and not d.startswith((".", "_")):
            want = Counter(x for x in d.split("+") if x)
            items += [(os.path.join(ddir, f), want) for f in sorted(os.listdir(ddir)) if f.lower().endswith(EXT)]
    if not items:
        sys.exit(f"No frames found. Expected {a.folder}/<character>+<character>/*.jpg")
    names_c = set(rec.det.names.values())
    unknown = sorted({c for _, w in items for c in w if c not in names_c})
    if unknown:
        print("WARNING: not classes of the detector (always counted missing): " + ", ".join(unknown))
    rec.warmup()
    os.makedirs(a.out, exist_ok=True)
    exact = expected = predicted = matched = too_few = too_many = 0
    per_char = defaultdict(lambda: [0, 0])                # found, expected
    confusions = Counter()
    det_ms, pose_ms = [], []
    for path, want in items:
        img = cv2.imread(path)
        if img is None:
            print(f"  can't read {path}")
            continue
        found = rec(img)
        det_ms.append(rec.det_ms)
        if found and rec.pose is not None:
            pose_ms.append(rec.pose_ms / len(found))
        got = Counter(f[1] for f in found)
        hit = sum((got & want).values())
        matched += hit
        expected += sum(want.values())
        predicted += sum(got.values())
        exact += got == want
        too_few += sum(got.values()) < sum(want.values())
        too_many += sum(got.values()) > sum(want.values())
        for c, k in want.items():
            per_char[c][0] += min(k, got[c])
            per_char[c][1] += k
        missing, extra = want - got, got - want
        for m in missing.elements():
            confusions[(m, next(iter(extra.elements()), "(nothing)"))] += 1
        if got != want:
            view = draw(img, found, "truth: " + " + ".join(sorted(want.elements())), 2.0)
            cv2.imwrite(os.path.join(a.out, "MULTI__" + "+".join(sorted(want.elements())) + "__"
                                     + os.path.basename(path)), view)
    n = len(det_ms)
    print(f"\n{n} real frames with several figures ({expected} figures)")
    print(f"  exactly right characters: {exact}/{n}  ({exact / max(1, n):.1%})")
    print(f"  figures found (recall):   {matched}/{expected}  ({matched / max(1, expected):.1%})")
    print(f"  boxes correct (precision):{matched:>4}/{predicted}  ({matched / max(1, predicted):.1%})")
    print(f"  frames with too few boxes: {too_few}, too many: {too_many}")
    print(f"\n{'character':<16}{'found':>10}")
    for c in sorted(per_char):
        ok, tot = per_char[c]
        print(f"{c:<16}{ok:>5}/{tot:<4}")
    if confusions:
        print("\nMost common mistakes (expected -> got instead):")
        for (t, g), k in confusions.most_common(8):
            print(f"  {t} -> {g}: {k}")
    avg = lambda v: sum(v) / max(1, len(v))
    print(f"\nSpeed: detector {avg(det_ms):.0f} ms per frame"
          + (f", classifier {avg(pose_ms):.0f} ms per figure" if pose_ms else ""))
    if exact < n:
        print(f"Mistakes saved in {os.path.abspath(a.out)}/")


def main():
    ap = argparse.ArgumentParser(description="Accuracy of the live models on real labelled camera frames.")
    ap.add_argument("folder", help="<folder>/<character>/<arm_position>/*.jpg")
    ap.add_argument("--det", required=True, help="character detector: best.pt or an NCNN folder")
    ap.add_argument("--pose", help="arm-position classifier: best.pt or an NCNN folder")
    ap.add_argument("--det-imgsz", type=int, default=320)
    ap.add_argument("--pose-imgsz", type=int, default=224)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="live_eval_errors", help="folder for the annotated mistakes")
    ap.add_argument("--multi", action="store_true",
                    help="frames with several figures, in folders like batman+joker/")
    ap.add_argument("--max-figures", type=int, default=6, help="with --multi: most boxes per frame (default 6)")
    a = ap.parse_args()
    if a.multi:
        rec = Recognizer(a.det, a.pose, a.det_imgsz, a.pose_imgsz, a.conf, a.max_figures, a.margin, a.device)
        evaluate_multi(a, rec)
        return

    items = []
    for c in sorted(os.listdir(a.folder)):
        cdir = os.path.join(a.folder, c)
        if not os.path.isdir(cdir) or c.startswith((".", "_")):
            continue
        for p in sorted(os.listdir(cdir)):
            pdir = os.path.join(cdir, p)
            if os.path.isdir(pdir):
                items += [(os.path.join(pdir, f), c, p) for f in sorted(os.listdir(pdir)) if f.lower().endswith(EXT)]
    if not items:
        sys.exit(f"No frames found. Expected {a.folder}/<character>/<arm_position>/*.jpg")

    rec = Recognizer(a.det, a.pose, a.det_imgsz, a.pose_imgsz, a.conf, 1, a.margin, a.device)
    names_c = set(rec.det.names.values())
    names_p = set(rec.pose.names.values()) if rec.pose is not None else set()
    unknown = sorted({c for _, c, _ in items if c not in names_c} |
                     ({p for _, _, p in items if p not in names_p} if rec.pose is not None else set()))
    if unknown:
        print("WARNING: these folder names are not classes of the models (always counted wrong): "
              + ", ".join(unknown))
    rec.warmup()
    os.makedirs(a.out, exist_ok=True)

    char_ok = pose_ok = both_ok = missed = 0
    char_err, pose_err = Counter(), Counter()
    per_char, per_pose = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    det_ms, pose_ms = [], []
    for path, c, p in items:
        img = cv2.imread(path)
        if img is None:
            print(f"  can't read {path}")
            continue
        found = rec(img)
        det_ms.append(rec.det_ms)
        if not found:
            missed += 1
            per_char[c][1] += 1
            per_pose[p][1] += 1
            cv2.imwrite(os.path.join(a.out, f"MISSED__{c}__{p}__{os.path.basename(path)}"), img)
            continue
        if rec.pose is not None:
            pose_ms.append(rec.pose_ms)
        _, pc, _, pp, _ = found[0]
        okc, okp = pc == c, (pp == p) if rec.pose is not None else True
        char_ok += okc
        pose_ok += okp
        both_ok += okc and okp
        per_char[c][0] += okc
        per_char[c][1] += 1
        per_pose[p][0] += okp
        per_pose[p][1] += 1
        if not okc:
            char_err[(c, pc)] += 1
        if not okp:
            pose_err[(p, pp)] += 1
        if not (okc and okp):
            view = draw(img, found, f"truth: {c} | {p}", 2.0)
            cv2.imwrite(os.path.join(a.out, f"{c}__{p}__{os.path.basename(path)}"), view)

    n = len(det_ms)
    pct = lambda k: f"{k / n:.1%}" if n else "-"
    print(f"\n{n} real frames")
    print(f"  figure found:       {n - missed}/{n}  ({pct(n - missed)})")
    print(f"  character correct:  {char_ok}/{n}  ({pct(char_ok)})")
    if rec.pose is not None:
        print(f"  arm pos. correct:   {pose_ok}/{n}  ({pct(pose_ok)})")
        print(f"  both correct:       {both_ok}/{n}  ({pct(both_ok)})")

    print(f"\n{'character':<16}{'correct':>10}")
    for c in sorted(per_char):
        ok, tot = per_char[c]
        print(f"{c:<16}{ok:>5}/{tot:<4}")
    if rec.pose is not None:
        print(f"\n{'arm position':<16}{'correct':>10}")
        for p in sorted(per_pose):
            ok, tot = per_pose[p]
            print(f"{p:<16}{ok:>5}/{tot:<4}")

    for title, err in (("character", char_err), ("arm position", pose_err)):
        if err:
            print(f"\nMost common {title} mistakes (truth -> predicted):")
            for (t, pr), k in err.most_common(8):
                print(f"  {t} -> {pr}: {k}")

    avg = lambda v: sum(v[2:]) / max(1, len(v) - 2) if len(v) > 2 else (sum(v) / max(1, len(v)))
    print(f"\nSpeed: detector {avg(det_ms):.0f} ms" + (f", classifier {avg(pose_ms):.0f} ms" if pose_ms else "")
          + " per frame")
    if missed or char_err or pose_err:
        print(f"Mistakes saved in {os.path.abspath(a.out)}/")


if __name__ == "__main__":
    t = time.time()
    main()
