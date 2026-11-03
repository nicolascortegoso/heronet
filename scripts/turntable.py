#!/usr/bin/env python3
"""
Turntable photo capture for the ESP32-CAM (or ESP32 + OV7670) + two standard
(180°) servos: one turns the turntable, one tilts the camera.

For every tilt (default 0°, 15°, 30°) it photographs every turntable angle
(default every 45°) and saves the photos on THIS computer.
Uses only the Python standard library. JPEG photos (ESP32-CAM) are saved as they
arrive; BMP photos (OV7670) are saved as PNG if Pillow is installed.
--shots > 1 and --despeckle need Pillow for JPEG photos:  pip3 install pillow

Two setups:
  * 2:1 gears (servo turns 180°, turntable turns 360°): fully automatic.
      python3 turntable.py 192.168.0.92 --gear-ratio 2
  * Servo turns the turntable directly (no gears): the script takes 0-135°,
    then asks you once to turn the figure 180° by hand, and takes 180-315°.
      python3 turntable.py 192.168.0.92

Tilt: set --tilt-level to the tilt-servo angle where the figure stands straight.
"""

import argparse
import datetime
import http.client
import io
import os
import sys
import time
import urllib.error
import urllib.request


def get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def move_servo(base, deg, servo="turn", retries=2):
    # The ESP32 answers only after the servo has reached the angle.
    for attempt in range(retries + 1):
        try:
            get(f"{base}/angle?deg={deg:.1f}&servo={servo}", timeout=30)
            return
        except NET_ERRORS as e:
            if attempt == retries:
                raise RuntimeError(f"could not move the {servo} servo: {e}")
            print(f"    retrying servo move ({e})")
            time.sleep(3)


def photo_name(angle, tilt):
    """angle_045_tilt_10, or angle_022.5_tilt_10 for angles with decimals."""
    a = f"{angle:03d}" if isinstance(angle, int) else f"{angle:05.1f}"
    return f"angle_{a}_tilt_{tilt:02d}"


def existing(out, name):
    """Path of an already-saved photo with this name, or None."""
    for ext in (".jpg", ".png", ".bmp"):
        if os.path.exists(os.path.join(out, name + ext)):
            return os.path.join(out, name + ext)
    return None


# Network errors worth retrying. OSError covers timeouts, refused/reset
# connections and URLError on every Python version (3.9's socket.timeout too).
NET_ERRORS = (OSError, http.client.HTTPException)


def capture(base, retries=3, timeout=180):
    last_error = None
    for attempt in range(retries + 1):
        try:
            data = get(f"{base}/capture?t={time.time():.0f}", timeout=timeout)
            if data[:2] == b"BM" and len(data) == int.from_bytes(data[2:6], "little"):
                return data
            if data[:2] == b"\xff\xd8":
                # JPEG: the camera may add filler bytes after the end marker (FF D9),
                # so look for the last end marker and cut there.
                end = data.rfind(b"\xff\xd9")
                if end > 0:
                    return data[:end + 2]
            last_error = f"incomplete image ({len(data)} bytes)"
        except NET_ERRORS as e:
            last_error = e
        if attempt < retries:
            print(f"    retrying photo ({last_error})")
            time.sleep(3)
    raise RuntimeError(f"could not get a photo: {last_error}")


def median_photo(base, shots):
    """Takes `shots` photos and keeps the median of each pixel value.
    A glitch or grain that appears in only one photo is removed."""
    photos = [capture(base) for _ in range(shots)]
    if shots < 3:
        return photos[0]
    mid = shots // 2
    if photos[0][:2] == b"BM":
        offset = int.from_bytes(photos[0][10:14], "little")    # where pixel data starts
        header = photos[0][:offset]
        pixels = [ph[offset:] for ph in photos]
        if len({len(px) for px in pixels}) != 1:
            return photos[0]
        return header + bytes(sorted(vals)[mid] for vals in zip(*pixels))
    # JPEG: decode, merge, and return the result as a lossless PNG.
    try:
        from PIL import Image
    except ImportError:
        print("    (--shots needs Pillow for JPEG photos: run  pip3 install pillow ; using 1 shot)")
        return photos[0]
    imgs = [Image.open(io.BytesIO(ph)).convert("RGB") for ph in photos]
    if len({im.size for im in imgs}) != 1:
        return photos[0]
    raw = [im.tobytes() for im in imgs]
    merged = Image.frombytes("RGB", imgs[0].size, bytes(sorted(v)[mid] for v in zip(*raw)))
    buf = io.BytesIO()
    merged.save(buf, "PNG")
    return buf.getvalue()


_warned_no_pillow = False


def _background_model(img, rough_limit=70, cells=(40, 30), sub=10):
    """Estimates the background colour across the photo as a smooth image, so that
    uneven lighting (darker corners, tinted areas, gradients) is handled.

    The photo is divided into cells. Only plain cells (almost all pixels roughly
    background-coloured and little variation inside) are used; cells containing
    the figure, edges or other objects are filled in smoothly from the plain ones."""
    from PIL import Image
    w, h = img.size
    cw, ch = cells
    small = img.resize((cw * sub, ch * sub), Image.BOX)
    spx = small.load()

    # Rough background colour: median along the photo's border.
    sw, sh = small.size
    border = [spx[x, 0] for x in range(sw)] + [spx[x, sh - 1] for x in range(sw)] + \
             [spx[0, y] for y in range(sh)] + [spx[sw - 1, y] for y in range(sh)]
    bg = tuple(sorted(c[i] for c in border)[len(border) // 2] for i in range(3))

    grid = [[None] * cw for _ in range(ch)]
    for cy in range(ch):
        for cx in range(cw):
            pix = [spx[cx * sub + i, cy * sub + j] for j in range(sub) for i in range(sub)]
            near = [p for p in pix if max(abs(p[0] - bg[0]), abs(p[1] - bg[1]), abs(p[2] - bg[2])) <= rough_limit]
            if len(near) < 0.95 * len(pix):
                continue                                   # contains the figure or an object
            lum = sorted(p[0] + p[1] + p[2] for p in pix)
            if lum[int(0.95 * len(lum)) - 1] - lum[int(0.05 * len(lum))] > 60:
                continue                                   # has detail or an edge: not plain background
            grid[cy][cx] = tuple(sorted(p[i] for p in pix)[len(pix) // 2] for i in range(3))

    known = [[v is not None for v in row] for row in grid]
    if not any(any(r) for r in known):
        return Image.new("RGB", (w, h), bg)
    # First fill unknown cells from their neighbours, then smooth them (known cells stay fixed).
    while any(v is None for row in grid for v in row):
        new = [row[:] for row in grid]
        for y in range(ch):
            for x in range(cw):
                if grid[y][x] is None:
                    nb = [grid[yy][xx] for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                          if 0 <= yy < ch and 0 <= xx < cw and grid[yy][xx] is not None]
                    if nb:
                        new[y][x] = tuple(sum(c[i] for c in nb) / len(nb) for i in range(3))
        grid = new
    for _ in range(60):
        new = [row[:] for row in grid]
        for y in range(ch):
            for x in range(cw):
                if not known[y][x]:
                    nb = [grid[yy][xx] for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                          if 0 <= yy < ch and 0 <= xx < cw]
                    new[y][x] = tuple(sum(c[i] for c in nb) / len(nb) for i in range(3))
        grid = new

    model = Image.new("RGB", cells)
    model.putdata([tuple(int(round(c)) for c in grid[y][x]) for y in range(ch) for x in range(cw)])
    return model.resize((w, h), Image.BILINEAR)


def remove_background(img, threshold=18, shrink=4):
    """Returns an RGBA copy of `img` with the plain background made transparent
    (and painted white, for viewers that ignore transparency).

    The background colour is estimated across the photo (it may vary with uneven
    lighting). Pixels within `threshold` (0-255) of it that are connected to the
    photo's border become transparent; background-coloured areas inside the object
    (e.g. a white shirt) are kept. Needs Pillow."""
    from PIL import Image, ImageChops, ImageDraw, ImageFilter
    img = img.convert("RGB")
    w, h = img.size

    model = _background_model(img)
    diff = ImageChops.difference(img, model)
    r, g, b = diff.split()
    dist = ImageChops.lighter(ImageChops.lighter(r, g), b)
    candidate = dist.point(lambda v: 255 if v <= threshold else 0)      # 255 = looks like background

    # Keep only background regions connected to the border (flood fill at reduced size, for speed).
    small = candidate.resize((max(1, w // shrink), max(1, h // shrink)), Image.NEAREST)
    sw, sh = small.size
    spx = small.load()
    seeds = [(x, 0) for x in range(sw)] + [(x, sh - 1) for x in range(sw)] + \
            [(0, y) for y in range(sh)] + [(sw - 1, y) for y in range(sh)]
    for sx, sy in seeds:
        if spx[sx, sy] == 255:
            ImageDraw.floodfill(small, (sx, sy), 128)
    connected = small.point(lambda v: 255 if v == 128 else 0).resize((w, h), Image.NEAREST)
    connected = connected.filter(ImageFilter.MaxFilter(3))               # cover the block edges
    background = ImageChops.multiply(candidate, connected)               # full-res, connected only

    # Alpha: opaque object, transparent background; clean specks and soften the edge.
    alpha = ImageChops.invert(background)
    alpha = alpha.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))   # drop thin stray lines
    alpha = alpha.filter(ImageFilter.MedianFilter(5))
    alpha = alpha.filter(ImageFilter.GaussianBlur(1))
    rgb = Image.composite(img, Image.new("RGB", img.size, (255, 255, 255)), alpha)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def rotate_photo(data, degrees):
    """Rotates a photo clockwise; returns JPEG (high quality) or PNG bytes."""
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    fmt = "JPEG" if img.format == "JPEG" else "PNG"
    img = img.convert("RGB").rotate(-degrees, expand=True)
    buf = io.BytesIO()
    img.save(buf, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return buf.getvalue()


_rembg_session = None


def remove_background_ai(img):
    """Background removal with the rembg AI model (u2net): recognises the object
    itself, so it also removes props, mounts and enclosed background.
    Needs:  pip3 install "rembg[cpu]"  (the model, ~170 MB, downloads on first use)."""
    global _rembg_session
    from PIL import Image
    from rembg import new_session, remove
    if _rembg_session is None:
        _rembg_session = new_session("u2net")
    out = remove(img.convert("RGB"), session=_rembg_session).convert("RGBA")
    alpha = out.split()[3]
    rgb = Image.composite(out.convert("RGB"), Image.new("RGB", out.size, (255, 255, 255)), alpha)
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    return result


def save_without_background(path, out_path, threshold, ai=False):
    """Writes a transparent-background PNG of the photo at `path`."""
    from PIL import Image
    img = Image.open(path)
    (remove_background_ai(img) if ai else remove_background(img, threshold)).save(out_path)


def save(data, path_no_ext, want_png, despeckle=0):
    """Saves the photo.
    JPEG / PNG data is written as is (.jpg / .png), unless --despeckle is used.
    BMP data is converted to PNG when Pillow is installed (unless want_png is False).
    With `despeckle` > 0 (needs Pillow), stray pixels that differ from their
    neighbours' median by more than `despeckle` are replaced by that median."""
    global _warned_no_pillow
    kind = "bmp" if data[:2] == b"BM" else "png" if data[:4] == b"\x89PNG" else "jpg"
    raw_path = f"{path_no_ext}.{kind}"
    if not despeckle and (kind != "bmp" or not want_png):
        with open(raw_path, "wb") as f:
            f.write(data)
        return raw_path
    try:
        from PIL import Image, ImageChops, ImageFilter
    except ImportError:
        if not _warned_no_pillow:
            print("    (Pillow not installed: run  pip3 install pillow ; saving the photo unprocessed)")
            _warned_no_pillow = True
        with open(raw_path, "wb") as f:
            f.write(data)
        return raw_path
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if despeckle:
        med = img.filter(ImageFilter.MedianFilter(3))
        diff = ImageChops.difference(img, med).convert("L")
        mask = diff.point(lambda v: 255 if v > despeckle else 0)
        img = Image.composite(med, img, mask)
    if kind == "jpg":
        out_path = path_no_ext + ".jpg"
        img.save(out_path, quality=95)
    else:
        out_path = path_no_ext + (".png" if want_png else ".bmp")
        img.save(out_path)
    return out_path


def plan(angles, ratio, servo_max):
    """For each turntable angle, works out the servo angle and whether the figure
    must have been turned 180° by hand first. Returns [(angle, servo_deg, flipped)]."""
    steps = []
    for a in angles:
        if a / ratio < servo_max:
            steps.append((a, a / ratio, False))
        elif (a - 180) / ratio < servo_max and a >= 180:
            steps.append((a, (a - 180) / ratio, True))
        else:
            raise SystemExit(f"Angle {a}° is out of reach with --gear-ratio {ratio}")
    # All "before flip" angles first, then all "after flip" ones.
    return sorted(steps, key=lambda s: (s[2], s[0]))


def main():
    p = argparse.ArgumentParser(description="Move the turntable to each angle and save a photo.")
    p.add_argument("ip", nargs="?", help="ESP32 address shown in Serial Monitor, e.g. 192.168.0.92")
    p.add_argument("--angles", default="0,45,90,135,180,225,270,315",
                   help="turntable angles to photograph (default: every 45°)")
    p.add_argument("--angle-step", type=float, default=None, metavar="DEG",
                   help="photograph every DEG degrees around the full turn, e.g. 30 or 22.5 "
                        "(replaces --angles)")
    p.add_argument("--gear-ratio", type=float, default=1,
                   help="turntable degrees per servo degree: 1 = no gears (one manual flip), 2 = 2:1 gears")
    p.add_argument("--servo-max", type=float, default=180,
                   help="largest servo angle to use (default 180)")
    p.add_argument("--tilts", default="0,15,30",
                   help="tilt angles to photograph (default 0,15,30); use 0 for no tilting")
    p.add_argument("--tilt-level", type=float, default=90,
                   help="tilt-servo angle where the figure stands level (default 90)")
    p.add_argument("--tilt-reverse", action="store_true",
                   help="tilt the other way (if positive tilts lean the figure the wrong way)")
    p.add_argument("--shots", type=int, default=1,
                   help="photos per angle, combined to remove noise (default 1 = no combining)")
    p.add_argument("--despeckle", type=int, nargs="?", const=40, default=0, metavar="LEVEL",
                   help="remove stray wrong pixels (needs Pillow). LEVEL = how different a pixel must be "
                        "from its neighbours to be replaced: lower = stronger (default 40)")
    p.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                   help="rotate every photo clockwise by this many degrees (e.g. 90 if the camera "
                        "is mounted sideways)")
    p.add_argument("--png", action="store_true",
                   help="also save every photo as PNG (needs Pillow)")
    p.add_argument("--remove-bg", action="store_true",
                   help="save the photo as a PNG with the background made transparent (needs Pillow). "
                        "Only that PNG is kept, unless you add --keep-original")
    p.add_argument("--keep-original", action="store_true",
                   help="with --remove-bg / --ai: also keep the original photo; the cutout is "
                        "then saved as <name>_nobg.png")
    p.add_argument("--bg-threshold", type=int, default=18, metavar="N",
                   help="how close (0-255) a colour must be to the background to be removed "
                        "(default 18). Raise it if background is left over, lower it if parts "
                        "of the figure disappear")
    p.add_argument("--ai", action="store_true",
                   help="use the rembg AI model for --remove-bg / --remove-bg-folder: much cleaner "
                        "cutouts (removes the turntable too). Needs:  pip3 install \"rembg[cpu]\"")
    p.add_argument("--remove-bg-folder", metavar="FOLDER",
                   help="don't take photos: make transparent PNGs for every photo already in FOLDER")
    p.add_argument("--settle", type=float, default=1.5, help="seconds to wait after moving before the photo")
    p.add_argument("--out", default=None,
                   help="output folder (default: photos_<date>_<time>). Reusing a folder resumes the run: "
                        "photos already in it are skipped")
    p.add_argument("--bmp", action="store_true", help="keep BMP files even if Pillow is installed")
    p.add_argument("--dry-run", action="store_true", help="move to every angle without taking photos")
    args = p.parse_args()

    if args.ai:
        try:
            import rembg  # noqa: F401
        except ImportError:
            raise SystemExit('--ai needs rembg: run  pip3 install "rembg[cpu]"')
        if not (args.remove_bg or args.remove_bg_folder):
            args.remove_bg = True          # --ai on its own means: remove the background with AI
    if args.remove_bg_folder:
        try:
            import PIL  # noqa: F401
        except ImportError:
            raise SystemExit("Background removal needs Pillow: run  pip3 install pillow")
        files = sorted(f for f in os.listdir(args.remove_bg_folder)
                       if f.lower().endswith((".jpg", ".png", ".bmp")) and "_nobg" not in f)
        for f in files:
            src = os.path.join(args.remove_bg_folder, f)
            dst = os.path.join(args.remove_bg_folder, os.path.splitext(f)[0] + "_nobg.png")
            save_without_background(src, dst, args.bg_threshold, ai=args.ai)
            print(f"  {f} -> {os.path.basename(dst)}")
        print(f"Done: {len(files)} photos.")
        return
    if not args.ip:
        p.error("the ESP32 address is required (unless you use --remove-bg-folder)")
    if args.remove_bg or args.png or args.rotate:
        try:
            import PIL  # noqa: F401
        except ImportError:
            raise SystemExit("--png, --remove-bg and --rotate need Pillow: run  pip3 install pillow")

    base = "http://" + args.ip.removeprefix("http://").rstrip("/")
    if args.angle_step:
        if not 0 < args.angle_step <= 180:
            raise SystemExit("--angle-step must be between 0 and 180")
        n = int(360 / args.angle_step) + 1
        angles = [round(i * args.angle_step, 1) for i in range(n) if i * args.angle_step < 360 - 1e-6]
    else:
        angles = [float(a) % 360 for a in args.angles.split(",")]
    angles = [int(a) if a == int(a) else a for a in angles]   # 45.0 -> 45, keep 22.5
    tilts = [int(t) for t in args.tilts.split(",")]
    steps = plan(angles, args.gear_ratio, args.servo_max)
    tilt_sign = -1 if args.tilt_reverse else 1
    for t in tilts:
        if not 0 <= args.tilt_level + tilt_sign * t <= 180:
            raise SystemExit(f"Tilt {t}° is out of the tilt servo's range with --tilt-level {args.tilt_level}")
    total = len(steps) * len(tilts)

    out = None
    if not args.dry_run:
        out = args.out or datetime.datetime.now().strftime("photos_%Y%m%d_%H%M%S")
        os.makedirs(out, exist_ok=True)
        print(f"Saving {total} photos to {os.path.abspath(out)}")

    print("Moving to the start position...")
    move_servo(base, args.tilt_level + tilt_sign * tilts[0], "tilt")
    move_servo(base, steps[0][1])
    if not args.dry_run:
        print("Warming up the camera...")
        capture(base)          # one throwaway photo lets auto-exposure settle

    # Grouped so the manual flip (no-gear setup) happens only once per run:
    # all tilts for the first half of the turn, then the flip, then all tilts again.
    n = 0
    flipped = False
    for half in (False, True):
        half_steps = [s for s in steps if s[2] == half]
        if not half_steps:
            continue
        if not args.dry_run and all(existing(out, photo_name(a, t))
                                    for t in tilts for a, _, _ in half_steps):
            n += len(half_steps) * len(tilts)
            continue                     # this whole half is already done
        if half and not flipped:
            move_servo(base, 0)
            move_servo(base, args.tilt_level, "tilt")
            input("\n  >>> Turn the FIGURE 180° on the turntable (it should face the opposite way),\n"
                  "      then press Enter to continue... ")
            flipped = True
        for tilt in tilts:
            move_servo(base, args.tilt_level + tilt_sign * tilt, "tilt")
            for angle, servo_deg, _ in half_steps:
                n += 1
                name = photo_name(angle, tilt)
                if not args.dry_run and existing(out, name):
                    print(f"  [{n}/{total}] tilt {tilt:2d}°  angle {angle:>5}°  already saved, skipping")
                    continue
                move_servo(base, servo_deg)
                time.sleep(args.settle)
                label = f"[{n}/{total}] tilt {tilt:2d}°  angle {angle:>5}°  (servo {servo_deg:.1f}°)"
                if args.dry_run:
                    print("  " + label)
                    continue
                t0 = time.time()
                data = median_photo(base, max(1, args.shots))
                if args.rotate:
                    data = rotate_photo(data, args.rotate)
                path = save(data, os.path.join(out, name), want_png=not args.bmp,
                            despeckle=args.despeckle)
                if args.remove_bg and not args.keep_original:
                    # Save only the background-free PNG, named like the photo (angle_045_tilt_10.png).
                    final = os.path.join(out, name + ".png")
                    tmp = os.path.join(out, name + "_tmp_nobg.png")
                    save_without_background(path, tmp, args.bg_threshold, ai=args.ai)
                    os.remove(path)
                    os.replace(tmp, final)
                    print(f"  {label} -> {os.path.basename(final)}  ({time.time() - t0:.1f} s)")
                    continue
                extra = ""
                if args.png and not path.endswith(".png"):
                    from PIL import Image
                    Image.open(path).save(os.path.join(out, name + ".png"))
                    extra += f" + {name}.png"
                if args.remove_bg:
                    save_without_background(path, os.path.join(out, name + "_nobg.png"),
                                            args.bg_threshold, ai=args.ai)
                    extra += f" + {name}_nobg.png"
                print(f"  {label} -> {os.path.basename(path)}{extra}  ({time.time() - t0:.1f} s)")

    move_servo(base, 0)
    move_servo(base, args.tilt_level, "tilt")
    if flipped:
        print("Done. (Turn the figure back to its original position for the next run.)")
    else:
        print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except (RuntimeError,) + NET_ERRORS as e:
        sys.exit(f"Error talking to the ESP32: {e}\n"
                 "To continue where it stopped, run the same command again with --out <that folder>.")
