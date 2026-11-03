"""
Shared helpers for make_yolo_dataset.py, make_pose_crops.py and live_detect.py:
make training images look like live ESP32-CAM stream frames.

  random_background()  plain white, one of your own background photos, or a generated one
  low_res()            shrink + enlarge + JPEG, like a small (QVGA/VGA) stream frame
  square_crop_box()    the square crop around a figure, identical in training and live use
"""

import io
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_backgrounds(folder):
    """Paths of the background photos in a folder (searched recursively)."""
    if not folder:
        return []
    paths = []
    for root, _, files in os.walk(folder):
        paths += [os.path.join(root, f) for f in files if f.lower().endswith(IMAGE_EXT) and not f.startswith(".")]
    if not paths:
        raise SystemExit(f"No images found in --backgrounds folder {folder}")
    return sorted(paths)


def _generated(size, rng):
    """A random synthetic background: flat colour, gradient, blotchy texture or shapes."""
    w, h = size
    rc = lambda lo=0, hi=255: tuple(rng.randint(lo, hi) for _ in range(3))
    kind = rng.choice(["flat", "gradient", "texture", "shapes"])
    if kind == "flat":
        img = Image.new("RGB", size, rc())
    elif kind == "gradient":
        a, b = rc(), rc()
        strip = Image.new("RGB", (256, 1))
        for i in range(256):
            t = i / 255
            strip.putpixel((i, 0), tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3)))
        img = strip.resize((max(w, h) * 2, max(w, h) * 2), Image.BILINEAR).rotate(rng.uniform(0, 360))
        cx, cy = img.size[0] // 2, img.size[1] // 2
        img = img.crop((cx - w // 2, cy - h // 2, cx - w // 2 + w, cy - h // 2 + h))
    elif kind == "texture":
        cells = rng.randint(3, 24)
        small = Image.new("RGB", (cells, cells))
        base = rc(40, 230)
        for x in range(cells):
            for y in range(cells):
                small.putpixel((x, y), tuple(max(0, min(255, c + rng.randint(-60, 60))) for c in base))
        img = small.resize(size, Image.BICUBIC)
    else:
        img = Image.new("RGB", size, rc())
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(2, 8)):
            x0, y0 = rng.randint(-w // 2, w), rng.randint(-h // 2, h)
            x1, y1 = x0 + rng.randint(w // 8, w), y0 + rng.randint(h // 8, h)
            (d.ellipse if rng.random() < 0.4 else d.rectangle)([x0, y0, x1, y1], fill=rc())
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 3)))
    return img


def _photo(path, size, rng):
    """A random crop of one of the user's background photos."""
    with Image.open(path) as im:
        im = im.convert("RGB")
    w, h = size
    f = max(w / im.size[0], h / im.size[1]) * rng.uniform(1.0, 2.0)
    im = im.resize((max(w, int(im.size[0] * f)), max(h, int(im.size[1] * f))), Image.BILINEAR)
    x, y = rng.randint(0, im.size[0] - w), rng.randint(0, im.size[1] - h)
    im = im.crop((x, y, x + w, y + h))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return ImageEnhance.Brightness(im).enhance(rng.uniform(0.7, 1.3))


def random_background(size, rng, photos=(), white_fraction=0.2, white=(255, 255, 255)):
    """RGB background: white with probability white_fraction, else one of the photos
    (if any were given; 70% of the time) or a generated one."""
    r = rng.random()
    if r < white_fraction:
        return Image.new("RGB", size, white)
    if photos and rng.random() < 0.7:
        return _photo(rng.choice(photos), size, rng)
    return _generated(size, rng)


def composite(figure_rgba, background_rgb):
    bg = background_rgb.convert("RGBA")
    bg.alpha_composite(figure_rgba)
    return bg.convert("RGB")


def low_res(img, rng, figure_height, fig_px=(80, 240), jpeg_quality=(35, 80)):
    """Makes an RGB image look like a small stream frame: shrinks it so the figure is
    fig_px pixels tall (random in that range), enlarges it back and adds JPEG compression.
    Returns (image, description)."""
    target = rng.uniform(*fig_px)
    ops = []
    if figure_height > target:
        f = target / figure_height
        w, h = img.size
        small = img.resize((max(8, round(w * f)), max(8, round(h * f))), Image.BOX)
        img = small.resize((w, h), Image.BILINEAR)
        ops.append(f"lowres{round(target)}px")
    q = rng.randint(*jpeg_quality)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    ops.append(f"jpeg{q}")
    return img, " ".join(ops)


def square_crop_box(box, width, height, margin=0.10):
    """Square around box (x0, y0, x1, y1) with `margin` (fraction of the box size) on each
    side, moved to stay inside the image. Used identically for training crops and live."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    side = int(round(min(side, width, height)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    return left, top, left + side, top + side


def jitter_box(box, rng, amount=0.05):
    """Moves each edge of a box by up to `amount` of its size: imitates an imperfect detector."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    j = lambda d: rng.uniform(-amount, amount) * d
    return x0 + j(w), y0 + j(h), x1 + j(w), y1 + j(h)


def seeded_rng(seed, key):
    import zlib
    return random.Random(seed ^ zlib.crc32(key.encode()))
