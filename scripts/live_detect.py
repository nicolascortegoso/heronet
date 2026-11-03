#!/usr/bin/env python3
"""
Real-time recognition on the ESP32-CAM live stream (Mac or Raspberry Pi):

  1. the character detector (YOLO) finds the figure and names the character
  2. the figure is cropped (square, exactly like make_pose_crops.py does for training)
  3. the arm-position classifier (YOLO) names the pose on that crop

Models can be .pt files or NCNN folders (made by export_models.py; fastest on a Raspberry Pi).

Needs:  pip3 install ultralytics      (+ ncnn for NCNN models; + "rembg[cpu]" for --ai photo mode)

Usage:
  python3 live_detect.py <ESP32-IP> --det <detector> --pose <classifier>
  python3 live_detect.py <ESP32-IP> --det det_ncnn_model --pose pose_ncnn_model --serve 8000
      -> also watch it in a browser: http://<this computer's IP>:8000
  python3 live_detect.py --image photo.png --det ... --pose ...      (one image file)
  python3 live_detect.py <ESP32-IP> --photos --ai --det ...          (full-size photos, slow)

Keys in the window:  q = quit   space = pause/resume   s = save the current frame
"""

import argparse
import http.server
import os
import socketserver
import sys
import threading
import time
import urllib.request

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("This tool needs OpenCV (installed with Ultralytics):  pip3 install ultralytics")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_effects import square_crop_box  # noqa: E402

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


# ----------------------------------------------------------------------------- camera input

class StreamReader(threading.Thread):
    """Reads the ESP32-CAM MJPEG stream in the background and keeps only the newest
    frame, so a slow step never makes the picture lag behind reality."""

    def __init__(self, url, timeout=10):
        super().__init__(daemon=True)
        self.url, self.timeout = url, timeout
        self.frame, self.jpeg, self.frame_id, self.error = None, None, 0, None
        self.cond = threading.Condition()
        self.running = True
        self.cam_fps, self._times = 0.0, []

    def _frames(self, resp):
        """Yields the JPEG bytes of each part of a multipart/x-mixed-replace response."""
        while self.running:
            line = resp.readline()
            if not line:
                return                                        # connection closed
            if not line.lower().startswith(b"content-length:"):
                continue                                      # boundary or other header
            length = int(line.split(b":", 1)[1])
            while resp.readline() not in (b"\r\n", b"\n", b""):
                pass                                          # rest of the part headers
            data = resp.read(length)
            if len(data) < length:
                return
            yield data

    def run(self):
        while self.running:
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                    self.error = None
                    for data in self._frames(resp):
                        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        now = time.time()
                        self._times = [t for t in self._times if now - t < 2] + [now]
                        self.cam_fps = (len(self._times) - 1) / max(1e-6, now - self._times[0])
                        with self.cond:
                            self.frame, self.jpeg, self.frame_id = img, data, self.frame_id + 1
                            self.cond.notify_all()
            except (OSError, ValueError) as e:
                self.error = str(e)
            if self.running:
                print(f"  stream interrupted ({self.error or 'closed'}), reconnecting...")
                time.sleep(1)

    def latest(self, after_id, timeout=5):
        """Waits for a frame newer than after_id; returns (frame, jpeg_bytes, id) or (None, None, after_id)."""
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id > after_id or not self.running, timeout)
            if self.frame_id > after_id:
                return self.frame, self.jpeg, self.frame_id
            return None, None, after_id

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()


def stream_url(ip, size, quality):
    """The sketch serves the stream on port 81 (so servos can move meanwhile).
    An address with its own port (e.g. 192.168.1.50:81) is used as given."""
    host = ip if ":" in ip else f"{ip}:81"
    return f"http://{host}/stream?size={size}&q={quality}"


def grab_photo(ip, timeout):
    """One full-size JPEG from /capture (slow, but the best quality)."""
    data = urllib.request.urlopen(f"http://{ip}/capture", timeout=timeout).read()
    end = data.rfind(b"\xff\xd9")                 # the camera pads JPEGs after the end marker
    if end < 0:
        raise ValueError("incomplete image")
    img = cv2.imdecode(np.frombuffer(data[:end + 2], np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


class BackgroundRemover:
    """Same AI cutout as turntable.py --ai, pasted on white. Too slow for the stream."""

    def __init__(self):
        try:
            from rembg import new_session, remove
            from PIL import Image
        except ImportError:
            sys.exit('--ai needs rembg:  pip3 install "rembg[cpu]"')
        self.remove, self.Image = remove, Image
        self.session = new_session("u2net")

    def __call__(self, bgr):
        pil = self.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cut = self.remove(pil, session=self.session).convert("RGBA")
        white = self.Image.new("RGBA", cut.size, (255, 255, 255, 255))
        white.alpha_composite(cut)
        return cv2.cvtColor(np.array(white.convert("RGB")), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------- models

class Recognizer:
    """Detector (+ optional classifier). Works with .pt files and exported NCNN folders."""

    def __init__(self, det, pose=None, det_imgsz=320, pose_imgsz=224, conf=0.4, max_figures=1,
                 margin=0.10, device=None, verbose=True):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("This tool needs Ultralytics:  pip3 install ultralytics")
        self.det = YOLO(det, task="detect")
        self.pose = YOLO(pose, task="classify") if pose else None
        self.det_imgsz, self.pose_imgsz, self.conf = det_imgsz, pose_imgsz, conf
        self.max_figures, self.margin, self.device = max_figures, margin, device
        self.det_ms = self.pose_ms = 0.0
        if verbose:
            print("Characters: " + ", ".join(self.det.names[i] for i in sorted(self.det.names)))
            if self.pose is not None:
                print("Arm positions: " + ", ".join(self.pose.names[i] for i in sorted(self.pose.names)))

    def warmup(self):
        dummy = np.full((240, 320, 3), 255, np.uint8)
        for _ in range(2):
            self(dummy)

    def __call__(self, frame):
        """Returns a list of (box, character, conf, pose or None, pose_conf), best first."""
        t0 = time.perf_counter()
        r = self.det.predict(frame, imgsz=self.det_imgsz, conf=self.conf, device=self.device,
                             max_det=self.max_figures, verbose=False)[0]
        found = []
        for b in r.boxes[: self.max_figures]:
            box = [float(v) for v in b.xyxy[0]]
            found.append([box, self.det.names[int(b.cls)], float(b.conf), None, 0.0])
        t1 = time.perf_counter()
        if self.pose is not None and found:
            h, w = frame.shape[:2]
            crops = []
            for f in found:
                x0, y0, x1, y1 = square_crop_box(f[0], w, h, self.margin)
                crops.append(frame[y0:y1, x0:x1])
            for f, pr in zip(found, self.pose.predict(crops, imgsz=self.pose_imgsz, device=self.device,
                                                      verbose=False)):
                f[3], f[4] = self.pose.names[int(pr.probs.top1)], float(pr.probs.top1conf)
        t2 = time.perf_counter()
        self.det_ms, self.pose_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
        return found


# ----------------------------------------------------------------------------- output

def draw(frame, found, info, scale):
    h, w = frame.shape[:2]
    view = cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_NEAREST if scale >= 2 else cv2.INTER_LINEAR)
    for box, char, conf, pose, pconf in found:
        x0, y0, x1, y1 = (int(v * scale) for v in box)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 200, 0), 2)
        label = f"{char} {conf:.0%}" + (f" | {pose} {pconf:.0%}" if pose else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(th + 6, y0)
        cv2.rectangle(view, (x0, ty - th - 6), (x0 + tw + 6, ty), (0, 200, 0), -1)
        cv2.putText(view, label, (x0 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(view, (0, view.shape[0] - th - 12), (tw + 12, view.shape[0]), (0, 0, 0), -1)
    cv2.putText(view, info, (6, view.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view


def describe(found):
    if not found:
        return "nothing detected"
    return ", ".join(f"{c} {cf:.0%}" + (f" / {p} {pc:.0%}" if p else "") for _, c, cf, p, pc in found)


class BrowserView:
    """Serves the annotated video at http://<ip>:<port>/ for any browser on the network."""

    PAGE = (b"<!doctype html><html><head><meta name=viewport content='width=device-width'>"
            b"<title>live_detect</title></head><body style='margin:0;background:#111;text-align:center'>"
            b"<img src='/stream' style='max-width:100%;max-height:100vh'></body></html>")

    def __init__(self, port):
        self.jpeg, self.cond = None, threading.Condition()
        view = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            with view.cond:
                                view.cond.wait_for(lambda: view.jpeg is not last, timeout=5)
                                data = last = view.jpeg
                            if data is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                             % len(data) + data + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(view.PAGE)

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = Server(("0.0.0.0", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, bgr):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()


def local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))                      # no data is sent; just picks the network interface
        return s.getsockname()[0]
    except OSError:
        return "<this computer's IP>"


def has_display():
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Run the character detector + arm-position classifier live.")
    ap.add_argument("ip", nargs="?", help="ESP32-CAM IP address")
    ap.add_argument("--det", required=True, help="character detector: best.pt or an NCNN folder")
    ap.add_argument("--pose", help="arm-position classifier: best.pt or an NCNN folder")
    ap.add_argument("--size", default="qvga", choices=["qqvga", "qvga", "hvga", "vga", "svga"],
                    help="stream size (default qvga = 320x240)")
    ap.add_argument("--quality", type=int, default=12, help="stream JPEG quality 8..63, lower = better (default 12)")
    ap.add_argument("--photos", action="store_true", help="use full-size /capture photos instead of the stream")
    ap.add_argument("--image", help="test one image file instead of the camera")
    ap.add_argument("--ai", action="store_true", help="remove the background first (photo/image mode only)")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="rotate camera frames")
    ap.add_argument("--conf", type=float, default=0.4, help="minimum detection confidence (default 0.4)")
    ap.add_argument("--max-figures", type=int, default=1, help="figures to label per frame (default 1)")
    ap.add_argument("--det-imgsz", type=int, default=320, help="detector input size, as trained (default 320)")
    ap.add_argument("--pose-imgsz", type=int, default=224, help="classifier input size, as trained (default 224)")
    ap.add_argument("--margin", type=float, default=0.10, help="crop margin, as in make_pose_crops.py (0.10)")
    ap.add_argument("--device", default=None, help="mps, cpu ... (default: automatic)")
    ap.add_argument("--serve", type=int, default=0, metavar="PORT",
                    help="also show the annotated video in a browser on this port (e.g. 8000)")
    ap.add_argument("--scale", type=float, default=0, help="window zoom (default: fit about 800 px)")
    ap.add_argument("--out", default="live_results", help="folder for saved frames")
    ap.add_argument("--no-window", action="store_true", help="no window (e.g. over SSH): print / serve only")
    ap.add_argument("--timeout", type=float, default=10, help="camera timeout in seconds")
    args = ap.parse_args()
    if not args.image and not args.ip:
        sys.exit("Give the ESP32-CAM IP address, or --image <file>.")
    if args.ai and not (args.photos or args.image):
        sys.exit("--ai is too slow for the stream: use it with --photos or --image.")

    show = not args.no_window
    if show and not has_display():
        print("No screen found (SSH session?): running without a window. Use --serve to watch in a browser.")
        show = False

    rec = Recognizer(args.det, args.pose, args.det_imgsz, args.pose_imgsz, args.conf, args.max_figures,
                     args.margin, args.device)
    print("Warming up the models...")
    rec.warmup()
    remover = BackgroundRemover() if args.ai else None
    os.makedirs(args.out, exist_ok=True)
    web = None
    if args.serve:
        web = BrowserView(args.serve)
        print(f"Browser view: http://{local_ip()}:{args.serve}/")

    reader = None
    if not args.image and not args.photos:
        url = stream_url(args.ip, args.size, args.quality)
        print(f"Opening {url}")
        reader = StreamReader(url, args.timeout)
        reader.start()

    title = "live_detect (q quit, space pause, s save)"
    last_id, paused, view, n = 0, False, None, 0
    times, last_print = [], 0.0
    try:
        while True:
            if not paused or view is None:
                if reader:
                    frame, _, last_id = reader.latest(last_id)
                    if frame is None:
                        if show and cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue
                else:
                    try:
                        frame = cv2.imread(args.image) if args.image else grab_photo(args.ip, args.timeout)
                    except (OSError, ValueError) as e:
                        print(f"  camera error: {e} - retrying")
                        time.sleep(1)
                        continue
                    if frame is None:
                        sys.exit(f"Can't open {args.image}")
                if ROTATIONS[args.rotate] is not None:
                    frame = cv2.rotate(frame, ROTATIONS[args.rotate])
                if remover:
                    frame = remover(frame)

                t = time.time()
                found = rec(frame)
                n += 1
                times = [x for x in times if t - x < 2] + [t]
                fps = (len(times) - 1) / max(1e-6, t - times[0])
                h, w = frame.shape[:2]
                info = f"{w}x{h} | det {rec.det_ms:.0f} ms"
                if rec.pose is not None:
                    info += f" + pose {rec.pose_ms:.0f} ms"
                info += f" | {fps:.1f} fps"
                if reader:
                    info += f" (camera {reader.cam_fps:.1f})"
                if not reader or t - last_print > 0.5:          # don't flood the terminal
                    print(f"[{n}] {describe(found)}   ({info})")
                    last_print = t
                scale = args.scale or max(1.0, min(4.0, 800 / max(h, w)))
                view = draw(frame, found, info, scale)
                if web:
                    web.publish(view)

            if not show:
                if args.image:
                    path = os.path.join(args.out, "result_" + os.path.basename(args.image).rsplit(".", 1)[0] + ".jpg")
                    cv2.imwrite(path, view)
                    print(f"  saved {path}")
                    if web:
                        print("  (browser view stays on: Ctrl+C to quit)")
                        while True:
                            time.sleep(1)
                    break
                continue
            cv2.imshow(title, view)
            key = cv2.waitKey(0 if args.image else 1) & 0xFF
            if key == ord("q") or args.image:
                if args.image:
                    path = os.path.join(args.out, "result_" + os.path.basename(args.image).rsplit(".", 1)[0] + ".jpg")
                    cv2.imwrite(path, view)
                    print(f"  saved {path}")
                break
            if key == ord(" "):
                paused = not paused
                print("  paused" if paused else "  resumed")
            if key == ord("s"):
                path = os.path.join(args.out, f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{n}.jpg")
                cv2.imwrite(path, view)
                print(f"  saved {path}")
    except KeyboardInterrupt:
        pass
    finally:
        if reader:
            reader.stop()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
