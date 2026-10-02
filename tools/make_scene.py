"""Turn a photo into a layered depth scene for index.html.

    python tools/make_scene.py photo.jpg                 # writes scene.js next to index.html
    python tools/make_scene.py photo.jpg --standalone beach.html

Steps:
  1. Depth Anything V2 estimates relative (inverse) depth for every pixel.
  2. The nearest large object (usually a person) is cut out as a foreground layer.
  3. LaMa fills in what's hidden behind it, so moving your head reveals plausible
     background instead of a smeared "rubber sheet".
  4. Everything is packed as data URIs into scene.js, which index.html picks up.

Requirements (CPU is fine, ~1-2 minutes per photo):
    pip install torch transformers pillow opencv-python-headless numpy

Note: the Large depth model is CC-BY-NC-4.0 (non-commercial). Pass
--depth-model depth-anything/Depth-Anything-V2-Small-hf for an Apache-2.0 model.
"""

import argparse
import base64
import io
import json
import pathlib
import re
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_PATH = pathlib.Path.home() / ".cache" / "3d-desktop" / "big-lama.pt"


def estimate_disparity(img, model_name):
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).eval()
    # A larger working resolution than the default 518 keeps hair edges sharper.
    long_side = 1036
    w, h = img.size
    size = {"height": long_side, "width": round(long_side * w / h)} if h >= w else \
           {"height": round(long_side * h / w), "width": long_side}
    inputs = proc(images=img, return_tensors="pt", size=size, keep_aspect_ratio=True, ensure_multiple_of=14)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    pred = torch.nn.functional.interpolate(pred[:, None], size=(h, w), mode="bicubic", align_corners=False)
    d = pred[0, 0].numpy()
    return (d - d.min()) / (d.max() - d.min())


def load_lama():
    if not LAMA_PATH.exists():
        LAMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading LaMa inpainting model (~200 MB)...")
        urllib.request.urlretrieve(LAMA_URL, LAMA_PATH)
    return torch.jit.load(str(LAMA_PATH), map_location="cpu").eval()


def inpaint(model, image, mask):
    """LaMa works best around 1 megapixel or less, so inpaint at half size and
    composite the result back only inside the (feathered) mask."""
    h, w = image.shape[:2]
    sw, sh = w // 2, h // 2
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    ph, pw = (8 - sh % 8) % 8, (8 - sw % 8) % 8
    im = np.pad(small, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mk = np.pad(small_mask, ((0, ph), (0, pw)), mode="reflect")
    it = torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255
    mt = torch.from_numpy((mk > 0).astype(np.float32))[None, None]
    with torch.no_grad():
        out = model(it, mt)[0].permute(1, 2, 0).numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)[:sh, :sw]
    up = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 3)[..., None]
    return (up * soft + image * (1 - soft)).astype(np.uint8)


def build_layers(img, d, threshold):
    h, w = d.shape

    # Foreground: the largest connected region nearer than the threshold.
    hard = (d > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
    if n < 2:
        raise SystemExit("No foreground found; try a lower --threshold.")
    hard = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    near = cv2.dilate(hard, np.ones((9, 9), np.uint8))
    alpha = np.clip((d - (threshold - 0.08)) / 0.16, 0, 1) * near
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.2)

    # Grow the hole generously so no foreground-coloured fringe stays behind.
    k = max(9, round(min(h, w) * 0.033)) | 1
    hole = cv2.dilate((alpha > 0.01).astype(np.uint8) | hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    # Background depth behind the foreground: ground depth mostly depends on the
    # image row, so interpolate across each row, then smooth.
    d_bg = d.copy()
    xs = np.arange(w)
    for y in range(h):
        m = hole[y] > 0
        if m.any() and (~m).any():
            d_bg[y, m] = np.interp(xs[m], xs[~m], d[y, ~m])
    d_bg = cv2.GaussianBlur(d_bg.astype(np.float32), (0, 0), 15) * hole + d_bg * (1 - hole)

    # Foreground depth: extend the edge values outward so the layer's mesh doesn't
    # stretch back toward the background along its silhouette.
    _, idx = cv2.distanceTransformWithLabels(1 - hard, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros((idx.max() + 1, 2), np.int64)
    lut[1:] = np.argwhere(hard > 0)[: idx.max()]
    src = lut[idx]
    d_fg = np.where(hard > 0, d, d[src[..., 0], src[..., 1]])
    # Smooth heavily: depth jumps *inside* the cut-out (hair in front of a jacket,
    # say) would otherwise tear into streaks when the view moves.
    d_fg = cv2.GaussianBlur(d_fg.astype(np.float32), (0, 0), min(h, w) * 0.02)

    return alpha, hole, d_bg, d_fg


def data_uri(arr, fmt):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {"optimize": True}))
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("--out", default=str(ROOT / "scene.js"))
    ap.add_argument("--standalone", help="also write a single self-contained HTML file")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="relative depth (0 = far, 1 = near) that separates foreground from background")
    ap.add_argument("--focus", type=float, default=0.5,
                    help="vertical point of interest, 0 = top of photo, 1 = bottom")
    ap.add_argument("--max-size", type=int, default=2208, help="longest side of the stored images")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Large-hf")
    args = ap.parse_args()

    pil = ImageOps.exif_transpose(Image.open(args.photo)).convert("RGB")
    pil.thumbnail((args.max_size, args.max_size), Image.LANCZOS)
    img = np.array(pil)

    print("Estimating depth...")
    d = estimate_disparity(pil, args.depth_model)
    print("Separating layers...")
    alpha, hole, d_bg, d_fg = build_layers(img, d, args.threshold)
    print("Inpainting background...")
    bg = inpaint(load_lama(), img, hole * 255)

    # depth.png: R = background depth, G = foreground depth, B = foreground alpha.
    packed = np.dstack([d_bg, d_fg, alpha])
    packed = (np.clip(packed, 0, 1) * 255).round().astype(np.uint8)

    scene = {
        "width": img.shape[1],
        "height": img.shape[0],
        "focus": args.focus,
        "photo": data_uri(img, "JPEG"),
        "background": data_uri(bg, "JPEG"),
        "depth": data_uri(packed, "PNG"),
    }
    js = f"window.SCENE = {json.dumps(scene)};\n"
    pathlib.Path(args.out).write_text(js)
    print(f"Wrote {args.out} ({len(js) / 1e6:.1f} MB)")

    if args.standalone:
        html = (ROOT / "index.html").read_text()
        tag = '<script src="scene.js"></script>'
        assert tag in html, "index.html is missing the scene.js script tag"
        html = html.replace(tag, f"<script>{js}</script>")
        pathlib.Path(args.standalone).write_text(html)
        print(f"Wrote {args.standalone}")


if __name__ == "__main__":
    main()
