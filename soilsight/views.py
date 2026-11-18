"""Turn one 45 MP tray photo into the cached views used for training and the demo.

Outputs per photo:
  overview   soil area resized so the long side is 1536 px   (~0.8 MB)
  crops      N full-resolution 768x768 crops from inside the soil area (~0.26 MB each)
  halfres    whole photo at half resolution, 4096x2732          (~5.2 MB, separate shards)
  meta       crop boxes, soil bounding box, soil coverage, QC flags

Finding the soil: the tray is light and smooth, soil is textured and usually darker.
A pixel counts as tray if it is close in colour to the photo border AND locally smooth,
and is connected to the border. Everything else is soil. If the result looks implausible,
we fall back to a conservative inner box and flag the photo for review.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

WORK_SCALE = 1 / 8          # mask is computed on a 1024x683 copy
CROP = 768                  # full-res crop side (~30 mm at ~39.8 µm/px)
OVERVIEW_LONG = 1536
FALLBACK_INSET = 0.08       # inner box used when the mask is implausible
MIN_IN_SOIL = 0.95          # a crop must be at least this much soil


def soil_mask(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (mask at WORK_SCALE, info). mask: uint8 {0,1}, 1 = soil."""
    sm = cv2.resize(img_bgr, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    h, w = sm.shape[:2]
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = 6
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    tray_col = np.median(border, axis=0)
    dist = np.linalg.norm(lab - tray_col, axis=2)

    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - mean * mean, 0))

    tray_like = ((dist < 22) & (std < 9)).astype(np.uint8)
    tray_like = cv2.morphologyEx(tray_like, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # keep only tray-like regions that touch the photo border
    n, lbl = cv2.connectedComponents(tray_like)
    edge_ids = set(np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))) - {0}
    tray = np.isin(lbl, list(edge_ids)).astype(np.uint8)

    soil = (1 - tray).astype(np.uint8)
    soil = cv2.morphologyEx(soil, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    soil = cv2.morphologyEx(soil, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    # largest component, holes filled
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(soil)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        soil = (lbl == k).astype(np.uint8)
        cnts, _ = cv2.findContours(soil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        soil = cv2.drawContours(np.zeros_like(soil), cnts, -1, 1, thickness=cv2.FILLED)
    # the outermost band of every photo is tray wall / rim, never usable soil
    m = int(0.03 * min(h, w))
    soil[:m] = 0; soil[-m:] = 0; soil[:, :m] = 0; soil[:, -m:] = 0
    # stay off the rim: erode by ~2% of the short side
    e = max(3, int(0.02 * min(h, w)))
    soil = cv2.erode(soil, np.ones((e, e), np.uint8))

    frac = float(soil.mean())
    info = {"soil_frac": round(frac, 4), "tray_lab": [round(float(v), 1) for v in tray_col],
            "fallback": False, "qc_flags": []}
    if not (0.10 <= frac <= 0.97):
        info["qc_flags"].append(f"mask_implausible_{frac:.2f}")
        info["fallback"] = True
        soil = np.zeros((h, w), np.uint8)
        y0, x0 = int(FALLBACK_INSET * h), int(FALLBACK_INSET * w)
        soil[y0:h - y0, x0:w - x0] = 1
    if frac < 0.35 and not info["fallback"]:
        info["qc_flags"].append("low_coverage")
    return soil, info


def _seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def crop_boxes(mask: np.ndarray, full_hw: tuple[int, int], n: int, name: str) -> list[list[int]]:
    """Pick n full-res CROPxCROP boxes, spread over a grid, each >= MIN_IN_SOIL soil."""
    H, W = full_hw
    s = WORK_SCALE
    c = int(round(CROP * s))                      # crop side at mask scale
    ii = cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_32S)
    h, w = mask.shape

    def soil_in(y, x):
        return (ii[y + c, x + c] - ii[y, x + c] - ii[y + c, x] + ii[y, x]) / (c * c)

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    y0, y1, x0, x1 = ys.min(), ys.max() - c, xs.min(), xs.max() - c
    if y1 < y0 or x1 < x0:
        return []
    rng = np.random.default_rng(_seed(name))
    rows = 3 if n >= 9 else 2
    cols = int(np.ceil(n / rows))
    boxes: list[list[int]] = []

    def overlaps(X, Y, max_frac=0.10):
        for bx0, by0, _, _ in boxes:
            ox = max(0, CROP - abs(X - bx0)); oy = max(0, CROP - abs(Y - by0))
            if ox * oy > max_frac * CROP * CROP:
                return True
        return False

    def try_add(y, x, max_frac):
        if soil_in(y, x) < MIN_IN_SOIL:
            return False
        Y = min(int(round(y / s)), H - CROP); X = min(int(round(x / s)), W - CROP)
        if overlaps(X, Y, max_frac):
            return False
        boxes.append([X, Y, X + CROP, Y + CROP])
        return True

    # one crop per grid cell, so crops cover the whole tray
    for r in range(rows):
        for q in range(cols):
            if len(boxes) == n:
                break
            cy0 = y0 + (y1 - y0) * r // rows
            cy1 = y0 + (y1 - y0) * (r + 1) // rows
            cx0 = x0 + (x1 - x0) * q // cols
            cx1 = x0 + (x1 - x0) * (q + 1) // cols
            for _ in range(80):
                y = int(rng.integers(cy0, max(cy0 + 1, cy1)))
                x = int(rng.integers(cx0, max(cx0 + 1, cx1)))
                if try_add(y, x, 0.10):
                    break
    # top up anywhere in the soil, allowing more overlap only if space is tight
    for max_frac in (0.10, 0.35, 1.0):
        tries = 0
        while len(boxes) < n and tries < 1500:
            tries += 1
            try_add(int(rng.integers(y0, y1 + 1)), int(rng.integers(x0, x1 + 1)), max_frac)
    return boxes


def make_views(img_bgr: np.ndarray, name: str, n_crops: int = 12,
               jpeg_q: int = 90, with_halfres: bool = True) -> dict:
    """Return {'overview': bytes, 'crops': [bytes], 'halfres': bytes|None, 'meta': dict}."""
    H, W = img_bgr.shape[:2]
    mask, info = soil_mask(img_bgr)
    ys, xs = np.nonzero(mask)
    s = WORK_SCALE
    bbox = [int(xs.min() / s), int(ys.min() / s), int(min(W, (xs.max() + 1) / s)), int(min(H, (ys.max() + 1) / s))]
    X0, Y0, X1, Y1 = bbox
    roi = img_bgr[Y0:Y1, X0:X1]
    f = OVERVIEW_LONG / max(roi.shape[:2])
    ov = cv2.resize(roi, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    enc = lambda a, q=jpeg_q: cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()

    boxes = crop_boxes(mask, (H, W), n_crops, name)
    if len(boxes) < n_crops:
        info["qc_flags"].append(f"only_{len(boxes)}_crops")
    crops = [enc(img_bgr[y0:y1, x0:x1], 92) for x0, y0, x1, y1 in boxes]
    half = enc(cv2.resize(img_bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)) if with_halfres else None
    mask_png = cv2.imencode(".png", mask * 255)[1].tobytes()
    meta = {"name": name, "full_hw": [H, W], "soil_bbox_xyxy": bbox, "overview_scale": round(f, 6),
            "crop_boxes_xyxy": boxes, "crop_size": CROP, **info}
    return {"overview": enc(ov), "crops": crops, "halfres": half, "mask_png": mask_png, "meta": meta}
