"""Image scale (micrometres per pixel) from the dataset's calibration photos.

Each calibration photo shows two 50 mm ArUco markers (DICT_4X4_50):
ID 1 lies on the tray floor, ID 2 sits on a raised block. The soil surface lies
between those heights, so we report both and use their mean as the scale.
"""
from __future__ import annotations

import cv2
import numpy as np

MARKER_MM = 50.0
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _detector() -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(_DICT, p)


def marker_scales(gray: np.ndarray, detect_scale: float = 0.25) -> dict[int, float]:
    """Return {marker_id: µm per pixel} at full resolution.

    Detects on a downscaled copy (fast), then refines corners on the full image.
    """
    small = cv2.resize(gray, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
    corners, ids, _ = _detector().detectMarkers(small)
    out: dict[int, float] = {}
    if ids is None:
        return out
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    for c, i in zip(corners, ids.ravel()):
        pts = (c.reshape(4, 2) / detect_scale).astype(np.float32)
        pts = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (11, 11), (-1, -1), crit).reshape(4, 2)
        side_px = np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)])
        out[int(i)] = float(MARKER_MM * 1000.0 / side_px)
    return out


def sample_scale(gray: np.ndarray) -> dict:
    """Scale summary for one calibration photo (keys are safe to store as JSON)."""
    s = marker_scales(gray)
    floor, raised = s.get(1), s.get(2)
    vals = [v for v in (floor, raised) if v is not None]
    return {
        "um_per_px_floor": floor,
        "um_per_px_raised": raised,
        "um_per_px": float(np.mean(vals)) if vals else None,
        "markers_found": sorted(s),
    }
