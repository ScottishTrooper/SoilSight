"""Inference for the demo app: photos of one sample in -> estimate, range, flags, evidence, heatmap out.

Reproduces the training pipeline exactly: same soil mask and crop placement (views.py), same JPEG
round-trip as the cache, same frozen backbones and feature layout (features.py / heads.feature_sets).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import features as F
from . import heads as H
from .views import CROP, OVERVIEW_LONG, crop_boxes, soil_mask

EXPECTED_MIN_WIDTH = 6000      # training photos are 8192 px wide at ~39.8 µm/px


def _jpeg_roundtrip(bgr, q):
    return cv2.imdecode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def backbones_for(names: list[str], available: list[str]) -> set[str]:
    """Which backbones a list of feature-set names needs."""
    need = set()
    for n in names:
        if n == "handcrafted":
            continue
        if n == "dinov3_both":
            need |= {b for b in available if b.startswith("dinov3")}
        else:
            need.add(n[: -len("_patchmean")] if n.endswith("_patchmean") else n)
    return need


class Predictor:
    def __init__(self, art_dir: str | Path, device: str = "cpu"):
        d = Path(art_dir)
        self.cfg = json.loads((d / "config.json").read_text())
        h = np.load(d / "head.npz")
        self.head = H.RidgeHead(h["mean"], h["scale"], h["coef"], h["intercept"])
        hm = np.load(d / "heatmap_head.npz")
        self.hm_W, self.hm_b = hm["W"], hm["b"]
        r = np.load(d / "retrieval_index.npz")
        self.idx = {k: r[k] for k in r.files}
        self.thumbs = d / "thumbs"
        self.best, self.ret, self.hm = self.cfg["features"], self.cfg["retrieval_features"], self.cfg["heatmap_features"]
        need = backbones_for([self.best, self.ret, self.hm], self.cfg["backbones"])
        self.models = {b: F.Backbone(b, device) for b in sorted(need)}
        self.hm_model = self.models[backbones_for([self.hm], self.cfg["backbones"]).pop()]

    # ------------------------------------------------------------ per photo
    def photo_views(self, rgb: np.ndarray, name: str = "upload"):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        H_, W_ = bgr.shape[:2]
        mask, info = soil_mask(bgr)
        qc = list(info["qc_flags"])
        if W_ < EXPECTED_MIN_WIDTH:
            qc.append(f"small image ({W_} px wide): the model expects full-resolution lab photos (~8192 px, ~40 µm/px)")
        ys, xs = np.nonzero(mask)
        s = 1 / 8
        X0, Y0 = int(xs.min() / s), int(ys.min() / s)
        X1, Y1 = int(min(W_, (xs.max() + 1) / s)), int(min(H_, (ys.max() + 1) / s))
        roi = bgr[Y0:Y1, X0:X1]
        f = OVERVIEW_LONG / max(roi.shape[:2])
        ov = _jpeg_roundtrip(cv2.resize(roi, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), 90)
        boxes = crop_boxes(mask, (H_, W_), self.cfg.get("n_crops", 12), name)
        if not boxes:
            boxes = [[max(0, W_ // 2 - CROP // 2), max(0, H_ // 2 - CROP // 2), 0, 0]]
            boxes[0][2:] = [boxes[0][0] + CROP, boxes[0][1] + CROP]
            qc.append("no clean soil area found for crops; using the image centre")
        crops = [_jpeg_roundtrip(bgr[y0:y1, x0:x1], 92) for x0, y0, x1, y1 in boxes]
        g = cv2.cvtColor(cv2.resize(roi, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        bright = float(g.mean())
        if sharp < 15:
            qc.append(f"looks blurry (sharpness {sharp:.0f})")
        if bright < 40 or bright > 235:
            qc.append(f"exposure looks off (mean brightness {bright:.0f}/255)")
        rgbify = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        return rgbify(ov), [rgbify(c) for c in crops], {"qc": qc, "soil_frac": info["soil_frac"], "boxes": boxes,
                                                       "sharpness": sharp, "brightness": bright}

    def photo_features(self, ov_rgb, crops_rgb) -> dict:
        ov = F.resize_overview(ov_rgb)
        out = {}
        for b, m in self.models.items():
            pc, _ = m.embed(crops_rgb)
            po, pto = m.embed([ov])
            out[f"{b}.crop"], out[f"{b}.ovw"] = pc.mean(0), po[0]
            if pto is not None and getattr(m, "is_vit", False):
                out[f"{b}.ptok"] = pto[0]
        out["hand.crop"] = np.mean([F.handcrafted(c) for c in crops_rgb], 0)
        out["hand.ovw"] = F.handcrafted(ov)
        return out

    # ------------------------------------------------------------ per sample
    def predict(self, photos_rgb: list[np.ndarray], k: int = 5) -> dict:
        views, feats = [], []
        for i, p in enumerate(photos_rgb):
            ov, crops, info = self.photo_views(p, f"upload_{i}")
            views.append((ov, crops, info))
            feats.append(self.photo_features(ov, crops))
        stacked = {key: np.stack([f[key] for f in feats]) for key in feats[0]}
        sets = H.feature_sets(stacked)
        P = self.head.predict(sets[self.best])                 # per photo
        r = assess(P, sets[self.ret], self.cfg, self.idx, self.thumbs, k)
        qc = [f"photo {i + 1}: {m}" for i, v in enumerate(views) for m in v[2]["qc"]]
        return {**r, "qc": qc, "review_needed": bool(r["flags"] or qc),
                "overview": views[0][0], "model": self.cfg.get("model_version", self.best)}

    def heatmap(self, overview_rgb: np.ndarray, fraction: str = "gravel", alpha: float = 0.45):
        """Overlay of per-patch estimates for one fraction. Returns (RGB image, per-patch map in %)."""
        ov = F.resize_overview(overview_rgb)
        T = self.hm_model.patch_tokens(ov)
        P = H.simplex((T.reshape(-1, T.shape[-1]) @ self.hm_W + self.hm_b)).reshape(T.shape[0], T.shape[1], 3)
        m = P[..., H.FRACTIONS.index(fraction)] * 100
        up = cv2.resize(m.astype(np.float32), (ov.shape[1], ov.shape[0]), interpolation=cv2.INTER_CUBIC)
        col = cv2.applyColorMap(np.clip(up * 2.55, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        over = cv2.addWeighted(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), 1 - alpha, col, alpha, 0)
        return cv2.cvtColor(over, cv2.COLOR_BGR2RGB), m


def assess(P: np.ndarray, R_rows: np.ndarray, cfg: dict, idx: dict, thumbs: Path, k: int = 5) -> dict:
    """Sample-level estimate, calibrated range, flags and evidence from per-photo predictions.

    P: (n_photos, 3) head predictions (fractions 0..1); R_rows: (n_photos, d) retrieval features.
    Shared by the live app and the precomputed showcase so both apply identical rules.
    """
    pred = P.mean(0)
    n = len(P)
    cp, rpp = cfg["conformal"], cfg["review_pp"]
    Eidx, Yidx = idx["emb"], idx["fractions"]
    unit = lambda v: v / max(np.linalg.norm(v), 1e-9)
    if n >= 2:
        knn = H.knn_stats(unit(R_rows.mean(0))[None], Eidx, Yidx, cp.get("k", 10))
        if "method" in cp:   # v2: normalised by the chosen difficulty signal
            half = H.norm_half_pp(H.difficulty(cp["method"], pred[None], P.std(0, ddof=1)[None], knn), cp)[0]
        else:                # v1 config: photo spread
            spread = P.std(0, ddof=1)
            half = np.array([cp["q"][f] * (spread[j] + cp["eps"][f]) for j, f in enumerate(H.FRACTIONS)]) * 100
        range_note = f"calibrated for {cp['coverage_target']:.0%} coverage per fraction, from {n} photos"
    else:
        pc = cfg.get("photo_conformal")
        if pc and "method" in pc:
            knn = H.knn_stats(unit(R_rows[0])[None], Eidx, Yidx, pc.get("k", 10))
            half = H.norm_half_pp(H.difficulty(pc["method"], P, None, knn), pc)[0]
        elif pc:
            half = np.array([pc["q_pp"][f] for f in H.FRACTIONS])
        else:
            half = np.full(3, np.nan)
        range_note = "single-photo range (wider); add more photos of the same sample for a tighter range"
    flags = []
    if np.any(half > rpp):
        flags.append(f"range wider than ±{rpp:.0f} pp")
    sims = Eidx @ unit(R_rows.mean(0))
    nn = np.argsort(-sims)[:k]
    thr = cfg.get("ood_similarity")
    if thr is not None and sims[nn[0]] < thr:
        flags.append("unusual photo: unlike anything in the training data")
    neighbours = [{"sample_id": str(idx["sample_id"][j]), "type": str(idx["type"][j]),
                   "similarity": float(sims[j]), "fractions": (idx["fractions"][j] * 100).tolist(),
                   "sieves": idx["sieves"][j].tolist(),
                   "thumb": str(Path(thumbs) / f"{idx['sample_id'][j]}.jpg")} for j in nn]
    return {"fractions_pct": dict(zip(H.FRACTIONS, (pred * 100).round(1).tolist())),
            "half_width_pp": dict(zip(H.FRACTIONS, np.round(half, 1).tolist())),
            "per_photo_pct": (P * 100).round(1).tolist(), "flags": flags, "range_note": range_note,
            "neighbours": neighbours, "sieve_names": [str(x) for x in idx["sieve_names"]]}
