"""Unit tests on synthetic data: no network, GPU or dataset needed."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soilsight import features as F  # noqa: E402
from soilsight import heads as H  # noqa: E402
from soilsight.predict import assess  # noqa: E402
from soilsight.scale import sample_scale  # noqa: E402
from soilsight.views import CROP, make_views, soil_mask  # noqa: E402

RNG = np.random.default_rng(0)


def tray(h=2732, w=4096, soil=(0.15, 0.85, 0.12, 0.88)):
    """Light, smooth tray with a dark, textured soil rectangle (fractions of height/width)."""
    img = np.full((h, w, 3), 205, np.uint8) + RNG.integers(0, 4, (h, w, 3), dtype=np.uint8)
    y0, y1, x0, x1 = int(soil[0] * h), int(soil[1] * h), int(soil[2] * w), int(soil[3] * w)
    tex = cv2.resize(RNG.normal(110, 35, ((y1 - y0) // 6, (x1 - x0) // 6)), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    img[y0:y1, x0:x1] = np.clip(tex, 0, 255).astype(np.uint8)[..., None]
    return img, (x0, y0, x1, y1)


# ---------------------------------------------------------------- views
def test_soil_mask_and_crops_stay_on_soil():
    img, (x0, y0, x1, y1) = tray()
    v = make_views(img, "t", n_crops=12, with_halfres=False)
    m = v["meta"]
    assert not m["fallback"] and not m["qc_flags"]
    assert len(m["crop_boxes_xyxy"]) == 12
    for a, b, c, d in m["crop_boxes_xyxy"]:
        assert c - a == CROP and d - b == CROP
        assert a >= x0 - 16 and b >= y0 - 16 and c <= x1 + 16 and d <= y1 + 16, "crop leaves the soil area"


def test_empty_tray_is_flagged_not_crashing():
    img = np.full((2732, 4096, 3), 205, np.uint8)
    mask, info = soil_mask(img)
    assert info["fallback"] and info["qc_flags"]
    assert mask.sum() > 0


def test_crops_are_deterministic():
    img, _ = tray()
    a = make_views(img, "same-name", with_halfres=False)["meta"]["crop_boxes_xyxy"]
    b = make_views(img, "same-name", with_halfres=False)["meta"]["crop_boxes_xyxy"]
    assert a == b


# ---------------------------------------------------------------- scale
def test_aruco_scale_is_recovered():
    px = 1000                                   # a 50 mm marker drawn 1000 px wide -> 50 µm/px
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 1, px)
    canvas = np.full((3000, 4000), 255, np.uint8)
    canvas[800:800 + px, 1200:1200 + px] = marker
    s = sample_scale(canvas)
    assert s["markers_found"] == [1]
    assert s["um_per_px_floor"] == pytest.approx(50.0, rel=0.01)


# ---------------------------------------------------------------- features
def test_handcrafted_features_shape_and_finite():
    img, _ = tray(768, 768, (0, 1, 0, 1))
    f = F.handcrafted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    assert f.shape == (21,) and np.isfinite(f).all()


def test_overview_resized_to_multiple_of_16():
    ov = F.resize_overview(np.zeros((1024, 1536, 3), np.uint8))
    assert max(ov.shape[:2]) == F.OVW_LONG and ov.shape[0] % 16 == 0 and ov.shape[1] % 16 == 0


# ---------------------------------------------------------------- heads
def synthetic_photos(n_samples=60, photos=3, d=12):
    Y = RNG.dirichlet([2, 3, 2], n_samples)
    W = RNG.normal(0, 1, (3, d))
    rows, X = [], []
    for i in range(n_samples):
        for p in range(photos):
            X.append(Y[i] @ W + RNG.normal(0, 0.05, d))
            rows.append({"key": f"s{i}_p{p}", "sample_id": f"s{i:03d}", "split": "dev", "fold": i % 5,
                         "gravel": Y[i, 0], "sand": Y[i, 1], "fines": Y[i, 2], "type": "bulk"})
    df = pd.DataFrame(rows)
    df["w"] = 1.0 / df.groupby("sample_id").key.transform("size")
    return np.array(X), df


def test_ridge_predictions_are_valid_fractions_and_learn():
    X, df = synthetic_photos()
    h = H.RidgeHead.fit(X, df[H.FRACTIONS].values, df.w.values, alpha=1.0)
    P = h.predict(X)
    assert np.allclose(P.sum(1), 1) and (P >= 0).all()
    assert np.abs(P - df[H.FRACTIONS].values).mean() < 0.05


def test_cv_never_trains_on_the_held_out_fold():
    # Pure-noise features and random labels: a model can only "predict" a row by memorising it.
    # With 300 features for 60 rows and tiny alpha, ridge memorises its training rows perfectly,
    # so out-of-fold error stays high only if held-out rows were never in training.
    n, d = 60, 300
    Y = RNG.dirichlet([1, 1, 1], n)
    df = pd.DataFrame({"key": [f"k{i}" for i in range(n)], "sample_id": [f"s{i:03d}" for i in range(n)],
                       "split": "dev", "fold": np.arange(n) % 5, "gravel": Y[:, 0], "sand": Y[:, 1],
                       "fines": Y[:, 2], "w": 1.0})
    X = RNG.normal(0, 1, (n, d))
    oof = H.cv_oof(X, df, alpha=1e-6)
    fit_in = H.RidgeHead.fit(X, Y, df.w.values, 1e-6).predict(X)
    assert np.abs(fit_in - Y).mean() < 0.01                        # memorises what it trains on...
    assert np.abs(oof - Y).mean() > 0.1, "out-of-fold predictions look memorised: fold leakage"


def test_simplex_clips_and_renormalises():
    P = H.simplex(np.array([[-0.2, 0.7, 0.5], [0.0, 0.0, 0.0], [0.3, 0.3, 0.3]]))
    assert (P >= 0).all() and np.allclose(P.sum(1), 1)
    assert np.allclose(P[0], [0, 0.7 / 1.2, 0.5 / 1.2]) and np.allclose(P[1], 1 / 3)
    X, df = synthetic_photos()
    h = H.RidgeHead.fit(X, df[H.FRACTIONS].values, df.w.values, alpha=1.0)
    far = X[:5] * 40                                               # extrapolation gives negative raw outputs
    assert (h.raw(far) < 0).any() and (h.predict(far) >= 0).all()


def test_heatmap_patches_average_to_photo_prediction():
    X, df = synthetic_photos()
    h = H.RidgeHead.fit(X, df[H.FRACTIONS].values, df.w.values, alpha=1.0)
    W, b = h.unscaled()
    patches = RNG.normal(0, 1, (300, X.shape[1]))
    assert np.allclose((patches @ W + b).mean(0), patches.mean(0) @ W + b)
    assert np.allclose(h.raw(patches), patches @ W + b)


def test_normalised_conformal_reaches_target_coverage():
    n = 4000
    d = RNG.uniform(0.01, 0.1, (n, 3))
    err = np.abs(RNG.normal(0, d))
    cp = H.norm_conformal_fit(err[: n // 2], d[: n // 2], 0.9)
    half = H.norm_half_pp(d[n // 2:], cp) / 100
    cov = (err[n // 2:] <= half).mean(0)
    assert np.all((cov > 0.87) & (cov < 0.93))


def test_knn_stats_finds_nearest():
    E = np.eye(4)
    Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, 0.3, 0.4]])
    mu, sd, dist = H.knn_stats(E[:1], E, Y, k=1)
    assert np.allclose(mu[0], Y[0]) and np.allclose(sd[0], 0) and dist[0] == pytest.approx(0)


# ---------------------------------------------------------------- assess (app + showcase logic)
def test_assess_ranges_flags_and_neighbours():
    E = RNG.normal(0, 1, (30, 8)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    idx = {"emb": E, "fractions": RNG.dirichlet([2, 3, 2], 30), "sample_id": np.array([f"{i:04d}" for i in range(30)]),
           "type": np.array(["bulk"] * 30), "sieves": np.zeros((30, 16)), "sieve_names": np.array([f"%_{i}mm" for i in range(16)])}
    cp = {"method": "knn_spread", "k": 10, "q": {f: 2.0 for f in H.FRACTIONS}, "eps": {f: 0.01 for f in H.FRACTIONS},
          "coverage_target": 0.9}
    cfg = {"conformal": cp, "photo_conformal": {**cp, "q": {f: 3.0 for f in H.FRACTIONS}}, "review_pp": 1.0}
    P = np.array([[0.2, 0.5, 0.3], [0.22, 0.48, 0.3]])
    R = E[[3, 3]] + 0.01
    r = assess(P, R, cfg, idx, Path("thumbs"))
    assert set(r["fractions_pct"]) == set(H.FRACTIONS)
    assert r["neighbours"][0]["sample_id"] == "0003"
    assert any("range wider" in f for f in r["flags"])            # review_pp of 1 pp forces the flag
    one = assess(P[:1], R[:1], cfg, idx, Path("thumbs"))
    assert "single-photo" in one["range_note"]
