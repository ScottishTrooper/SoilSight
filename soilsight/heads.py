"""Prediction heads on frozen features, evaluation, conformal intervals, retrieval and heatmap head.

Targets: gravel / sand / fines fractions (sum to 1). Heads are ridge regressions trained on
photo-level rows (each photo weighted 1/n_photos of its sample, so every sample counts equally)
and evaluated at sample level (mean of that sample's photo predictions), with 5-fold CV grouped
by sample on the dev split. Everything here is CPU-only and takes seconds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

FRACTIONS = ["gravel", "sand", "fines"]
ALPHAS = np.logspace(-1, 4, 11)


# ---------------------------------------------------------------- data assembly
def photo_table(feats: dict, samples: pd.DataFrame) -> pd.DataFrame:
    """One row per photo with its sample's labels (fractions in 0..1) and split/fold."""
    idx = pd.DataFrame(feats["index"])
    s = samples.set_index("sample_id")
    for f in FRACTIONS:
        idx[f] = idx.sample_id.map(s[f]) / 100.0
    idx["type"] = idx.sample_id.map(s["type"])
    idx["w"] = 1.0 / idx.groupby("sample_id").key.transform("size")
    return idx


def feature_sets(feats: dict) -> dict[str, np.ndarray]:
    """Photo-level design matrices to compare. crop+overview concatenated per backbone."""
    bbs = sorted({k.split(".")[0] for k in feats if "." in k and k != "index"} - {"hand"})
    sets = {"handcrafted": np.hstack([feats["hand.crop"], feats["hand.ovw"]])}
    for b in bbs:
        sets[b] = np.hstack([feats[f"{b}.crop"], feats[f"{b}.ovw"]])
    dino = [b for b in bbs if b.startswith("dinov3")]
    if len(dino) > 1:
        sets["dinov3_both"] = np.hstack([sets[b] for b in dino])
    for b in bbs:
        if f"{b}.ptok" in feats:
            sets[f"{b}_patchmean"] = feats[f"{b}.ptok"]   # the heatmap head's input
    return sets


# ---------------------------------------------------------------- ridge head
@dataclass
class RidgeHead:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray      # (d, 3) in standardized space
    intercept: np.ndarray

    @staticmethod
    def fit(X, Y, w, alpha):
        mu = np.average(X, axis=0, weights=w)
        sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=w)) + 1e-6
        Z = (X - mu) / sd
        ym = np.average(Y, axis=0, weights=w)
        sw = np.sqrt(w)[:, None]
        A, B = Z * sw, (Y - ym) * sw
        d = Z.shape[1]
        if d <= A.shape[0]:
            coef = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ B)
        else:  # dual form when there are more features than photos
            coef = A.T @ np.linalg.solve(A @ A.T + alpha * np.eye(A.shape[0]), B)
        return RidgeHead(mu, sd, coef, ym)

    def raw(self, X):
        return ((X - self.mean) / self.scale) @ self.coef + self.intercept

    def predict(self, X):
        return simplex(self.raw(X))

    def unscaled(self):
        """Weights that apply directly to raw features: y = X @ W + b (used per patch for the heatmap)."""
        W = self.coef / self.scale[:, None]
        return W, self.intercept - self.mean @ W

    def to_npz(self):
        return {"mean": self.mean, "scale": self.scale, "coef": self.coef, "intercept": self.intercept}


def simplex(P):
    P = np.clip(P, 0, None)
    s = P.sum(1, keepdims=True)
    return np.where(s > 0, P / np.maximum(s, 1e-9), 1.0 / P.shape[1])


def per_sample(df: pd.DataFrame, P: np.ndarray):
    """Average photo predictions per sample; also return the spread across photos (difficulty score)."""
    t = pd.DataFrame(P, columns=FRACTIONS)
    t["sample_id"] = df.sample_id.values
    g = t.groupby("sample_id")
    return g[FRACTIONS].mean(), g[FRACTIONS].std().fillna(0.0)


def sample_truth(df):
    return df.groupby("sample_id")[FRACTIONS + ["type"]].first()


def mae_pp(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    e = (pred[FRACTIONS] - truth.loc[pred.index, FRACTIONS]).abs() * 100
    out = {f"mae_{f}": float(e[f].mean()) for f in FRACTIONS}
    out["mae_mean"] = float(e.values.mean())
    nat = truth.loc[pred.index, "type"] != "artificial"
    out["mae_mean_natural"] = float(e[nat.values].values.mean())
    return out


# ---------------------------------------------------------------- cross-validation
def cv_oof(X, df, alpha, folds=range(5)):
    """Out-of-fold photo predictions on dev for one alpha."""
    dev = (df.split == "dev").values
    P = np.full((len(df), 3), np.nan)
    Y = df[FRACTIONS].values
    for k in folds:
        tr = dev & (df.fold != k).values
        te = dev & (df.fold == k).values
        h = RidgeHead.fit(X[tr], Y[tr], df.w.values[tr], alpha)
        P[te] = h.predict(X[te])
    return P


def compare(sets: dict, df: pd.DataFrame, log=print) -> tuple[pd.DataFrame, dict]:
    """CV every feature set over the alpha grid. Returns a results table and the best alpha per set."""
    dev = (df.split == "dev").values
    truth = sample_truth(df[dev])
    rows, best = [], {}
    # baseline: always predict the dev mean (per fold)
    P = np.full((len(df), 3), np.nan)
    for k in range(5):
        tr = dev & (df.fold != k).values
        P[dev & (df.fold == k).values] = np.average(df[FRACTIONS].values[tr], axis=0, weights=df.w.values[tr])
    pred, _ = per_sample(df[dev], P[dev])
    rows.append({"features": "predict the mean (baseline)", "alpha": None, **mae_pp(pred, truth)})
    for name, X in sets.items():
        res = []
        for a in ALPHAS:
            P = cv_oof(X, df, a)
            pred, _ = per_sample(df[dev], P[dev])
            res.append((mae_pp(pred, truth), a))
        m, a = min(res, key=lambda r: r[0]["mae_mean"])
        best[name] = float(a)
        rows.append({"features": name, "alpha": float(a), **m})
        log(f"  {name:28s} mean MAE {m['mae_mean']:.2f} pp (alpha={a:.3g})")
    tab = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)
    return tab, best


def knn_predict(E_q, E_ref, Y_ref, k=5):
    """Similarity-weighted average of the k nearest reference samples (cosine)."""
    S = E_q @ E_ref.T
    nn = np.argsort(-S, axis=1)[:, :k]
    w = np.take_along_axis(S, nn, 1).clip(1e-6)
    return (Y_ref[nn] * w[..., None]).sum(1) / w.sum(1, keepdims=True), nn


def sample_embeddings(X, df):
    """L2-normalised mean photo embedding per sample (for retrieval)."""
    t = pd.DataFrame(X)
    t["sample_id"] = df.sample_id.values
    E = t.groupby("sample_id").mean()
    v = E.values
    v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
    return pd.DataFrame(v, index=E.index)


def knn_cv(X, df, k=5) -> dict:
    """Retrieval-as-prediction, evaluated with the same folds (shows the neighbours are meaningful)."""
    dev = df[df.split == "dev"]
    truth = sample_truth(dev)
    E = sample_embeddings(X[(df.split == "dev").values], dev)
    fold = dev.groupby("sample_id").fold.first().loc[E.index]
    preds = []
    for f in range(5):
        q, r = E[fold == f], E[fold != f]
        P, _ = knn_predict(q.values, r.values, truth.loc[r.index, FRACTIONS].values, k)
        preds.append(pd.DataFrame(P, index=q.index, columns=FRACTIONS))
    return mae_pp(pd.concat(preds), truth)


# ---------------------------------------------------------------- conformal intervals
def conformal_fit(pred, spread, truth, coverage=0.9, eps=None):
    """Normalised split conformal: interval = pred ± q * (spread + eps), per fraction.

    spread = std of the photo-level predictions within the sample: samples whose photos
    disagree get wider intervals. eps keeps intervals sane when all photos agree.
    """
    eps = eps if eps is not None else {f: float(np.median(spread[f]) + 1e-3) for f in FRACTIONS}
    n = len(pred)
    lvl = min(1.0, np.ceil((n + 1) * coverage) / n)
    q = {}
    for f in FRACTIONS:
        s = (truth.loc[pred.index, f] - pred[f]).abs() / (spread[f] + eps[f])
        q[f] = float(np.quantile(s, lvl, method="higher"))
    return {"coverage_target": coverage, "q": q, "eps": eps, "n_cal": n}


def conformal_apply(pred, spread, cp, review_pp=10.0):
    half = pd.DataFrame({f: cp["q"][f] * (spread[f] + cp["eps"][f]) * 100 for f in FRACTIONS})
    review = (half > review_pp).any(axis=1)
    return half, review


def conformal_report(pred, spread, truth, cp, review_pp=10.0) -> dict:
    half, review = conformal_apply(pred, spread, cp, review_pp)
    err = (pred[FRACTIONS] - truth.loc[pred.index, FRACTIONS]).abs() * 100
    covered = (err <= half).all(axis=1)
    acc = ~review
    return {
        "n": int(len(pred)),
        "joint_coverage": float(covered.mean()),
        "per_fraction_coverage": {f: float((err[f] <= half[f]).mean()) for f in FRACTIONS},
        "median_half_width_pp": {f: float(half[f].median()) for f in FRACTIONS},
        "share_auto_accepted": float(acc.mean()),
        "mae_mean_accepted": float(err[acc.values].values.mean()) if acc.any() else None,
        "mae_mean_flagged": float(err[review.values].values.mean()) if review.any() else None,
    }


def to_json(o) -> str:
    return json.dumps(o, indent=2, default=float)


# ---------------------------------------------------------------- uncertainty v2: difficulty signals
# A normalised conformal range is pred ± q * (difficulty + eps). It only helps if the difficulty signal
# actually tracks the error, so several are compared on out-of-fold predictions and the best is kept.
SCORES = ["constant", "photo_spread", "knn_spread", "knn_disagree", "knn_dist"]


def knn_stats(Eq, Eref, Yref, k=10):
    """Similarity-weighted mean and spread of the k nearest reference samples' labels, plus distance."""
    S = Eq @ Eref.T
    k = min(k, Eref.shape[0])
    nn = np.argsort(-S, axis=1)[:, :k]
    s = np.take_along_axis(S, nn, 1)
    w = s.clip(1e-6)[..., None]
    Yn = Yref[nn]
    mu = (Yn * w).sum(1) / w.sum(1)
    sd = np.sqrt(((Yn - mu[:, None]) ** 2 * w).sum(1) / w.sum(1))
    return mu, sd, 1.0 - s.mean(1)


def difficulty(method, pred, photo_spread=None, knn=None):
    """(n, 3) difficulty per fraction for a scoring method."""
    n = len(pred)
    if method == "constant":
        return np.ones((n, 3))
    if method == "photo_spread":
        return np.asarray(photo_spread) if photo_spread is not None else np.ones((n, 3))
    mu, sd, dist = knn
    if method == "knn_spread":
        return sd
    if method == "knn_disagree":
        return np.abs(np.asarray(pred) - mu)
    if method == "knn_dist":
        return np.repeat(dist[:, None], 3, 1)
    raise ValueError(method)


def norm_conformal_fit(err, d, coverage=0.9):
    """err, d: (n, 3) absolute errors (fractions 0..1) and difficulty. Returns q, eps per fraction."""
    eps = np.median(d, axis=0) + 1e-3
    n = len(err)
    lvl = min(1.0, np.ceil((n + 1) * coverage) / n)
    q = np.quantile(err / (d + eps), lvl, axis=0, method="higher")
    return {"q": dict(zip(FRACTIONS, map(float, q))), "eps": dict(zip(FRACTIONS, map(float, eps))), "level": float(lvl)}


def norm_half_pp(d, cp):
    """Half-widths in percentage points, (n, 3)."""
    q = np.array([cp["q"][f] for f in FRACTIONS]); eps = np.array([cp["eps"][f] for f in FRACTIONS])
    return q * (d + eps) * 100


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def selective_table(err_pp, half_pp, shares=(0.25, 0.5, 0.75, 1.0)):
    """Accept the most confident share of samples (narrowest max half-width): error of accepted vs rest."""
    w = half_pp.max(1)
    order = np.argsort(w)
    e = err_pp.mean(1)
    rows = []
    for sh in shares:
        k = max(1, int(round(sh * len(e))))
        acc, rej = order[:k], order[k:]
        rows.append({"accept_share": sh, "max_half_width_pp": float(w[order[k - 1]]),
                     "mae_accepted": float(e[acc].mean()), "mae_flagged": float(e[rej].mean()) if len(rej) else np.nan})
    return pd.DataFrame(rows)
