"""Generate notebooks/02b_uncertainty.ipynb (Kaggle, CPU, ~1 min). Usage: python3 make_uncertainty_notebook.py <repo_root>"""
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ["__init__.py", "features.py", "heads.py"]:
        t.add(root / "soilsight" / f, arcname=f"soilsight/{f}")
payload = base64.b64encode(buf.getvalue()).decode()
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("""# SoilSight · 02b · Uncertainty v2: which signal predicts the error?

The first "review needed" rule used how much a sample's photos disagree. On the calibration samples it flagged 96%
and the accepted ones were *worse*. This notebook compares five difficulty signals against real out-of-fold errors,
keeps the one that tracks error best, calibrates its range, and sets the review threshold so a chosen share of
samples is auto-accepted. CPU only (Accelerator **None**), Internet **On**, `HF_TOKEN` attached. Takes about a minute.
The test split stays untouched."""),
code("""# ---- settings ----
CACHE_REPO_NAME = "photogranulometry-cache"
MODEL_REPO_NAME = "soilsight-models"
K               = 10      # neighbours used by the kNN signals
COVERAGE        = 0.90
TARGET_ACCEPT   = 0.50    # share of samples to auto-accept (the most confident ones)
"""),
code(f"""# ---- unpack ----
import base64, io, tarfile, sys, os, json
PAYLOAD = "{payload}"
WORK = "/kaggle/working/soilsight_src" if os.path.isdir("/kaggle/working") else os.path.abspath("soilsight_src")
os.makedirs(WORK, exist_ok=True)
_t = tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz")
try:
    _t.extractall(WORK, filter="data")
except TypeError:
    _t.extractall(WORK)
sys.path.insert(0, WORK)
import numpy as np, pandas as pd
from pathlib import Path
from soilsight import features as F, heads as H
pd.set_option("display.width", 200)
"""),
code("""# ---- load model config, features, labels ----
from huggingface_hub import HfApi, hf_hub_download
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN and TOKEN.startswith("hf_"), "HF_TOKEN secret not attached"
api = HfApi(token=TOKEN); user = api.whoami()["name"]
CACHE, MODELS = f"{user}/{CACHE_REPO_NAME}", f"{user}/{MODEL_REPO_NAME}"
cfg = json.loads(Path(hf_hub_download(MODELS, "config.json", token=TOKEN)).read_text())
feat_file = sorted(f for f in api.list_repo_files(CACHE, repo_type="dataset") if f.startswith("features/"))[-1]
feats = F.load_features(Path(hf_hub_download(CACHE, feat_file, repo_type="dataset", token=TOKEN)))
samples = pd.read_csv(hf_hub_download(CACHE, "data/split_samples_seed0.csv", repo_type="dataset", token=TOKEN), dtype={"sample_id": str})
df = H.photo_table(feats, samples); sets = H.feature_sets(feats)
FR = H.FRACTIONS
X, a, R = sets[cfg["features"]], cfg["alpha"], sets[cfg["retrieval_features"]]
dev, cal = (df.split == "dev").values, (df.split == "calib").values
print("model:", cfg["features"], "| retrieval features:", cfg["retrieval_features"])
"""),
code("""# ---- out-of-fold predictions and every difficulty signal, at sample level (dev) ----
oof = H.cv_oof(X, df, a)
p_dev, s_dev = H.per_sample(df[dev], oof[dev])
ids = p_dev.index
y_dev = H.sample_truth(df[dev]).loc[ids, FR].values
E_dev = H.sample_embeddings(R[dev], df[dev]).loc[ids].values
fold = df[dev].groupby("sample_id").fold.first().loc[ids].values
mu, sd, dist = np.zeros_like(y_dev), np.zeros_like(y_dev), np.zeros(len(ids))
for f in range(5):
    q = fold == f
    mu[q], sd[q], dist[q] = H.knn_stats(E_dev[q], E_dev[~q], y_dev[~q], K)
err_dev = np.abs(p_dev.values - y_dev)

# calibration split: model trained on dev, neighbours from the dev index (as in the app)
head_dev = H.RidgeHead.fit(X[dev], df[FR].values[dev], df.w.values[dev], a)
p_cal, s_cal = H.per_sample(df[cal], head_dev.predict(X[cal]))
cids = p_cal.index
y_cal = H.sample_truth(df[cal]).loc[cids, FR].values
E_cal = H.sample_embeddings(R[cal], df[cal]).loc[cids].values
knn_cal = H.knn_stats(E_cal, E_dev, y_dev, K)
err_cal = np.abs(p_cal.values - y_cal)

rows = []
for m in H.SCORES:
    d_dev = H.difficulty(m, p_dev.values, s_dev.values, (mu, sd, dist))
    d_cal = H.difficulty(m, p_cal.values, s_cal.values, knn_cal)
    rho = np.nan if m == "constant" else np.mean([H.spearman(d_dev[:, j], err_dev[:, j]) for j in range(3)])
    cp = H.norm_conformal_fit(err_dev, d_dev, COVERAGE)
    h_dev, h_cal = H.norm_half_pp(d_dev, cp), H.norm_half_pp(d_cal, cp)
    thr = float(np.quantile(h_dev.max(1), TARGET_ACCEPT))
    acc = h_cal.max(1) <= thr
    e = err_cal.mean(1) * 100
    rows.append({"signal": m, "rank_corr_with_error_dev": rho,
                 "calib_coverage_joint": float((err_cal * 100 <= h_cal).all(1).mean()),
                 "calib_median_halfwidth_pp": float(np.median(h_cal)),
                 "calib_share_accepted": float(acc.mean()),
                 "calib_mae_accepted": float(e[acc].mean()) if acc.any() else np.nan,
                 "calib_mae_flagged": float(e[~acc].mean()) if (~acc).any() else np.nan})
report = pd.DataFrame(rows)
print(report.round(3).to_string(index=False))
"""),
code("""# ---- pick the signal that tracks error best on dev (calib is only for reporting) ----
cand = report[report.signal != "constant"].sort_values("rank_corr_with_error_dev", ascending=False)
best = cand.iloc[0]
METHOD = best.signal if best.rank_corr_with_error_dev > 0.1 else "constant"
print("chosen signal:", METHOD)

d_dev = H.difficulty(METHOD, p_dev.values, s_dev.values, (mu, sd, dist))
d_cal = H.difficulty(METHOD, p_cal.values, s_cal.values, knn_cal)
cp_dev = H.norm_conformal_fit(err_dev, d_dev, COVERAGE)
REVIEW_PP = float(np.quantile(H.norm_half_pp(d_dev, cp_dev).max(1), TARGET_ACCEPT))
h_cal = H.norm_half_pp(d_cal, cp_dev)
print(f"review threshold: flag if any half-width > {REVIEW_PP:.1f} pp")
print("\\ncalibration split, accepting the most confident samples:")
print(H.selective_table(err_cal * 100, h_cal).round(2).to_string(index=False))

# final calibration on dev (out-of-fold) + calib, for the app
cp = H.norm_conformal_fit(np.vstack([err_dev, err_cal]), np.vstack([d_dev, d_cal]), COVERAGE)

# single-photo range: same signal, calibrated on out-of-fold photo predictions (photo_spread is undefined for 1 photo)
m1 = METHOD if METHOD != "photo_spread" else "constant"
Pdev = oof[dev]; ydp = df[FR].values[dev]
Rp = R[dev] / np.linalg.norm(R[dev], axis=1, keepdims=True).clip(1e-9)
pf = df.fold.values[dev]; s_of = dict(zip(ids, fold))
mu1, sd1, di1 = np.zeros_like(Pdev), np.zeros_like(Pdev), np.zeros(len(Pdev))
for f in range(5):
    q = pf == f
    mu1[q], sd1[q], di1[q] = H.knn_stats(Rp[q], E_dev[fold != f], y_dev[fold != f], K)
cp1 = H.norm_conformal_fit(np.abs(Pdev - ydp), H.difficulty(m1, Pdev, None, (mu1, sd1, di1)), COVERAGE)
print("\\nsingle-photo signal:", m1, "| q:", {k: round(v, 2) for k, v in cp1["q"].items()})
"""),
code("""# ---- save to the model repo (config.json only) ----
cfg["conformal"] = {**cp, "method": METHOD, "k": K, "coverage_target": COVERAGE,
                    "n_cal": int(len(err_dev) + len(err_cal))}
cfg["photo_conformal"] = {**cp1, "method": m1, "k": K, "coverage_target": COVERAGE}
cfg["review_pp"] = REVIEW_PP
cfg["target_accept"] = TARGET_ACCEPT
cfg["uncertainty_report"] = report.round(4).to_dict(orient="records")
sel = H.selective_table(err_cal * 100, h_cal)
cfg["calib_report"] = {**cfg.get("calib_report", {}), "conformal_v2": {
    "method": METHOD, "n": int(len(err_cal)),
    "joint_coverage": float((err_cal * 100 <= h_cal).all(1).mean()),
    "share_auto_accepted": float((h_cal.max(1) <= REVIEW_PP).mean()),
    "mae_mean_accepted": float((err_cal.mean(1) * 100)[h_cal.max(1) <= REVIEW_PP].mean()) if (h_cal.max(1) <= REVIEW_PP).any() else None,
    "mae_mean_flagged": float((err_cal.mean(1) * 100)[h_cal.max(1) > REVIEW_PP].mean()) if (h_cal.max(1) > REVIEW_PP).any() else None,
    "selective": sel.round(3).to_dict(orient="records")}}
api.upload_file(path_or_fileobj=H.to_json(cfg).encode(), path_in_repo="config.json", repo_id=MODELS,
                repo_type="model", commit_message=f"Uncertainty v2: {METHOD}")
print("config.json updated in", MODELS)
print(H.to_json(cfg["calib_report"]["conformal_v2"]))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = root / "notebooks" / "02b_uncertainty.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out, round(out.stat().st_size / 1e3), "kB")
