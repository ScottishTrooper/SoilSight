"""Generate notebooks/02_train.ipynb (Kaggle, GPU). Usage: python3 make_train_notebook.py <repo_root>"""
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ["soilsight/__init__.py", "soilsight/features.py", "soilsight/heads.py", "data/sample_psd.csv"]:
        t.add(root / f, arcname=f)
payload = base64.b64encode(buf.getvalue()).decode()

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("""# SoilSight · 02 · Features, model, uncertainty, retrieval, heatmap

Needs the cache from notebook 01. Runs frozen backbones once on the GPU (EfficientNet-B0, DINOv3 ConvNeXt-T,
DINOv3 ViT-S/16 + handcrafted colour/texture features), then everything else on CPU in seconds:

1. **Comparison table**: 5-fold cross-validation on the dev split, error per fraction in percentage points.
2. **Uncertainty**: conformal intervals; samples with wide intervals are flagged "review needed".
3. **Similar-sample index** for the "show me the evidence" feature.
4. **Heatmap head**: a linear head on ViT patch tokens, so every patch gets its own grain-size estimate.

Artifacts go to a private Hugging Face model repo, ready for the demo app.

**Settings (right panel):** Accelerator **GPU P100** (or T4), Internet **On**, secret `HF_TOKEN` attached.
Use **Save Version → Save & Run All** so it runs in the background (expect roughly 30–90 min, mostly feature extraction).
The test split stays locked (`RUN_TEST = False`) until the model is final."""),
code("""# ---- settings ----
CACHE_REPO_NAME = "photogranulometry-cache"   # dataset from notebook 01
MODEL_REPO_NAME = "soilsight-models"          # created private
BACKBONE_LIST   = ["effnet_b0", "dinov3_convnext_t", "dinov3_vits16"]
FORCE_EXTRACT   = False   # reuse features already on the Hub if present
COVERAGE        = 0.90    # target interval coverage
REVIEW_PP       = 10.0    # flag a sample if any interval half-width exceeds this many percentage points
RUN_TEST        = False   # flip to True ONCE, after choosing the final model
"""),
code(f"""# ---- unpack code ----
import base64, io, tarfile, sys, os, json, time
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
from soilsight import features as F, heads as H
pd.set_option("display.width", 160)
"""),
code("""# ---- Hugging Face ----
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, login
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN and TOKEN.startswith("hf_"), "HF_TOKEN secret not attached"
login(token=TOKEN, add_to_git_credential=False)
api = HfApi(token=TOKEN)
user = api.whoami()["name"]
CACHE = f"{user}/{CACHE_REPO_NAME}"
MODELS = f"{user}/{MODEL_REPO_NAME}"
api.create_repo(MODELS, repo_type="model", private=True, exist_ok=True)
cache_files = set(api.list_repo_files(CACHE, repo_type="dataset"))
n_batches = sum(f.startswith("meta/batch_") for f in cache_files)
print(f"cache: {CACHE} ({n_batches} batches) | models: {MODELS}")
if n_batches < 32:
    print("WARNING: the cache is incomplete; results will use only the batches present.")
samples = pd.read_csv(hf_hub_download(CACHE, "data/split_samples_seed0.csv", repo_type="dataset", token=TOKEN),
                      dtype={"sample_id": str})
"""),
code("""# ---- features: reuse from the Hub, or extract on the GPU (the only slow step) ----
import torch
from pathlib import Path
FEAT_NAME = f"features/features_{'-'.join(BACKBONE_LIST)}_b{n_batches}.npz"
LOCAL = Path("/kaggle/tmp_train" if os.path.isdir("/kaggle") else "tmp_train"); LOCAL.mkdir(exist_ok=True)
if FEAT_NAME in cache_files and not FORCE_EXTRACT:
    feats = F.load_features(Path(hf_hub_download(CACHE, FEAT_NAME, repo_type="dataset", token=TOKEN)))
    print("loaded features from the Hub:", FEAT_NAME)
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "(no GPU: this will be slow)")
    t0 = time.time()
    d = snapshot_download(CACHE, repo_type="dataset", allow_patterns=["views/*"], local_dir=LOCAL / "cache", token=TOKEN)
    tars = sorted((LOCAL / "cache" / "views").glob("batch_*.tar"))
    print(f"downloaded {len(tars)} view shards in {time.time()-t0:.0f}s")
    t0 = time.time()
    feats = F.extract(tars, BACKBONE_LIST, device=device)
    print(f"extraction took {(time.time()-t0)/60:.1f} min")
    F.save_features(feats, LOCAL / "features.npz")
    api.upload_file(path_or_fileobj=str(LOCAL / "features.npz"), path_in_repo=FEAT_NAME, repo_id=CACHE,
                    repo_type="dataset", commit_message="Add frozen features")
    print("features saved to the Hub:", FEAT_NAME)
df = H.photo_table(feats, samples)
sets = H.feature_sets(feats)
print(df.groupby("split").agg(photos=("key", "size"), samples=("sample_id", "nunique")))
"""),
code("""# ---- 1. comparison table: 5-fold CV on dev, sample-level error in percentage points ----
print("cross-validating ...")
table, best_alpha = H.compare(sets, df)
# retrieval uses the best DINOv3 feature set (any backbone if DINOv3 is absent)
mae = table.set_index("features")["mae_mean"]
pool = [n for n in sets if n.startswith("dinov3") and "patchmean" not in n] or \
       [n for n in sets if n != "handcrafted" and "patchmean" not in n] or ["handcrafted"]
ret_name = min(pool, key=lambda n: mae[n])
knn = H.knn_cv(sets[ret_name], df)
table = pd.concat([table, pd.DataFrame([{"features": f"5 nearest samples ({ret_name})", "alpha": None, **knn}])])
table = table.sort_values("mae_mean").reset_index(drop=True)
print(table.round(2).to_string(index=False))
"""),
code("""# ---- 2. pick the model; calibrate intervals on dev out-of-fold predictions; check them on the calib split ----
cands = table[~table.features.str.contains("patchmean|nearest|mean \\\\(baseline")]
BEST = cands.iloc[0].features
X, a = sets[BEST], best_alpha[BEST]
print(f"chosen features: {BEST} (alpha={a:g})")
dev, cal = (df.split == "dev").values, (df.split == "calib").values
oof = H.cv_oof(X, df, a)
p_dev, s_dev = H.per_sample(df[dev], oof[dev])
cp_dev = H.conformal_fit(p_dev, s_dev, H.sample_truth(df[dev]), COVERAGE)
head_dev = H.RidgeHead.fit(X[dev], df[H.FRACTIONS].values[dev], df.w.values[dev], a)
p_cal, s_cal = H.per_sample(df[cal], head_dev.predict(X[cal]))
t_cal = H.sample_truth(df[cal])
calib_report = {"mae": H.mae_pp(p_cal, t_cal), "conformal": H.conformal_report(p_cal, s_cal, t_cal, cp_dev, REVIEW_PP)}
print(H.to_json(calib_report))
"""),
code("""# ---- 3. final artifacts: head (dev+calib), recalibrated intervals, retrieval index, heatmap head ----
dc = dev | cal
head = H.RidgeHead.fit(X[dc], df[H.FRACTIONS].values[dc], df.w.values[dc], a)
p_all = pd.concat([p_dev, p_cal]); s_all = pd.concat([s_dev, s_cal])
cp = H.conformal_fit(p_all, s_all, H.sample_truth(df[dc]), COVERAGE)

E = H.sample_embeddings(sets[ret_name][dc], df[dc])
truth_dc = H.sample_truth(df[dc]).loc[E.index]
rep_key = df[dc].groupby("sample_id").key.first().loc[E.index]   # a representative photo per sample
lab = pd.read_csv(f"{WORK}/data/sample_psd.csv", dtype={"sample_id": str}).set_index("sample_id")
sieve_cols = list(lab.columns)

hm_name = [n for n in sets if n.endswith("patchmean")][0]
hm_head = H.RidgeHead.fit(sets[hm_name][dc], df[H.FRACTIONS].values[dc], df.w.values[dc], best_alpha[hm_name])
W, b = hm_head.unscaled()
print("final model fitted on", int(dc.sum()), "photos;", len(E), "samples in the retrieval index")
"""),
code("""# ---- 4. locked test split: run once, at the very end ----
test_report = None
if RUN_TEST:
    te = (df.split == "test").values
    p_te, s_te = H.per_sample(df[te], head.predict(X[te]))
    t_te = H.sample_truth(df[te])
    test_report = {"mae": H.mae_pp(p_te, t_te), "conformal": H.conformal_report(p_te, s_te, t_te, cp, REVIEW_PP)}
    print(H.to_json(test_report))
else:
    print("test split untouched (RUN_TEST = False)")
"""),
code("""# ---- 5. upload artifacts to the private model repo ----
out = LOCAL / "artifacts"; out.mkdir(exist_ok=True)
np.savez(out / "head.npz", **head.to_npz())
np.savez(out / "heatmap_head.npz", W=W, b=b)
np.savez(out / "retrieval_index.npz", emb=E.values.astype(np.float32), sample_id=np.array(E.index, dtype=str),
         fractions=truth_dc[H.FRACTIONS].values.astype(np.float32), type=np.array(truth_dc["type"], dtype=str),
         rep_key=np.array(rep_key.values, dtype=str), sieves=lab.loc[E.index, sieve_cols].values.astype(np.float32),
         sieve_names=np.array(sieve_cols, dtype=str))
table.to_csv(out / "cv_results.csv", index=False)
config = {"features": BEST, "alpha": a, "backbones": BACKBONE_LIST, "retrieval_features": ret_name,
          "heatmap_features": hm_name, "conformal": cp, "review_pp": REVIEW_PP, "overview_long_side": F.OVW_LONG,
          "crop_size": 768, "n_crops": 12, "calib_report": calib_report, "test_report": test_report,
          "cache_repo": CACHE, "n_batches": n_batches}
(out / "config.json").write_text(H.to_json(config))
(out / "README.md").write_text(f'''---
license: cc-by-4.0
tags: [soil, geotechnics, dinov3, regression]
---
# SoilSight models (private, work in progress)
Ridge heads on frozen backbone features, predicting gravel/sand/fines fractions of a soil sample from lab tray
photos. Trained on Photogranulometry v1.0 (Plante St-Cyr et al., 2025; doi:10.20383/103.01316; CC BY 4.0).
Features: `{BEST}`. Cross-validated results: `cv_results.csv`. Intervals: normalised split conformal, target
coverage {COVERAGE:.0%}. Limits: Montreal-region soils, one lab camera setup, no plasticity information.
''')
api.upload_folder(folder_path=str(out), repo_id=MODELS, repo_type="model", commit_message=f"Model: {BEST}")
print("uploaded to https://huggingface.co/" + MODELS)
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = root / "notebooks" / "02_train.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out, round(out.stat().st_size / 1e3), "kB")
