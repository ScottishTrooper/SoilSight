"""Generate notebooks/04_test_and_showcase.ipynb (Kaggle). Usage: python3 make_showcase_notebook.py <repo_root>"""
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ["__init__.py", "views.py", "features.py", "heads.py", "predict.py"]:
        t.add(root / "soilsight" / f, arcname=f"soilsight/{f}")
    for f in ["index.html", "style.css", "app.js"]:
        t.add(root / "site" / f, arcname=f"site/{f}")
    t.add(root / "data" / "sample_psd.csv", arcname="data/sample_psd.csv")
payload = base64.b64encode(buf.getvalue()).decode()
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("""# SoilSight · 04 · Final test results + free showcase page

**This opens the locked test split, once.** Run it only when the model and its settings are final; whatever it
reports is the result. It then runs the model on all held-out test samples (estimate, range, review flag, similar
samples, heatmaps) and publishes a **static** Hugging Face Space (free) where visitors can browse them next to the
lab results.

Accelerator **None** is fine (a GPU just makes the heatmaps faster), Internet **On**, `HF_TOKEN` attached.
**Save Version → Save & Run All.** About 10–20 minutes."""),
code("""# ---- settings ----
CACHE_REPO_NAME = "photogranulometry-cache"
MODEL_REPO_NAME = "soilsight-models"
SHOWCASE_NAME   = "soilsight-showcase"
PRIVATE_SPACE   = True     # set to False (or flip it in the Space settings) when you're ready to share
IMG_WIDTH       = 900
"""),
code(f"""# ---- unpack ----
import base64, io, tarfile, sys, os, json, re, shutil
PAYLOAD = "{payload}"
WORK = "/kaggle/working/soilsight_src" if os.path.isdir("/kaggle/working") else os.path.abspath("soilsight_src")
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)
_t = tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz")
try:
    _t.extractall(WORK, filter="data")
except TypeError:
    _t.extractall(WORK)
sys.path.insert(0, WORK)
import numpy as np, pandas as pd, cv2
from pathlib import Path
from soilsight import features as F, heads as H
from soilsight.predict import Predictor, assess
LOCAL = Path("/kaggle/tmp_show" if os.path.isdir("/kaggle") else "tmp_show"); LOCAL.mkdir(exist_ok=True)
"""),
code("""# ---- load everything ----
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN and TOKEN.startswith("hf_"), "HF_TOKEN secret not attached"
os.environ["HF_TOKEN"] = TOKEN
api = HfApi(token=TOKEN); user = api.whoami()["name"]
CACHE, MODELS, SHOW = f"{user}/{CACHE_REPO_NAME}", f"{user}/{MODEL_REPO_NAME}", f"{user}/{SHOWCASE_NAME}"
ART = Path(snapshot_download(MODELS, repo_type="model", local_dir=LOCAL / "artifacts", token=TOKEN))
cfg = json.loads((ART / "config.json").read_text())
assert "method" in cfg["conformal"], "run notebook 02b first"
cache_files = api.list_repo_files(CACHE, repo_type="dataset")
feat_file = sorted(f for f in cache_files if f.startswith("features/"))[-1]
feats = F.load_features(Path(hf_hub_download(CACHE, feat_file, repo_type="dataset", token=TOKEN)))
samples = pd.read_csv(hf_hub_download(CACHE, "data/split_samples_seed0.csv", repo_type="dataset", token=TOKEN), dtype={"sample_id": str})
psd = pd.read_csv(f"{WORK}/data/sample_psd.csv", dtype={"sample_id": str}).set_index("sample_id")
df = H.photo_table(feats, samples); sets = H.feature_sets(feats)
try:
    import torch; DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
PR = Predictor(ART, device=DEVICE)
print("model:", cfg["features"], "| uncertainty:", cfg["conformal"]["method"], "| review if half-width >", round(cfg["review_pp"], 1), "pp | device:", DEVICE)
"""),
code("""# ---- THE TEST SPLIT (opened once) ----
te = df[df.split == "test"]
FR = H.FRACTIONS
res = {}
for sid, g in te.groupby("sample_id"):
    rows = g.index.values
    P = PR.head.predict(sets[cfg["features"]][rows])
    r = assess(P, sets[cfg["retrieval_features"]][rows], cfg, PR.idx, "thumbs")
    res[sid] = r | {"keys": g.key.tolist(), "states": g.state.tolist()}
truth = H.sample_truth(te)
est = pd.DataFrame({sid: r["fractions_pct"] for sid, r in res.items()}).T[FR]
half = pd.DataFrame({sid: r["half_width_pp"] for sid, r in res.items()}).T[FR]
lab = truth.loc[est.index, FR] * 100
err = (est - lab).abs()
acc = pd.Series({sid: not r["flags"] for sid, r in res.items()}).loc[est.index]
nat = truth.loc[est.index, "type"] != "artificial"
devcal = samples[samples.split.isin(["dev", "calib"])]
baseline = (lab - devcal[FR].mean()).abs()
test_report = {
    "n": int(len(est)),
    "mae": {f"mae_{f}": float(err[f].mean()) for f in FR} | {"mae_mean": float(err.values.mean()),
            "mae_mean_natural": float(err[nat].values.mean())},
    "baseline_mae_mean": float(baseline.values.mean()),
    "per_fraction_coverage": {f: float((err[f] <= half[f]).mean()) for f in FR},
    "joint_coverage": float((err <= half).all(axis=1).mean()),
    "share_auto_accepted": float(acc.mean()),
    "mae_mean_accepted": float(err[acc].values.mean()) if acc.any() else None,
    "mae_mean_flagged": float(err[~acc].values.mean()) if (~acc).any() else None,
}
print(H.to_json(test_report))
cfg["test_report"] = test_report
api.upload_file(path_or_fileobj=H.to_json(cfg).encode(), path_in_repo="config.json", repo_id=MODELS,
                repo_type="model", commit_message="Final test-split results")
"""),
code("""# ---- images: tray photo, heatmaps, thumbnails ----
SITE = LOCAL / "site"
shutil.rmtree(SITE, ignore_errors=True); shutil.copytree(f"{WORK}/site", SITE)
(SITE / "img").mkdir(); shutil.copytree(ART / "thumbs", SITE / "thumbs")
pick = {}
for sid, r in res.items():                    # show a dry photo if there is one
    order = sorted(zip(r["states"], r["keys"]), key=lambda t: (t[0] != "dry", t[1]))
    pick[order[0][1]] = sid
batch_of = {}
for f in sorted(f for f in cache_files if re.fullmatch(r"meta/batch_\\d{3}\\.jsonl", f)):
    for line in open(hf_hub_download(CACHE, f, repo_type="dataset", token=TOKEN)):
        m = json.loads(line)
        if m.get("key") in pick:
            batch_of[m["key"]] = f[5:14]
batches = sorted(set(batch_of.values()))
snapshot_download(CACHE, repo_type="dataset", allow_patterns=[f"views/{b}.tar" for b in batches], local_dir=LOCAL / "cache", token=TOKEN)
def save(img_rgb, path, width):
    h = int(round(img_rgb.shape[0] * width / img_rgb.shape[1]))
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(img_rgb, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 84])
done = 0
for b in batches:
    with tarfile.open(LOCAL / "cache" / "views" / f"{b}.tar") as t:
        for m in t:
            key, _, ext = m.name.partition(".")
            if ext != "overview.jpg" or key not in pick:
                continue
            sid = pick[key]
            ov = cv2.cvtColor(cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            base = F.resize_overview(ov)
            save(base, SITE / "img" / f"{sid}.jpg", IMG_WIDTH)
            save(base, SITE / "img" / f"{sid}_t.jpg", 160)
            for f in FR:
                save(PR.heatmap(ov, f)[0], SITE / "img" / f"{sid}_{f}.jpg", IMG_WIDTH)
            done += 1
shutil.rmtree(LOCAL / "cache", ignore_errors=True)
print(f"images for {done}/{len(res)} samples")
"""),
code("""# ---- data.json + publish the static Space ----
sieve_cols = list(psd.columns)
out = []
for sid in est.index:
    r = res[sid]
    out.append({
        "id": sid, "type": str(truth.loc[sid, "type"]), "status": "accepted" if not r["flags"] else "review",
        "reasons": r["flags"], "est": r["fractions_pct"], "half": r["half_width_pp"],
        "lab": {f: round(float(lab.loc[sid, f]), 1) for f in FR}, "lab_curve": [round(float(v), 2) for v in psd.loc[sid, sieve_cols]],
        "per_photo": r["per_photo_pct"], "n_photos": len(r["per_photo_pct"]), "range_note": r["range_note"],
        "img": f"img/{sid}.jpg", "thumb": f"img/{sid}_t.jpg", "heat": {f: f"img/{sid}_{f}.jpg" for f in FR},
        "neighbours": [{"sample_id": n["sample_id"], "type": n["type"], "similarity": round(n["similarity"], 3),
                        "fractions": [round(v, 1) for v in n["fractions"]], "sieves": [round(v, 2) for v in n["sieves"]],
                        "thumb": f"thumbs/{n['sample_id']}.jpg"} for n in r["neighbours"]],
    })
cv = pd.read_csv(ART / "cv_results.csv")
data = {"summary": {"test": {**test_report["mae"], **{k: v for k, v in test_report.items() if k != "mae"}},
                    "baseline_mae": test_report["baseline_mae_mean"], "cv": json.loads(cv.to_json(orient="records")),
                    "model": cfg.get("model_version", cfg["features"]), "review_pp": cfg["review_pp"]},
        "sieve_mm": [float(c.replace("%_", "").replace("mm", "")) for c in sieve_cols], "samples": out}
(SITE / "data.json").write_text(json.dumps(data, default=float))
(SITE / "README.md").write_text('''---
title: SoilSight showcase
emoji: 🪨
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
license: cc-by-4.0
short_description: Grain size from soil tray photos, on held-out test samples
---
''')
api.create_repo(SHOW, repo_type="space", space_sdk="static", private=PRIVATE_SPACE, exist_ok=True)
api.upload_folder(folder_path=str(SITE), repo_id=SHOW, repo_type="space", commit_message="Publish showcase")
print("Showcase:", f"https://huggingface.co/spaces/{SHOW}", "| files:", sum(1 for _ in SITE.rglob("*") if _.is_file()))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = root / "notebooks" / "04_test_and_showcase.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out, round(out.stat().st_size / 1e3), "kB")
