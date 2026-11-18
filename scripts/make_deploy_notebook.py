"""Generate notebooks/03_deploy_space.ipynb (Kaggle, CPU). Usage: python3 make_deploy_notebook.py <repo_root>"""
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ["__init__.py", "scale.py", "views.py", "cache.py", "features.py", "heads.py", "predict.py"]:
        t.add(root / "soilsight" / f, arcname=f"soilsight/{f}")
    for f in ["manifest_k2.csv", "split_samples_seed0.csv"]:
        t.add(root / "data" / f, arcname=f"data/{f}")
    for f in ["app.py", "requirements.txt", "README.md", "ABOUT.md"]:
        t.add(root / "space" / f, arcname=f"space/{f}")
payload = base64.b64encode(buf.getvalue()).decode()

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("""# SoilSight · 03 · Package and deploy the demo Space

Run after notebook 02. CPU is enough (Accelerator **None**), Internet **On**, `HF_TOKEN` attached.

1. Sets the **"unusual photo"** threshold from the saved features (run notebook 02b first: it calibrates the ranges).
2. Makes thumbnails of every sample in the similar-sample index.
3. Downloads a few **held-out test** samples from FRDR as clickable examples (never used in training).
4. Creates the Gradio **Space** and uploads the app. It starts **private**; make it public in its Settings when you're happy.

The Space reads your private model repo using `HF_TOKEN`, stored as a Space secret (never shown in the code or logs)."""),
code("""# ---- settings ----
EXAMPLE_LINK    = "https://www.frdr-dfdr.ca/repo/files/1/published/publication_1311/submitted_data/images_soil/0001_artificial_dry_tray045_02.jpg"
CACHE_REPO_NAME = "photogranulometry-cache"
MODEL_REPO_NAME = "soilsight-models"
SPACE_NAME      = "soilsight"
PRIVATE_SPACE   = True
N_EXAMPLES      = 4       # held-out test samples offered as examples
PHOTOS_PER_EXAMPLE = 2
EXAMPLE_ROWS    = None    # leave as None
"""),
code(f"""# ---- unpack ----
import base64, io, tarfile, sys, os, json, re, shutil, time
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
from soilsight import features as F, heads as H, cache as C
LOCAL = Path("/kaggle/tmp_deploy" if os.path.isdir("/kaggle") else "tmp_deploy"); LOCAL.mkdir(exist_ok=True)
"""),
code("""# ---- Hugging Face ----
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN and TOKEN.startswith("hf_"), "HF_TOKEN secret not attached"
api = HfApi(token=TOKEN)
user = api.whoami()["name"]
CACHE, MODELS, SPACE = f"{user}/{CACHE_REPO_NAME}", f"{user}/{MODEL_REPO_NAME}", f"{user}/{SPACE_NAME}"
ART = Path(snapshot_download(MODELS, repo_type="model", local_dir=LOCAL / "artifacts", token=TOKEN))
cfg = json.loads((ART / "config.json").read_text())
cache_files = api.list_repo_files(CACHE, repo_type="dataset")
feat_file = sorted(f for f in cache_files if f.startswith("features/"))[-1]
feats = F.load_features(Path(hf_hub_download(CACHE, feat_file, repo_type="dataset", token=TOKEN)))
samples = pd.read_csv(f"{WORK}/data/split_samples_seed0.csv", dtype={"sample_id": str})
df = H.photo_table(feats, samples); sets = H.feature_sets(feats)
print("model features:", cfg["features"], "| features file:", feat_file)
"""),
code("""# ---- 1. unusual-photo threshold (ranges were calibrated in notebook 02b) ----
# unusual photo: similarity of a single in-distribution photo to its nearest *other* sample in the index
cal = (df.split == "calib").values
idx = np.load(ART / "retrieval_index.npz")
E, ids = idx["emb"], idx["sample_id"].astype(str)
R = sets[cfg["retrieval_features"]][cal]
R = R / np.linalg.norm(R, axis=1, keepdims=True).clip(1e-9)
S = R @ E.T
own = np.asarray(df.sample_id.values[cal], dtype=str)[:, None] == ids[None, :]
S[own] = -1
nearest = S.max(1)
ood = float(np.quantile(nearest, 0.01))
assert "method" in cfg.get("conformal", {}), "run notebook 02b (uncertainty v2) before this one"
print(f"unusual-photo threshold (1st percentile of nearest-sample similarity): {ood:.3f}")
cfg["ood_similarity"] = ood
cfg["model_version"] = f"{cfg['features']} · {api.model_info(MODELS).sha[:7]}"
(ART / "config.json").write_text(H.to_json(cfg))
"""),
code("""# ---- 2. thumbnails for the similar-sample index ----
rep = dict(zip(idx["sample_id"].astype(str), idx["rep_key"].astype(str)))
need_keys = set(rep.values())
batch_of = {}
for f in sorted(f for f in cache_files if re.fullmatch(r"meta/batch_\\d{3}\\.jsonl", f)):
    for line in open(hf_hub_download(CACHE, f, repo_type="dataset", token=TOKEN)):
        m = json.loads(line)
        if m.get("key") in need_keys:
            batch_of[m["key"]] = f[5:14]
batches = sorted(set(batch_of.values()))
snapshot_download(CACHE, repo_type="dataset", allow_patterns=[f"views/{b}.tar" for b in batches],
                  local_dir=LOCAL / "cache", token=TOKEN)
th = ART / "thumbs"; th.mkdir(exist_ok=True)
key_to_sid = {v: k for k, v in rep.items()}
for b in batches:
    with tarfile.open(LOCAL / "cache" / "views" / f"{b}.tar") as t:
        for m in t:
            key, _, ext = m.name.partition(".")
            if ext == "overview.jpg" and key in key_to_sid:
                im = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_COLOR)
                im = cv2.resize(im, (360, int(360 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(th / f"{key_to_sid[key]}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
shutil.rmtree(LOCAL / "cache", ignore_errors=True)
print(f"{len(list(th.glob('*.jpg')))} / {len(rep)} thumbnails")
api.upload_folder(folder_path=str(ART), repo_id=MODELS, repo_type="model",
                  allow_patterns=["config.json", "thumbs/*"], commit_message="Single-photo range, OOD threshold, thumbnails")
"""),
code("""# ---- 3. held-out test examples from FRDR (natural soils, spread across fines content) ----
SRC = re.search(r"^(.*?)/(images_soil|images_calibration|metadata)/", EXAMPLE_LINK).group(1)
man = pd.read_csv(f"{WORK}/data/manifest_k2.csv", dtype={"sample_id": str, "tray": str})
if EXAMPLE_ROWS is None:
    t = samples[(samples.split == "test") & (samples.type != "artificial")].sort_values("fines")
    pick = t.iloc[np.linspace(0, len(t) - 1, N_EXAMPLES).round().astype(int)]
    rows = []
    for _, s in pick.iterrows():
        m = man[(man.sample_id == s.sample_id) & (man.kind == "soil")].sort_values(["dry_or_moist", "file"])
        rows += m.head(PHOTOS_PER_EXAMPLE).to_dict("records")
else:
    rows = EXAMPLE_ROWS
SPACE_DIR = LOCAL / "space"
shutil.rmtree(SPACE_DIR, ignore_errors=True)
shutil.copytree(f"{WORK}/space", SPACE_DIR)
shutil.copytree(f"{WORK}/soilsight", SPACE_DIR / "soilsight", ignore=shutil.ignore_patterns("__pycache__", "cache.py"))
ty = samples.set_index("sample_id")["type"]
for r in rows:
    tmp = C.fetch(SRC, r["file"], LOCAL / "ex_raw" / Path(r["file"]).name, r["sha256"])
    d = SPACE_DIR / "examples" / f"{r['sample_id']}_{ty.get(r['sample_id'], 'soil')}"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{Path(r['file']).stem}.jpg"), cv2.imread(str(tmp)), [cv2.IMWRITE_JPEG_QUALITY, 88])
shutil.rmtree(LOCAL / "ex_raw", ignore_errors=True)
for d in sorted((SPACE_DIR / "examples").iterdir()):
    print(d.name, [p.name for p in d.iterdir()])
"""),
code("""# ---- 4. create the Space and upload ----
api.create_repo(SPACE, repo_type="space", space_sdk="gradio", private=PRIVATE_SPACE, exist_ok=True)
api.add_space_secret(SPACE, "HF_TOKEN", TOKEN)                     # lets the Space read your private model repo
api.add_space_variable(SPACE, "SOILSIGHT_MODEL_REPO", MODELS)
api.upload_folder(folder_path=str(SPACE_DIR), repo_id=SPACE, repo_type="space", commit_message="Deploy SoilSight")
print("Space:", f"https://huggingface.co/spaces/{SPACE}", "(building; first start takes a few minutes)")
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = root / "notebooks" / "03_deploy_space.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out, round(out.stat().st_size / 1e3), "kB")
