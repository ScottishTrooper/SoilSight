"""Generate notebooks/05_live_demo.ipynb (Kaggle). Usage: python3 make_live_notebook.py <repo_root>"""
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
        t.add(root / "soilsight" / f, arcname=f"space/soilsight/{f}")
    for f in ["app.py", "ABOUT.md"]:
        t.add(root / "space" / f, arcname=f"space/{f}")
    for f in ["manifest_k2.csv", "split_samples_seed0.csv"]:
        t.add(root / "data" / f, arcname=f"data/{f}")
payload = base64.b64encode(buf.getvalue()).decode()
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("""# SoilSight · 05 · Live demo (temporary public link)

Starts the full app (photo upload, estimate, range, review flag, evidence, heatmap, technician log) and prints a
public `*.gradio.live` link that works **while this notebook is running** (Gradio share links last up to a week).
Use it for interviews or when someone wants to try their own photos.

Run it **interactively** (click **Run All** in the editor, don't use Save Version). GPU **T4** makes each analysis
take a few seconds instead of ~30 s, but uses GPU hours while the session is open; CPU works too.
Internet **On**, `HF_TOKEN` attached. Stop the session when you're done."""),
code("""# ---- settings ----
EXAMPLE_LINK    = "https://www.frdr-dfdr.ca/repo/files/1/published/publication_1311/submitted_data/images_soil/0001_artificial_dry_tray045_02.jpg"
MODEL_REPO_NAME = "soilsight-models"
N_EXAMPLES      = 4
PHOTOS_PER_EXAMPLE = 2
SHARE           = True     # public *.gradio.live link
EXAMPLE_ROWS    = None     # leave as None
"""),
code(f"""# ---- unpack ----
import base64, io, tarfile, sys, os, json, re, shutil
PAYLOAD = "{payload}"
WORK = "/kaggle/working/soilsight_live" if os.path.isdir("/kaggle/working") else os.path.abspath("soilsight_live")
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)
_t = tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz")
try:
    _t.extractall(WORK, filter="data")
except TypeError:
    _t.extractall(WORK)
SPACE = os.path.join(WORK, "space")
sys.path.insert(0, SPACE)
import numpy as np, pandas as pd, cv2
from pathlib import Path
from soilsight import cache as C
"""),
code("""# ---- model files from your private repo ----
import subprocess
try:
    import gradio
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
from huggingface_hub import HfApi, snapshot_download
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN and TOKEN.startswith("hf_"), "HF_TOKEN secret not attached"
os.environ["HF_TOKEN"] = TOKEN
user = HfApi(token=TOKEN).whoami()["name"]
ART = snapshot_download(f"{user}/{MODEL_REPO_NAME}", repo_type="model", local_dir=os.path.join(WORK, "artifacts"), token=TOKEN)
os.environ["SOILSIGHT_ARTIFACTS"] = ART
try:
    import torch; os.environ["SOILSIGHT_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    os.environ["SOILSIGHT_DEVICE"] = "cpu"
print("model files ready | device:", os.environ["SOILSIGHT_DEVICE"])
"""),
code("""# ---- example photos: held-out test samples, full resolution from FRDR ----
SRC = re.search(r"^(.*?)/(images_soil|images_calibration|metadata)/", EXAMPLE_LINK).group(1)
man = pd.read_csv(f"{WORK}/data/manifest_k2.csv", dtype={"sample_id": str, "tray": str})
samples = pd.read_csv(f"{WORK}/data/split_samples_seed0.csv", dtype={"sample_id": str})
if EXAMPLE_ROWS is None:
    t = samples[(samples.split == "test") & (samples.type != "artificial")].sort_values("fines")
    pick = t.iloc[np.linspace(0, len(t) - 1, N_EXAMPLES).round().astype(int)]
    rows = []
    for _, s in pick.iterrows():
        m = man[(man.sample_id == s.sample_id) & (man.kind == "soil")].sort_values(["dry_or_moist", "file"])
        rows += m.head(PHOTOS_PER_EXAMPLE).to_dict("records")
else:
    rows = EXAMPLE_ROWS
ty = samples.set_index("sample_id")["type"]
for r in rows:
    d = Path(SPACE) / "examples" / f"{r['sample_id']}_{ty.get(r['sample_id'], 'soil')}"
    d.mkdir(parents=True, exist_ok=True)
    C.fetch(SRC, r["file"], d / Path(r["file"]).name, r["sha256"])
print(sorted(p.name for p in (Path(SPACE) / "examples").iterdir()))
"""),
code("""# ---- launch ----
os.chdir(SPACE)
import app
app.demo.queue(max_size=8).launch(share=SHARE, prevent_thread_lock=not SHARE, quiet=False, allowed_paths=[os.environ["SOILSIGHT_ARTIFACTS"]])
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = root / "notebooks" / "05_live_demo.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out, round(out.stat().st_size / 1e3), "kB")
