"""Generate notebooks/01_build_cache.ipynb: a self-contained Kaggle notebook.

The soilsight package source and the download manifest are embedded (gzip+base64),
so the user uploads a single .ipynb to Kaggle.
Usage: python3 make_cache_notebook.py <repo_root>
"""
import base64
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ["soilsight/__init__.py", "soilsight/scale.py", "soilsight/views.py", "soilsight/cache.py",
              "data/manifest_k2.csv", "data/manifest_k3.csv", "data/split_samples_seed0.csv"]:
        t.add(root / f, arcname=f)
payload = base64.b64encode(buf.getvalue()).decode()

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("""# SoilSight · 01 · Build the image cache

Streams the Photogranulometry photos from FRDR in batches, turns each 45 MP photo into
small training views, uploads them to a **private** Hugging Face dataset, and deletes the originals.
Your disk never holds more than one batch (~3.5 GB).

**Before running**
1. Session options (right panel): **Internet → On**. Accelerator: **None** (CPU is enough; this saves GPU quota).
2. Add-ons → Secrets: `HF_TOKEN` attached (your *new* write token).
3. Paste one FRDR file link into `EXAMPLE_LINK` below (right-click any image on the dataset page → Copy Link Address).
4. First run: keep `MAX_BATCHES = 1` and check the QC contact sheet it prints. Then set it to `None` and **Run All**.

If the session stops (12 h limit, disconnect), just run it again: finished batches are skipped.

Data: Plante St-Cyr, Duhaime, Dubé & Grenier (2025), *Photogranulometry* v1.0, FRDR, doi:10.20383/103.01316, CC BY 4.0."""),
code("""# ---- settings ----
EXAMPLE_LINK = "PASTE_ONE_FRDR_FILE_LINK_HERE"   # e.g. https://www.frdr-dfdr.ca/repo/files/1/published/publication_NNN/submitted_data/.../images_soil/0001_..._01.jpg
REPO_NAME    = "photogranulometry-cache"         # created under your HF username, private
PHOTOS_PER_TRAY = 2        # 2 -> 3,062 photos (~111 GB streamed); 3 -> 4,573 photos (~163 GB)
N_CROPS      = 12          # full-res 768x768 crops per photo
WITH_HALFRES = True        # also keep a half-resolution copy (~5 MB/photo) for re-cropping later
MAX_BATCHES  = 1           # 1 for a test run; None for everything (32 batches)
TIME_LIMIT_H = 11.0        # stop cleanly before Kaggle's 12 h limit
"""),
code(f"""# ---- unpack the embedded code + manifests ----
import base64, io, tarfile, sys, os
PAYLOAD = "{payload}"
WORK = "/kaggle/working/soilsight_src" if os.path.isdir("/kaggle/working") else os.path.abspath("soilsight_src")
os.makedirs(WORK, exist_ok=True)
_t = tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz")
try:
    _t.extractall(WORK, filter="data")
except TypeError:   # Python without extraction filters
    _t.extractall(WORK)
sys.path.insert(0, WORK)
import soilsight.cache as cache
print("code unpacked to", WORK)
"""),
code("""# ---- Hugging Face login (token comes from Kaggle Secrets, never typed here) ----
from huggingface_hub import HfApi, CommitOperationAdd
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN and TOKEN.startswith("hf_"), "HF_TOKEN secret not found: Add-ons -> Secrets, and tick 'attached'"
api = HfApi(token=TOKEN)
user = api.whoami()["name"]
REPO = f"{user}/{REPO_NAME}"
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
print("logged in as", user, "| dataset repo:", REPO, "(private)")
"""),
code("""# ---- work out the FRDR base URL and test one download ----
import re, pandas as pd, tempfile
from pathlib import Path
m = re.search(r"^(.*?)/(images_soil|images_calibration|metadata)/", EXAMPLE_LINK)
assert m, "EXAMPLE_LINK must be a link to a file inside images_soil/ or images_calibration/"
SRC = m.group(1)
man = pd.read_csv(f"{WORK}/data/manifest_k{PHOTOS_PER_TRAY}.csv", dtype={"sample_id": str, "tray": str})
test = man[man.kind == "calibration"].iloc[0]
with tempfile.TemporaryDirectory() as d:
    p = cache.fetch(SRC, test.file, Path(d) / "t.jpg", test.sha256)
    print(f"FRDR download OK and checksum verified: {test.file} ({p.stat().st_size/1e6:.1f} MB)")
print("base URL:", SRC)
"""),
code("""# ---- upload the manifests + a dataset card once ----
existing = set(api.list_repo_files(REPO, repo_type="dataset"))
if "README.md" not in existing:
    card = f'''---
license: cc-by-4.0
pretty_name: Photogranulometry cache (SoilSight)
---
# Photogranulometry cache for SoilSight

Derived from **Photogranulometry v1.0** (Plante St-Cyr, Duhaime, Dubé & Grenier, 2025;
doi:10.20383/103.01316), licensed CC BY 4.0. Changes: {PHOTOS_PER_TRAY} photos per tray-state were
selected; each was reduced to a 1536 px overview of the soil area, {N_CROPS} full-resolution
768x768 crops, a soil mask and (optionally) a half-resolution copy. Labels are the original
sample-level sieve results. Splits are by sample (no sample appears in two splits).

Layout: `views/`, `halfres/` (WebDataset tars), `meta/` (JSON lines, incl. per-sample scale
from the calibration markers), `qc/` (contact sheets), `data/` (manifests and splits).
'''
    ops = [CommitOperationAdd("README.md", card.encode())]
    for f in ["manifest_k2.csv", "manifest_k3.csv", "split_samples_seed0.csv"]:
        ops.append(CommitOperationAdd(f"data/{f}", f"{WORK}/data/{f}"))
    api.create_commit(REPO, repo_type="dataset", operations=ops, commit_message="Add card and manifests")
    print("dataset card + manifests uploaded")
else:
    print("card already present")
"""),
code("""# ---- main loop: one batch at a time, resumable ----
import time, shutil
t_start = time.time()
done = {int(f[11:14]) for f in api.list_repo_files(REPO, repo_type="dataset") if re.fullmatch(r"meta/batch_\\d{3}\\.jsonl", f)}
todo = [b for b in sorted(man.batch.unique()) if b not in done]
if MAX_BATCHES is not None:
    todo = todo[:MAX_BATCHES]
print(f"{len(done)} batch(es) already done, {len(todo)} to run now")
work = Path("/kaggle/tmp_cache" if os.path.isdir("/kaggle") else "tmp_cache")
last_qc = None
for b in todo:
    if (time.time() - t_start) / 3600 > TIME_LIMIT_H:
        print("time limit reached; rerun the notebook to continue"); break
    rows = man[man.batch == b].to_dict("records")
    shutil.rmtree(work, ignore_errors=True)
    paths = cache.build_batch(rows, SRC, work, n_crops=N_CROPS, with_halfres=WITH_HALFRES)
    ops = [CommitOperationAdd(f"{k}/{p.name}", str(p)) for k, p in paths.items() if k != "meta"]
    ops.append(CommitOperationAdd(f"meta/{paths['meta'].name}", str(paths["meta"])))  # marks batch done
    for attempt in range(4):
        try:
            api.create_commit(REPO, repo_type="dataset", operations=ops, commit_message=f"batch {b:03d}")
            break
        except Exception as e:
            print("upload failed, retrying:", repr(e)[:120]); time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f"could not upload batch {b}")
    last_qc = paths["qc"].read_bytes()
    shutil.rmtree(work, ignore_errors=True)
    print(f"batch {b:03d} uploaded | elapsed {(time.time()-t_start)/60:.0f} min")
"""),
code("""# ---- look at the last QC contact sheet: yellow = detected soil, red = crops, '!' = flagged ----
from IPython.display import Image, display
if last_qc:
    display(Image(data=last_qc, width=1400))
"""),
code("""# ---- progress summary ----
files = api.list_repo_files(REPO, repo_type="dataset")
n_done = sum(bool(re.fullmatch(r"meta/batch_\\d{3}\\.jsonl", f)) for f in files)
print(f"{n_done}/{man.batch.nunique()} batches in https://huggingface.co/datasets/{REPO}")
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = root / "notebooks" / "01_build_cache.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(out, round(out.stat().st_size / 1e6, 2), "MB")
