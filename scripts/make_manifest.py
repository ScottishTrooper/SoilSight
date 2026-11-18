"""Build the download manifest: which photos to fetch, with checksums, labels and split.

Picks K photos per (sample, moisture state, tray), spread evenly across the
photo indices (photos of one tray are near-replicates, so a few are enough).
Calibration images (one per sample) are listed too.

Usage: python3 -I make_manifest.py <dataset_root> <split_images_csv> <out_dir> [K]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, split_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
K = int(sys.argv[4]) if len(sys.argv) > 4 else 2
out.mkdir(parents=True, exist_ok=True)

# sha256 for every file, from FRDR's checksum listing
sha, cur = {}, None
for line in (root / "frdr-dfdr-checksums.txt").read_text(encoding="utf-8").splitlines():
    if line.startswith("Filename/Fichier:"):
        cur = line.split(":", 1)[1].strip()
    elif line.startswith("SHA256") and cur:
        sha[cur] = line.split(":", 1)[1].strip().split()[0]
        cur = None

img = pd.read_csv(split_csv, dtype={"sample_id": str, "tray": str})
img["idx"] = img.file.str.extract(r"_(\d{2})\.jpg$")[0].astype(int)

def pick(g):
    """Return index labels of K photos spread evenly over the tray's photo sequence."""
    g = g.sort_values("idx")
    n = len(g)
    if n <= K:
        return list(g.index)
    pos = np.unique(np.round(np.linspace(0, n - 1, K + 1)[:-1] + (n - 1) / (2 * K)).astype(int))
    return list(g.index[pos])

keep = []
for _, g in img.groupby(["sample_id", "dry_or_moist", "tray"]):
    keep.extend(pick(g))
sub = img.loc[sorted(keep)].reset_index(drop=True)
sub["kind"] = "soil"
cal = pd.DataFrame({"sample_id": sorted(sub.sample_id.unique())})
cal["file"] = "images_calibration/" + cal.sample_id + "_calibration.jpg"
cal["kind"] = "calibration"
man = pd.concat([sub, cal], ignore_index=True)
man["sha256"] = man.file.map(sha)
assert man.sha256.notna().all(), man[man.sha256.isna()].file.tolist()[:5]

# batch number: whole samples per batch so a sample is never split across shards
samples = sorted(man.sample_id.unique())
batch_of = {s: i // 10 for i, s in enumerate(samples)}   # 10 samples per batch
man["batch"] = man.sample_id.map(batch_of)
man = man.sort_values(["batch", "sample_id", "kind", "file"]).reset_index(drop=True)
man.to_csv(out / f"manifest_k{K}.csv", index=False)

n_soil = (man.kind == "soil").sum()
print(f"K={K}: {n_soil} soil photos + {(man.kind=='calibration').sum()} calibration, "
      f"{man.sample_id.nunique()} samples, {man.batch.nunique()} batches")
print("est. download GB (soil ~34 MB, calib ~23.6 MB):",
      round((n_soil * 34 + (man.kind == 'calibration').sum() * 23.6) / 1000, 1))
print(man[man.kind == "soil"].groupby("split").size().to_dict())
