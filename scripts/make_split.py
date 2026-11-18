"""Sample-level split for Photogranulometry v1.0.

Every image of a sample (all trays, dry and moist) goes to the same split.
Splits: test (locked, ~20%), calib (~16%, for conformal/temperature), dev (rest),
and dev is further divided into 5 CV folds for model selection.
Stratified by sample type x fines bin x gravel-dominant flag.

Usage: python3 -I make_split.py <dataset_root> <out_dir> [seed]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(seed)

c = pd.read_csv(root / "metadata/psd_mass_path_80_to_0_08mm_no_ballast.csv", dtype={"sample_id": str})
h = pd.read_csv(root / "metadata/psd_mass_path_80_to_0_0018mm.csv", dtype={"sample_id": str})
sieves = [x for x in c.columns if x.startswith("%_") and x.endswith("mm")]
c["file"] = c.image_path.str.replace("photogranulometry_v1_0/", "", regex=False)
c["type"] = c.file.str.extract(r"images_soil/\d{4}_([a-z]+)_")[0]
c["tray"] = c.file.str.extract(r"_tray(\d{3})_")[0]

s = c.groupby("sample_id").agg(type=("type", "first"), **{k: (k, "first") for k in sieves})
s["gravel"] = 100 - s["%_5mm"]
s["fines"] = s["%_0.08mm"]
s["sand"] = s["%_5mm"] - s["%_0.08mm"]
s["has_hydrometer"] = s.index.isin(h.sample_id)
s["fines_bin"] = pd.cut(s.fines, [-1, 5, 12, 30, 50, 101], labels=["<5", "5-12", "12-30", "30-50", ">=50"])
s["gravel_dom"] = s.gravel > s.sand
s["stratum"] = s.type + "|" + s.fines_bin.astype(str) + "|" + s.gravel_dom.map({True: "G", False: "S"})

# Assign within each stratum by a shuffled round-robin over a fixed 25-slot pattern:
# 5 test, 4 calib, 16 dev  (=20% / 16% / 64%)
# Systematic sampling: order samples stratum by stratum (random order within and
# between strata), then walk an interleaved 25-slot pattern continuously, so every
# stratum gets close-to-proportional allocation and totals are exact.
counts = {"test": 5, "calib": 4, "dev": 16}
pos = sorted((((i + 0.5) / n), name) for name, n in counts.items() for i in range(n))
pattern = [name for _, name in pos]
strata = list(s.groupby("stratum").groups.items())
order = []
for j in rng.permutation(len(strata)):
    order.extend(rng.permutation(np.asarray(strata[j][1])))
start = int(rng.integers(len(pattern)))
s["split"] = [None] * len(s)
for k, sid in enumerate(order):
    s.loc[sid, "split"] = pattern[(start + k) % len(pattern)]

# 5-fold CV inside dev, stratified the same way
s["fold"] = -1
dev = s[s.split == "dev"]
off = 0
for stratum, g in dev.groupby("stratum"):
    idx = rng.permutation(g.index.to_numpy())
    for k, sid in enumerate(idx):
        s.loc[sid, "fold"] = (k + off) % 5
    off += len(g)

s_out = s[["type", "gravel", "sand", "fines", "fines_bin", "gravel_dom", "has_hydrometer", "split", "fold"]].round(3)
s_out.to_csv(out / f"split_samples_seed{seed}.csv")

img = c[["sample_id", "file", "dry_or_moist", "tray", "type"] + sieves + ["%_water"]].merge(
    s[["split", "fold"]], left_on="sample_id", right_index=True)
img.to_csv(out / f"split_images_seed{seed}.csv", index=False)

# ---- checks ----
assert img.groupby("sample_id").split.nunique().max() == 1, "sample leaked across splits"
assert set(s.index) == set(c.sample_id)
print("samples per split:\n", s.split.value_counts())
print("images per split:\n", img.split.value_counts())
print("\ntype x split:\n", pd.crosstab(s.type, s.split))
print("\nfines bin x split:\n", pd.crosstab(s.fines_bin, s.split))
print("\ngravel-dominant x split:\n", pd.crosstab(s.gravel_dom, s.split))
print("\nhydrometer x split:\n", pd.crosstab(s.has_hydrometer, s.split))
print("\nfold sizes:\n", s[s.split == "dev"].fold.value_counts().sort_index())
print("\nfraction means by split:\n", s.groupby("split")[["gravel", "sand", "fines"]].mean().round(1))
