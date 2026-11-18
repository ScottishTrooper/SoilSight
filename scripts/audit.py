"""Audit of the Photogranulometry v1.0 dataset (metadata only, no images needed).

Usage: python3 -I audit.py <dataset_root> <out_dir>
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
R = {}  # results collected for the report

# ---------- 1. Full file listing from the checksum file ----------
pat_soil = re.compile(r"images_soil/(\d{4})_([a-z]+)_(dry|moist)_tray(\d{3})_(\d{2})\.jpg")
pat_cal = re.compile(r"images_calibration/(\d{4})_calibration\.jpg")
soil_rows, cal_ids, bad_names = [], [], []
for line in (root / "frdr-dfdr-checksums.txt").read_text(encoding="utf-8").splitlines():
    if not line.startswith("Filename/Fichier:"):
        continue
    p = line.split(":", 1)[1].strip()
    if "images_soil/" in p:
        m = pat_soil.search(p)
        if m:
            soil_rows.append(dict(sample_id=m[1], type=m[2], state=m[3], tray=m[4], idx=int(m[5]),
                                  file=f"images_soil/{m[0].split('/')[-1]}"))
        else:
            bad_names.append(p)
    elif "images_calibration/" in p:
        m = pat_cal.search(p)
        (cal_ids.append(m[1]) if m else bad_names.append(p))

img = pd.DataFrame(soil_rows)
img.to_csv(out / "image_index_all.csv", index=False)
R["n_images_listed"] = len(img)
R["n_samples_with_images"] = img.sample_id.nunique()
R["n_calibration_images"] = len(cal_ids)
R["unparsed_filenames"] = bad_names
R["duplicate_files"] = int(img.file.duplicated().sum())

# per-sample consistency of type
type_per_sample = img.groupby("sample_id").type.nunique()
R["samples_with_multiple_types"] = type_per_sample[type_per_sample > 1].index.tolist()
samp = img.groupby("sample_id").agg(type=("type", "first"),
                                    n_images=("file", "size"),
                                    n_trays=("tray", "nunique"),
                                    states=("state", lambda s: "+".join(sorted(set(s)))))
samp["n_dry"] = img[img.state == "dry"].groupby("sample_id").size().reindex(samp.index, fill_value=0)
samp["n_moist"] = img[img.state == "moist"].groupby("sample_id").size().reindex(samp.index, fill_value=0)
samp["has_calibration"] = samp.index.isin(cal_ids)
R["samples_by_type"] = samp.type.value_counts().to_dict()
R["images_by_type"] = img.type.value_counts().to_dict()
R["images_by_type_state"] = {f"{k[0]}/{k[1]}": int(v) for k, v in img.groupby(["type", "state"]).size().items()}
R["states_by_sample"] = samp.states.value_counts().to_dict()
R["trays_per_sample_by_type"] = {t: samp[samp.type == t].n_trays.value_counts().sort_index().to_dict()
                                 for t in samp.type.unique()}
R["images_per_sample"] = samp.n_images.describe().round(1).to_dict()
ipt = img.groupby(["sample_id", "state", "tray"]).size()
R["images_per_tray_state"] = ipt.value_counts().sort_index().to_dict()
R["samples_missing_calibration"] = samp.index[~samp.has_calibration].tolist()
R["calibration_without_images"] = sorted(set(cal_ids) - set(samp.index))

# Tray reuse across samples (potential confound)
tray_samples = img.groupby("tray").sample_id.nunique()
R["n_distinct_trays"] = int(tray_samples.size)
R["trays_used_by_n_samples"] = tray_samples.describe().round(1).to_dict()
# within a sample: are dry and moist photos of the same trays?
same_trays = []
for sid, g in img.groupby("sample_id"):
    if g.state.nunique() == 2:
        same_trays.append(set(g[g.state == "dry"].tray) == set(g[g.state == "moist"].tray))
R["both_state_samples_same_trays_frac"] = float(np.mean(same_trays)) if same_trays else None

# ---------- 2. PSD labels ----------
csv08 = pd.read_csv(root / "metadata/psd_mass_path_80_to_0_08mm_no_ballast.csv", dtype={"sample_id": str})
csvH = pd.read_csv(root / "metadata/psd_mass_path_80_to_0_0018mm.csv", dtype={"sample_id": str})
sieves = [c for c in csv08.columns if c.startswith("%_") and c.endswith("mm")]
R["sieves_mm"] = [float(c[2:-2]) for c in sieves]
R["csv08_rows"], R["csv08_samples"] = len(csv08), csv08.sample_id.nunique()
R["csvH_rows"], R["csvH_samples"] = len(csvH), csvH.sample_id.nunique()
csv08["file"] = csv08.image_path.str.replace("photogranulometry_v1_0/", "", regex=False)
R["csv08_paths_not_in_listing"] = sorted(set(csv08.file) - set(img.file))[:20]
R["csv08_n_paths_not_in_listing"] = len(set(csv08.file) - set(img.file))
excl = sorted(set(img.sample_id) - set(csv08.sample_id))
R["samples_with_images_but_not_in_csv08"] = {s: samp.loc[s, "type"] for s in excl}
R["csv08_state_mismatch"] = int((csv08.merge(img[["file", "state"]], on="file").pipe(
    lambda d: d.dry_or_moist != d.state)).sum())

# PSD constant within sample?
nconst = csv08.groupby("sample_id")[sieves].nunique().max(axis=1)
R["samples_psd_varies_across_images"] = nconst[nconst > 1].index.tolist()
psd = csv08.groupby("sample_id")[sieves].first()
psd["type"] = samp.reindex(psd.index).type
# sanity: bounds and monotonicity (coarse -> fine should be non-increasing)
vals = psd[sieves].to_numpy(float)
R["psd_any_nan"] = int(np.isnan(vals).sum())
R["psd_out_of_bounds"] = int(((vals < 0) | (vals > 100.0001)).sum())
diffs = np.diff(vals, axis=1)
nonmono = psd.index[(diffs > 1e-6).any(axis=1)].tolist()
R["psd_non_monotonic_samples"] = nonmono
R["psd_non_monotonic_detail"] = {s: [(sieves[j], sieves[j + 1], round(float(diffs[i, j]), 3))
                                     for j in np.where(diffs[i] > 1e-6)[0]]
                                 for i, s in enumerate(psd.index) if s in nonmono}
R["psd_top_not_100"] = psd.index[psd["%_80mm"] < 99.999].tolist()

# Fractions (BNQ sieves: 5 mm gravel/sand, 0.08 mm sand/fines)
frac = pd.DataFrame(index=psd.index)
frac["gravel"] = 100 - psd["%_5mm"]
frac["sand"] = psd["%_5mm"] - psd["%_0.08mm"]
frac["fines"] = psd["%_0.08mm"]
frac["type"] = psd.type
frac.round(3).to_csv(out / "sample_fractions.csv")
R["fractions_summary"] = frac[["gravel", "sand", "fines"]].describe().round(2).to_dict()
R["fractions_by_type_median"] = frac.groupby("type")[["gravel", "sand", "fines"]].median().round(1).to_dict()

# USCS-like coarse grouping (approximate: uses 5 mm / 0.08 mm instead of 4.75 / 0.075, no plasticity)
def group(r):
    if r.fines >= 50:
        return "fine-grained (>=50% fines)"
    major = "gravel" if r.gravel > r.sand else "sand"
    f = "clean (<5% fines)" if r.fines < 5 else ("dual (5-12% fines)" if r.fines <= 12 else "with fines (>12%)")
    return f"{major}, {f}"
frac["coarse_group"] = frac.apply(group, axis=1)
R["coarse_groups"] = frac.coarse_group.value_counts().to_dict()
R["coarse_groups_by_type"] = pd.crosstab(frac.coarse_group, frac.type).to_dict()

# Water content of moist images
moist = csv08[csv08.dry_or_moist == "moist"]
R["water_pct_moist_summary"] = moist.groupby("sample_id")["%_water"].first().describe().round(2).to_dict()
R["water_varies_within_sample"] = int((moist.groupby("sample_id")["%_water"].nunique() > 1).sum())

# Hydrometer subset
R["hydrometer_samples_by_type"] = samp.reindex(csvH.sample_id.unique()).type.value_counts().to_dict()

# ---------- 3. Excel overview cross-check ----------
wb = openpyxl.load_workbook(root / "metadata/psd_mass_overview_v1_0.xlsx", data_only=True)
ws = wb["BNQ"]
rows = list(ws.iter_rows(values_only=True))
hdr = list(rows[0])
xl = pd.DataFrame(rows[1:], columns=hdr)
xl = xl[xl.num_publication.notna()].copy()
xl["num_publication"] = xl.num_publication.astype(str).str.zfill(4)
xl = xl.set_index("num_publication")
R["xlsx_bnq_rows"] = len(xl)
R["xlsx_sheets"] = wb.sheetnames
R["xlsx_asth_samples"] = [str(r[0]) for r in list(wb["ASTM"].iter_rows(values_only=True))[1:] if r[0]]
R["xlsx_type_mismatch_vs_filenames"] = [s for s in xl.index if s in samp.index and xl.loc[s, "type"] != samp.loc[s, "type"]]
tr = pd.to_numeric(xl["trays"], errors="coerce")
R["xlsx_trays_mismatch"] = [s for s in xl.index if s in samp.index and tr[s] != samp.loc[s, "n_trays"]]
common = psd.index.intersection(xl.index)
d = (xl.loc[common, sieves].apply(pd.to_numeric, errors="coerce") - psd.loc[common, sieves]).abs()
R["xlsx_vs_csv_max_abs_diff"] = round(float(np.nanmax(d.to_numpy())), 4)

# Highlighted (interpolated / extrapolated) cells: count yellow fills per column (sieve columns)
yellow_cols = Counter()
red_text = Counter()
col_idx = {h: i for i, h in enumerate(hdr)}
for row in ws.iter_rows(min_row=2):
    for h in sieves:
        c = row[col_idx[h]]
        fill = c.fill.fgColor.rgb if c.fill and c.fill.fgColor is not None else None
        if isinstance(fill, str) and fill.upper().endswith("FFFF00"):
            yellow_cols[h] += 1
            if c.font and c.font.color is not None and isinstance(c.font.color.rgb, str) and c.font.color.rgb.upper().endswith("FF0000"):
                red_text[h] += 1
R["xlsx_yellow_cells_per_sieve"] = dict(yellow_cols)
R["xlsx_red_text_cells_per_sieve"] = dict(red_text)

samp = samp.join(frac[["gravel", "sand", "fines", "coarse_group"]])
samp["in_csv08"] = samp.index.isin(csv08.sample_id)
samp["has_hydrometer"] = samp.index.isin(csvH.sample_id)
samp.to_csv(out / "sample_table.csv")

(out / "audit_results.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps(R, indent=1, default=str))
