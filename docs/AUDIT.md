# Photogranulometry v1.0: dataset audit

Dataset: Plante St-Cyr, Duhaime, Dubé & Grenier (2025), *Photogranulometry* v1.0, FRDR, doi:10.20383/103.01316.
Licence: **CC BY 4.0** (LICENSE.txt). Commercial and research use allowed with attribution.
This audit used only the metadata files, the checksum listing and 3 sample images. No full-dataset download was needed.

## 1. What's in it

| | Count |
|---|---|
| Soil images | 12,714 (all filenames parse cleanly, no duplicates) |
| Physical samples | 321 (every sample has exactly one calibration image) |
| Total size | 441.5 GB (434.2 GB soil images, ~7 GB calibration) |
| Image size | 8192 × 5464 px, ~39.4 µm/px |

| Sample type | Samples | Images | Trays per sample |
|---|---|---|---|
| bulk | 144 | 8,925 | 4 (141), 3 (2), 5 (1) |
| splitspoon | 152 | 1,999 | 1 (150), 2 (2) |
| artificial | 20 | 1,601 | 4 (19), 8 (1) |
| ballast | 5 | 189 | 11–15 (random division) |

- Moisture: 287 samples have both dry and moist photos, 20 are moist only and 14 are dry only.
- Images per sample range from 5 to 84, with a median of 24. Most trays have 5 or 10 photos per moisture state.
- In 99% of dual-state samples, the dry and moist photos come from the same physical trays.

## 2. Labels

- `psd_mass_path_80_to_0_08mm_no_ballast.csv` covers **311 samples / 12,460 images**, with percent passing on 16 sieves from 80 mm to 0.08 mm.
- The particle-size curve is identical for every image of a sample. **The label belongs to the sample, not the image.**
- The CSV matches the Excel overview to within 0.005 percentage points, and every CSV image path exists in the full listing.

**Samples with images but no sieve label (10). Excluded from supervised training:**
- 5 ballast samples (0317–0321). They were excluded by the dataset authors because of their random division.
- 5 split-spoon samples (0239, 0256, 0260, 0262, 0267) have no sieve data. Four of them (all except 0239) have hydrometer values below 0.05 mm only.
- These samples can still be used for self-supervised pretraining.

**Interpolated sieve columns** (highlighted yellow in the Excel file):

| Sieve | Interpolated rows (of 316) |
|---|---|
| 31.5 mm | 286 |
| 2 mm | 172 |
| 28 mm | 25 |

- These are mostly not measured. Don't treat them as ground truth when scoring a full particle-size curve.
- **5 mm and 0.08 mm are fully measured**, so the gravel/sand/fines fractions rest on real sieve data.

**Other notes:**
- Samples 0312–0316 were tested to ASTM. Their values on the BNQ sieves (the Québec standard the rest of the dataset uses) are interpolated.
- 4 samples (0238, 0251, 0252, 0257) have tiny non-monotonic steps of 0.05–0.26 pp between 5 mm and 2.5 mm. Fix with a running minimum before using the full curve.
- Hydrometer data (fine fractions below 0.08 mm) exists for 77 samples: 42 split-spoon and 35 bulk, none artificial.

## 3. Target distribution (311 labelled samples)

Fractions use BNQ boundaries: gravel = 100 − %passing(5 mm), fines = %passing(0.08 mm).

| | Mean | Median | IQR | Range |
|---|---|---|---|---|
| Gravel % | 18.0 | 13.5 | 2.7–24.8 | 0–93.2 |
| Sand % | 54.3 | 51.8 | 42.1–69.5 | 0–98.9 |
| Fines % | 27.7 | 24.5 | 9.4–36.9 | 0.1–100 |

**Approximate coarse groups.** These use 5 mm and 0.08 mm in place of the USCS 4.75 mm and 0.075 mm, and include no plasticity, so they are not real USCS classes:

| Group | Samples |
|---|---|
| Sand with >12% fines | 167 |
| Fine-grained (≥50% fines) | 43 |
| Sand, 5–12% fines | 38 |
| Clean sand (<5% fines) | 33 |
| Clean gravel | 15 |
| Gravel with >12% fines | 8 |
| Gravel, 5–12% fines | 7 |

## 4. Risks for modelling

1. **Sample type is confounded with the target.** All 15 clean-gravel samples are artificial, and artificial samples have a median of 75% gravel and 1.8% fines. A gravel classifier could learn "artificial-looking" instead of "gravel". Only 15 natural samples are gravel-dominant. Report metrics on natural samples (bulk + split-spoon) separately, and run one experiment with artificial samples removed.
2. **Image counts are unbalanced across samples.** Bulk samples contribute 72% of labelled images but 46% of samples. Train with per-sample weighting or per-sample batch sampling, and score at sample level by averaging image predictions within a sample. Otherwise bulk soils dominate both training and metrics.
3. **Effective n is about 311, not 12k.** Photos of the same tray are near-replicates. All grouping is by `sample_id` (done in the split below).
4. **Fine-grained soils saturate the sieve curve.** 43 samples are ≥50% fines and 5 are ≥95%. For these, the 16 sieve values say almost nothing beyond "mostly fines". The finest particles are ~2 px wide at 39.4 µm/px, so fines are inferred from colour and texture, not resolved. Report error per fraction and per fines bin.
5. **Trays are reused, which is good for leakage.** There are 91 physical trays and each is used by a median of 10 different samples, so a model can't recover a label from a tray ID. Crops should still exclude the white tray rim, which is visible in every photo.
6. **Moisture is a real domain shift.** Water content of moist samples ranges from 0.6% to 52% (median 13%). It's constant within a sample in the metadata. The 287 dual-state samples support a clean dry→moist transfer test.

## 5. Calibration images

Each calibration image shows two 50 mm markers (ID 1 on the tray floor, ID 2 raised on a block) plus a steel ruler. These look like ArUco markers, which OpenCV can detect to compute µm/px automatically for each sample. That lets crops cover the same physical area across samples (needed for Enhancement A). Worth checking how much the scale actually varies before investing in it.

## 6. Split (`split_samples_seed0.csv`, `split_images_seed0.csv`)

- **Unit:** sample. Every image, tray and moisture state of a sample stays in one split (asserted in code).
- **Stratified by:** sample type × fines bin (<5, 5–12, 12–30, 30–50, ≥50) × gravel-dominant.
- **Seed:** 0, fixed.

| Split | Samples | Images | Use |
|---|---|---|---|
| dev | 199 | 7,987 | training and model selection, via 5 CV folds (40/40/40/40/39) |
| calib | 49 | 2,008 | temperature scaling and conformal calibration only |
| test | 63 | 2,465 | **locked**; touch once per final model |

| Breakdown | dev | calib | test |
|---|---|---|---|
| artificial | 13 | 3 | 4 |
| bulk | 90 | 24 | 30 |
| splitspoon | 96 | 22 | 29 |
| hydrometer | 41 | 15 | 17 |
| mean gravel % | 17.2 | 17.2 | 21.1 |
| mean sand % | 54.6 | 54.8 | 53.1 |
| mean fines % | 28.2 | 28.0 | 25.8 |

The test set is slightly gravel-heavier (7 of 63 samples are gravel-dominant, 3 of them artificial). This is acceptable, but it's another reason to report natural-only metrics.

## 7. Recommended first target

Gravel / sand / fines fractions, using a softmax head trained at sample level. Both boundaries are fully measured, the target is low-dimensional, and it maps onto how labs describe soils. The full particle-size curve (dropping or down-weighting the interpolated 31.5/28/2 mm sieves) can come second.

## Files

| File | Contents |
|---|---|
| `scripts/audit.py` | Reproduces all numbers here: `python3 audit.py <dataset_root> <out_dir>` |
| `scripts/make_split.py` | Regenerates the split: `python3 make_split.py <dataset_root> <out_dir> 0` |
| `image_index_all.csv` | All 12,714 images: sample, type, state, tray, index |
| `sample_table.csv` | Per-sample counts, states, trays, fractions, label availability |
| `sample_fractions.csv` | Gravel/sand/fines per labelled sample |
| `split_samples_seed0.csv` | One row per labelled sample with split and fold |
| `split_images_seed0.csv` | One row per labelled image with full sieve curve, split and fold. This is the training manifest. |
| `audit_results.json` | Raw audit output |
