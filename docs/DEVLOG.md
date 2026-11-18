# SoilSight development log (status and results as the project progressed)

Estimate a soil sample's gravel / sand / fines fractions from photos of a lab tray, with an uncertainty range,
a "review needed" flag, the most similar lab-tested samples as evidence, and a grain-size heatmap.
Built on the Photogranulometry v1.0 dataset (Plante St-Cyr et al., 2025; doi:10.20383/103.01316; CC BY 4.0).

> Work in progress. Visual estimate only: plasticity and engineering classification still need lab tests.

## Status

| Stage | State |
|---|---|
| Dataset audit and sample-level split | done (`docs/AUDIT.md`, `data/split_*`) |
| Download manifest, scale from calibration markers, tray cropper | done, tested on sample photos |
| Kaggle cache builder (`notebooks/01_build_cache.ipynb`) | ready to run |
| Features + model comparison, conformal flag, retrieval index, heatmap head (`notebooks/02_train.ipynb`) | ready, tested offline on synthetic data |
| Static showcase of held-out test samples (`site/`, `notebooks/04_test_and_showcase.ipynb`) | published: huggingface.co/spaces/petrichorgreen/soilsight-showcase |
| Live Gradio app (`space/`): temporary public link from Kaggle (`notebooks/05_live_demo.ipynb`) | ready, tested offline |
| Permanent Gradio Space (`notebooks/03_deploy_space.ipynb`) | needs Hugging Face PRO; not used |

## Layout

```
soilsight/        package: scale.py (ArUco µm/px), views.py (soil mask, overview, crops), cache.py (batch pipeline),
                  features.py (frozen backbones + handcrafted), heads.py (ridge heads, CV, conformal, retrieval)
scripts/          audit.py, make_split.py, make_manifest.py, make_cache_notebook.py
notebooks/        01 cache (CPU) · 02 train (GPU) · 02b uncertainty (CPU) · 03 Gradio Space (PRO only)
                  · 04 test results + static showcase (CPU) · 05 live demo link (CPU/GPU), all for Kaggle
site/             static showcase page (index.html, style.css, app.js) reading data.json
space/            Gradio app (app.py), Space card, About page; soilsight/predict.py is its inference code
data/             split_samples_seed0.csv, split_images_seed0.csv, manifest_k2.csv, manifest_k3.csv
docs/             AUDIT.md
```

## Data pipeline

1. `make_manifest.py` picks 2 photos per sample/moisture-state/tray (3,062 photos + 311 calibration photos, ~111 GB),
   with SHA-256 checksums from FRDR, grouped into 32 batches of 10 samples.
2. The Kaggle notebook streams each batch from FRDR, verifies checksums, and writes per photo: a 1536 px overview of
   the soil area, 12 full-resolution 768×768 crops (~30 mm), a soil mask, and a half-resolution copy.
   Batches are uploaded to a private Hugging Face dataset and the originals deleted. Reruns skip finished batches.
3. Calibration photos give the scale (~39.8 µm/px) from two 50 mm ArUco markers.

## Model approach

Frozen backbones (EfficientNet-B0 as the ImageNet baseline, DINOv3 ConvNeXt-T and ViT-S/16) are run once over every
cached photo; small ridge heads then predict the three fractions. The same embeddings drive the similar-sample index,
and a linear head on ViT patch tokens gives a per-patch estimate whose average equals the whole-photo prediction (the heatmap).
Evaluation is per sample with 5-fold CV grouped by sample; intervals are normalised split-conformal, widened when a
sample's photos disagree; samples with wide intervals are flagged for review. The test split is used once, at the end.

## Results so far (5-fold CV on 199 dev samples, mean absolute error, percentage points)

| Features | Mean error |
|---|---|
| Predict the dev average (baseline) | 16.9 |
| Handcrafted colour/texture | 12.2 |
| EfficientNet-B0 (ImageNet) | 6.7 |
| DINOv3 ConvNeXt-T | 6.2 |
| DINOv3 ViT-S/16 | 6.1 |
| **DINOv3 ConvNeXt-T + ViT-S** | **5.8** |
| ViT-S patch-token head (heatmap) | 7.0 |

Calibration split (49 samples): 7.5 pp. The first review rule (photo disagreement) flagged 96% of samples and did not
track error. `02b_uncertainty.ipynb` compared five difficulty signals; the spread of the 10 most similar lab samples'
results works best: auto-accepting the most confident half gives 4.2 pp error on accepted vs 10.7 pp on flagged
calibration samples.

## Final results on the held-out test split (63 samples, evaluated once)

| Metric | Value |
|---|---|
| Mean absolute error, all three fractions | **4.8 pp** (gravel 4.7, sand 4.9, fines 4.9) |
| Same, natural soils only (excluding lab-made mixtures) | 5.0 pp |
| Baseline: always predict the training average | 16.0 pp |
| 90% ranges: share containing the lab value, per fraction | 94% gravel, 95% sand, 95% fines |
| All three fractions inside their ranges at once | 87% |
| Auto-accepted by the review flag | 43% |
| Error on auto-accepted vs flagged samples | 4.3 pp vs 5.3 pp |

Notes: the test error (4.8 pp) is lower than cross-validation (5.8 pp) and the 49-sample calibration split (7.5 pp);
with 63 test samples the difference is within normal sampling noise, so ~5–6 pp is the honest expectation.
The ranges met their 90% target. The review flag separated errors strongly on the calibration split
(4.2 vs 10.7 pp) but only modestly on test (4.3 vs 5.3 pp), so it should be read as a useful triage signal, not a guarantee.
Out-of-distribution photos (e.g. web images without scale) trigger up to four independent warnings
(image size, no tray found, unlike training data, wide range).
