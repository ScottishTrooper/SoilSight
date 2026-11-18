<div align="center">

# 🪨 SoilSight

**Soil grain-size estimation from lab tray photos, with calibrated uncertainty, evidence and heatmaps**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![DINOv3](https://img.shields.io/badge/backbone-DINOv3-0467DF?logo=meta&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-F97316?logo=gradio&logoColor=white)
![Kaggle](https://img.shields.io/badge/runs%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/code-MIT-green)
![Data](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)

**[🔗 Live showcase: 63 held-out test samples next to their lab results](https://huggingface.co/spaces/petrichorgreen/soilsight-showcase)**

</div>

![Showcase: estimate vs lab result for a held-out sample](docs/images/showcase_sample.png)

## 🎯 Why

Geotechnical labs measure a soil's grain-size distribution with sieve tests that take hours per sample. SoilSight gives a first estimate from photos in seconds, so technicians can triage samples, catch inconsistencies early and keep a traceable record. It is built to be honest about its limits: every estimate comes with a calibrated range, uncertain samples are flagged for review, and photos outside its training conditions are flagged rather than measured.

## 📊 Results at a glance

| | Metric | Value |
|---|---|---|
| 🎯 | Average error on 63 held-out samples | **4.8 pp** |
| 📉 | Baseline: always guess the training average | 16.0 pp (**−70%** error) |
| 📏 | 90% ranges containing the lab value | **94–95%** per fraction |
| ✅ | Auto-accepted by the review flag | 43% of samples |
| 🛡️ | Out-of-scope photos (e.g. web images) | caught by up to **4** independent warnings |

<sub>pp = percentage points. Test split evaluated once, after all modelling choices were fixed.</sub>

## 🧪 Detailed results

### Held-out test split (63 samples)

| Fraction | Mean absolute error | 90% range coverage |
|---|---|---|
| 🟤 Gravel (> 5 mm) | 4.7 pp | 94% |
| 🟡 Sand (0.08–5 mm) | 4.9 pp | 95% |
| 🔵 Fines (< 0.08 mm) | 4.9 pp | 95% |
| **All three** | **4.8 pp** | **87%** (all inside at once) |
| Natural soils only | 5.0 pp | |

### Model comparison (5-fold cross-validation, 199 development samples)

| Features | Mean error (pp) | vs baseline |
|---|---|---|
| Always predict the average | 16.9 | |
| 🎨 Hand-made colour / texture features | 12.2 | −28% |
| 🖼️ EfficientNet-B0 (ImageNet) | 6.7 | −60% |
| 🦖 DINOv3 ConvNeXt-T | 6.2 | −63% |
| 🦖 DINOv3 ViT-S/16 | 6.1 | −64% |
| ⭐ **DINOv3 ConvNeXt-T + ViT-S (used)** | **5.8** | **−66%** |
| 🔥 ViT-S patch head (powers the heatmap) | 7.0 | −59% |

<sub>Test error (4.8) is lower than cross-validation (5.8); with 63 test samples that gap is within sampling noise, so about 5–6 pp is the honest expectation.</sub>

### Review flag: accuracy vs workload

Auto-accepting only the most confident samples trades coverage for accuracy. Calibration split (49 samples):

| Auto-accept | Widest range accepted | Error on accepted | Error on flagged |
|---|---|---|---|
| 25% most confident | ±12.7 pp | **3.6 pp** | 8.8 pp |
| 50% (setting used) | ±15.2 pp | **4.2 pp** | 10.7 pp |
| 75% | ±20.2 pp | 5.7 pp | 13.0 pp |
| 100% (no flag) | ±26.1 pp | 7.5 pp | |

On the test split the flag accepted 43% of samples with 4.3 pp error vs 5.3 pp for flagged ones: a useful triage signal, weaker than on calibration.

## ✨ What it shows for each sample

| | Feature | Description |
|---|---|---|
| 📈 | **Estimate** | gravel / sand / fines percentages with calibrated 90% ranges |
| ⚠️ | **Review flag** | "Auto-accepted" or "Review needed", with the reasons |
| 🔍 | **Evidence** | the 5 most similar lab-tested samples, with their real sieve curves plotted against the estimate |
| 🔥 | **Heatmap** | per-patch estimate of each fraction across the tray |
| 📝 | **Technician log** | confirm or correct, searchable, CSV export (live app) |
| 🛡️ | **Photo checks** | image scale, tray detection, sharpness, exposure, "unlike the training data" |

![Evidence view: similar lab-tested samples and their sieve curves](docs/images/showcase_evidence.png)

## ⚙️ How it works

1. 🧭 **Find the soil.** The white tray is masked out; 12 full-resolution 768 px crops (~30 mm at ~40 µm/px) and an overview are taken from inside the soil area.
2. 🦖 **Describe it.** Frozen DINOv3 backbones (ConvNeXt-T and ViT-S/16) embed the crops and the overview. Nothing is fine-tuned: with ~250 labelled samples, frozen features with a small head generalise well and make every experiment take seconds.
3. 📐 **Estimate.** A ridge regression predicts the three fractions per photo, each photo weighted so every *sample* counts equally; a sample's photos are averaged.
4. 📏 **Say how sure it is.** A normalised split-conformal range, scaled by how much the 10 most similar lab samples disagree with each other. The first version used disagreement between a sample's own photos; it flagged 96% of samples and did not track error, so five candidate signals were compared on out-of-fold errors and the best kept.
5. 🔍 **Show the evidence.** The five most similar lab-tested samples (cosine similarity of DINOv3 embeddings) with their real sieve curves.
6. 🔥 **Show where.** A linear head on ViT patch tokens gives every 16 px patch its own estimate. Because the head is linear, the patch estimates average back exactly to the photo's estimate.
7. 🛡️ **Refuse gracefully.** Photos without a tray, at the wrong scale, or unlike the training data trigger independent warnings instead of a confident number.

## 🗄️ Data pipeline

| | |
|---|---|
| 📦 Source dataset | [Photogranulometry v1.0](https://doi.org/10.20383/103.01316): 321 soil samples, 12,714 photos at 45 MP, **441 GB** |
| 🌊 Streamed | 3,062 photos (2 per tray and moisture state) + 311 calibration photos, ~111 GB, in 32 resumable batches |
| ✅ Integrity | every file verified against FRDR's SHA-256 checksums; QC contact sheet per batch (0 photos flagged) |
| 💾 Kept | ~30 GB of overviews, crops, masks and half-resolution copies on a private Hugging Face dataset |
| ⏱️ Time | cache build 104 min (Kaggle CPU); feature extraction 111 min (Kaggle T4); everything after: seconds |
| 📏 Scale | ~39.8 µm/px, measured from ArUco markers in the calibration photos |

All splits are by **sample**: every photo, tray and moisture state of a sample stays in one split.

| Split | Samples | Photos | Used for |
|---|---|---|---|
| 🛠️ Development | 199 | 1,928 | training and model selection (5 folds) |
| 📏 Calibration | 49 | 504 | uncertainty ranges and review threshold |
| 🔒 Test | 63 | 630 | final evaluation, opened once |

The dataset audit (label coverage, confounds, interpolated sieves) is in [`docs/AUDIT.md`](docs/AUDIT.md).

## 🧰 Tech stack

| Area | Tools |
|---|---|
| Vision models | DINOv3 ConvNeXt-T & ViT-S/16 (Hugging Face Transformers), EfficientNet-B0 (timm), PyTorch |
| Modelling | NumPy ridge regression, grouped cross-validation, normalised split-conformal prediction, kNN retrieval |
| Images | OpenCV (tray masking, ArUco scale), Pillow |
| Data & compute | Hugging Face Hub (private datasets and models), Kaggle notebooks (CPU + T4) |
| Apps | Gradio (live app), static HTML/CSS/JS showcase on Hugging Face Spaces |
| Quality | pytest unit tests on synthetic data, GitHub Actions |

## 📁 Repository layout

```
soilsight/      core package
  scale.py        image scale from the ArUco calibration markers
  views.py        soil mask, overview and crop placement
  cache.py        resumable download → verify → views → upload pipeline
  features.py     frozen backbones + handcrafted features
  heads.py        ridge heads, grouped CV, conformal ranges, retrieval, difficulty signals
  predict.py      inference used by both the app and the showcase
space/          Gradio app (upload photos, evidence, heatmap, technician log)
site/           static showcase page (HTML/CSS/JS reading data.json)
notebooks/      Kaggle notebooks, generated by scripts/make_*_notebook.py
scripts/        dataset audit, split, download manifest, notebook generators
data/           sample-level split, download manifests, per-sample lab curves
tests/          unit tests on synthetic data (run in CI)
docs/           dataset audit, development log, images
```

## 🚀 Reproduce

Runs on free Kaggle notebooks with a Hugging Face account (write token as the Kaggle secret `HF_TOKEN`; accept the DINOv3 licence on the Hub first).

| Step | Notebook | Hardware | Time |
|---|---|---|---|
| 1 | `01_build_cache.ipynb`: stream, verify and cache the photos | CPU | ~1.75 h |
| 2 | `02_train.ipynb`: features, model comparison, heads | GPU T4 | ~1.9 h (reruns: seconds) |
| 3 | `02b_uncertainty.ipynb`: choose and calibrate the uncertainty signal | CPU | ~1 min |
| 4 | `04_test_and_showcase.ipynb`: open the test split once, publish the showcase | CPU | ~10 min |
| 5 | `05_live_demo.ipynb`: temporary public link to the full app | CPU or T4 | on demand |

<sub>`03_deploy_space.ipynb` deploys the app as a permanent Gradio Space, which now needs Hugging Face PRO.</sub>

Regenerate a notebook after changing the package with `python scripts/make_train_notebook.py .` (and similarly for the others). Run the tests with `pip install -r requirements-dev.txt && pytest`.

## ⚠️ Limits

- 🧪 **Visual estimate only.** There is no plasticity information, so this is not an engineering soil classification; lab tests are still required.
- 📷 **One lab setup.** Trained on Montréal-region soils photographed at ~40 µm/px. Other cameras, distances and soils are outside what the model has learned; arbitrary web photos are flagged, not measured.
- 🚦 **The flag is triage, not a guarantee.** It separated errors strongly on calibration samples (4.2 vs 10.7 pp) but only modestly on the test split (4.3 vs 5.3 pp).
- 📐 Fractions follow the BNQ sieve boundaries (gravel > 5 mm, fines < 0.08 mm).

## 🙏 Credits

- **Data:** Plante St-Cyr, T., Duhaime, F., Dubé, J.-S. & Grenier, S. (2025). *Photogranulometry – Dataset of soil images with corresponding particle size distributions* (v1.0). Federated Research Data Repository. doi:10.20383/103.01316. CC BY 4.0. Images in `docs/images` are derived from it.
- **Backbones:** DINOv3 (Siméoni et al., 2025, Meta AI; DINOv3 License) and EfficientNet-B0 via `timm`. Model weights are not redistributed here.
- **Code:** MIT licence (see `LICENSE`).
