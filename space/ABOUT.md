## How it works

1. **Find the soil.** The tray is detected and masked; 12 full-resolution 768 px crops (~30 mm) and an overview are taken from inside it.
2. **Describe it.** Frozen pretrained vision models (DINOv3, Meta) turn the crops and overview into feature vectors. Nothing is fine-tuned.
3. **Estimate.** A small ridge regression, trained on ~250 lab-tested samples, predicts the gravel, sand and fines fractions.
4. **Say how sure it is.** A conformal range (calibrated on held-out samples) widens when a sample's photos disagree. Wide ranges, failed photo checks, or photos unlike the training data trigger **Review needed**.
5. **Show the evidence.** The five most similar lab-tested samples, with their real sieve curves.
6. **Show where.** The heatmap applies the same linear model to each 16 px patch of the overview; the patches average back to the whole-photo estimate.

## Limits

- Visual estimate only: no plasticity information, so it is not an engineering soil classification.
- Trained on soils from the Montréal region photographed with one lab setup at ~40 µm/px. Other cameras, distances or soils are outside what it has learned.
- Fractions use 5 mm (gravel/sand) and 0.08 mm (sand/fines) boundaries, as in the BNQ sieve series.

## Data

Plante St-Cyr, T., Duhaime, F., Dubé, J.-S., & Grenier, S. (2025). *Photogranulometry – Dataset of soil images with
corresponding particle size distributions* (v1.0). Federated Research Data Repository. doi:10.20383/103.01316. CC BY 4.0.
