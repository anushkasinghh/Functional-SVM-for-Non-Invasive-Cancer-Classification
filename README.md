## Thesis Code Overview

This repository contains code and data pipelines for my thesis on **infrared breath spectra**, focusing on:

- **Baseline correction** of raw spectra.
- **Baseline SVM model** for prediction of urogenital cancers.
- **Functional SVM** for class prediction.
- **Baseline-invariant representations** learned with contrastive learning.

### Top-level layout

- `ALLDataGross/` – raw / intermediate spectral datasets. This folder is lignored by git.
- `src/` – scripts and notebooks for classical baseline correction and analysis:
  - `load_data.py` – utilities to load `.dpt` spectra into pandas DataFrames.
  - `baseline_correct.py` – baseline correction routines.
  - `*.ipynb` – class prediction models and other data related experiments (PCA, ANOVA, SVM, FSVM, pipelines, plots).
- `exploratory_BaselineInvariance/` – PyTorch code for learning **baseline-invariant encoders**:
  - `models.py` – 1D CNN encoder + NT-Xent loss.
  - `augment.py` – spectral augmentations (baseline, noise, etc.).
  - `data_loader.py`, `train.py`, `evaluate.py`, `utils.py` – training and evaluation helpers.
- `exploratory_notebooks/` – higher-level experiment notebooks tying pieces together.
- `InfoDump/` – markdown notes describing ideas, background, and future directions (ignored by git).
- `thesis/` – local virtual environment (ignored by git).
- `requirements.txt` – minimal Python dependencies.

