# Urogenital Cancer Detection from Breath IR Spectra

Replication and extension of [Maiti et al. 2021](https://doi.org/10.1021/acs.analchem.0c04761) — binary classification of urogenital cancers (kidney, prostate, bladder) from infrared breath spectroscopy using classical SVM, functional SVM (FSVC), and contrastive representation learning.

---

## Results

**LOOCV on training set (replication of Table 1, Maiti 2021):**

| Task    | Accuracy | Sensitivity | Specificity | Kernel | C    |
|---------|----------|-------------|-------------|--------|------|
| H vs KC | 83.3%    | 94.0%       | 80.0%       | RBF    | 0.01 |
| H vs PC | 83.5%    | 78.6%       | 82.8%       | RBF    | 100  |
| H vs BC | 77.5%    | —           | —           | Linear | 1    |

**Blind set evaluation (Table 2):**

| Task    | Balanced Accuracy | Confusion Matrix       |
|---------|-------------------|------------------------|
| H vs KC | 0.80              | [[4, 0], [2, 3]]       |
| H vs BC | 0.625             | [[1, 3], [0, 3]]       |

---

## Repository Structure

```
.
├── src/
│   ├── baseline_correct.py          # Multi-stage baseline correction (1st/2nd/3rd order)
│   └── BaselineCorrect_pipeline.ipynb
│
├── classical_SVM_pipeline/
│   ├── SVM_implement.py             # SVMBreathClassifier: LOOCV, k-fold, blind eval
│   ├── grid_search.py               # Nested CV over 72 hyperparameter configs
│   ├── sr_preprocessing.py          # Extract & normalise spectral range windows
│   ├── SVM_notebook.ipynb
│   └── eval_result_data/            # Output CSVs (git-ignored)
│
├── FSVC/
│   ├── fsvm_implement.py            # FPCA via R (rpy2) + SVM on FPC scores
│   ├── FSVM_notebook.ipynb
│   └── eval_result_data/            # Output CSVs (git-ignored)
│
├── exploratory_notebooks/
│   └── exploratory_BaselineInvariance/
│       ├── models.py                # 1D CNN encoder + NT-Xent contrastive loss
│       ├── train.py
│       ├── augment.py               # Spectral augmentations (baseline, fringe, noise)
│       └── evaluate.py
│
├── ALLDataGross/                    # Raw .dpt spectra — git-ignored
├── data_processed/                  # Cached preprocessed data — git-ignored
└── requirements.txt
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**R dependency** (required for FSVC only):

```r
install.packages("refund")
```

---

## Pipeline

### 1. Baseline Correction
Run `src/BaselineCorrect_pipeline.ipynb`.

Applies a 3-stage correction to raw `.dpt` spectra:
1. Normalise by sample-specific reference absorbance
2. Uniform shift using a quiet reference region (2550–2600 cm⁻¹)
3. Segment-wise polynomial fitting across 3 spectral windows

Outputs cached to `data_processed/dataprocessedbreath_data.pkl`.

### 2. Classical SVM
Run `classical_SVM_pipeline/SVM_notebook.ipynb`.

- Features: 8 spectral range (SR) windows ± 15 cm⁻¹, optionally reduced with PCA (4 components)
- Hyperparameter search: nested CV over kernel, C, gamma, Gaussian smoothing sigma
- Evaluation: LOOCV (primary), 9-fold × 10 repeats, blind set

### 3. Functional SVM (FSVC)
Run `FSVC/FSVM_notebook.ipynb`.

- FPCA via `refund::fpca.face()` (FACE algorithm, Xiao et al. 2016)
- FPC scores estimated via BLUP with noise shrinkage
- Joint tuning of smoothing parameter τ, number of components K, SVM C and γ

### 4. Baseline-Invariant Encoder (Exploratory)
Run `exploratory_notebooks/baseline_invariant_contrastive_encoder.ipynb`.

- 1D CNN trained with NT-Xent contrastive loss
- Augmentations simulate baseline variability (polynomial drift, fringes, scaling, noise)
- Evaluated via UMAP embeddings and positive/negative pair distance metrics

---

## Data Format

Raw spectra are stored as `.dpt` files (comma or whitespace-delimited wavenumber/intensity pairs, ~14,500 points per spectrum). Three cohorts:

| Cohort         | Path                       | Notes              |
|----------------|----------------------------|--------------------|
| Healthy (H)    | `ALLDataGross/healthyCohort/` | 22 subjects     |
| Cancer (KC/PC/BC) | `ALLDataGross/allKgData/` | Multi-class      |
| Blind test set | `ALLDataGross/BlindData/`  | Held out          |

---

## References

- Maiti et al. (2021). *Breath Analysis Using IR Spectroscopy for Urogenital Cancer Detection.* Analytical Chemistry.
- Xie & Ogden (2024). *Functional Support Vector Machine.* Biostatistics, 25(4):1178–1194.
- Xiao et al. (2016). *Fast Covariance Estimation for Sparse Functional Data.* Statistics and Computing.
