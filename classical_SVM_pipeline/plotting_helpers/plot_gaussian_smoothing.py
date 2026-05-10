"""
Figure: Effect of Gaussian smoothing on a representative SR_1005 spectrum.
Left panel:  standardised spectrum (sigma=0, no smoothing)
Right panel: smoothed spectrum (sigma=15, optimal CV choice for H_vs_PC)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.abspath('/home/anushkasingh/Desktop/Thesis/Code/src'))
sys.path.append(os.path.abspath('/home/anushkasingh/Desktop/Thesis/Code/classical_SVM_pipeline'))

from load_data import read_data
from baseline_correct import baseline_roy
from sr_preprocessing import extract_sr_window, preprocess_sr

# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_IDX   = 1      # allkg sample index to use as representative (PC patient)
SIGMA_OPT    = 15         # optimal sigma* from 5-fold CV for SR_1005 / H_vs_PC
SR_CENTER    = 1005.0
SR_WIDTH     = 30.0
NORM_VP      = [504, 425, 451, 454, 450, 474, 451, 471, 540, 467,
                550, 468, 481, 450, 515, 441, 452, 462, 453, 450,
                452, 490, 504, 520, 525, 498, 542, 527, 550]
OUT_PATH     = '/home/anushkasingh/Desktop/Thesis/Images/gaussian_smoothing_example.png'

# ── Load one representative spectrum ───────────────────────────────────────────
data_path  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../ALLDataGross/allKgData')
)
dataframes = read_data(data_path)
filename, df = dataframes[SAMPLE_IDX]
x = df['Wavenumber'].values
y = df['Intensity'].values

# ── Baseline correction ─────────────────────────────────────────────────────────
y_bc, _, _ = baseline_roy(x, y, norm_factor_i=NORM_VP[SAMPLE_IDX])

# ── Extract SR_1005 window ──────────────────────────────────────────────────────
sr_spec, sr_wn = extract_sr_window(y_bc, x, center=SR_CENTER, window_width=SR_WIDTH)

# ── Standardise (mean-centre + unit-variance) ───────────────────────────────────
result       = preprocess_sr(sr_spec)
std_spec     = result['preprocessed']           # sigma = 0
smooth_spec  = gaussian_filter1d(std_spec, SIGMA_OPT)   # sigma = 15

# ── Plot ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)

for ax, spec, title in zip(
    axes,
    [std_spec, smooth_spec],
    [r'Raw standardised ($\sigma = 0$)', rf'Smoothed ($\sigma^* = {SIGMA_OPT}$)'],
):
    ax.plot(sr_wn, spec, color='steelblue', linewidth=1.4)
    ax.set_xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')
    ax.set_xlim(sr_wn.min(), sr_wn.max())
    ax.tick_params(labelsize=9)

axes[0].set_ylabel('Standardised absorbance', fontsize=11)

fig.suptitle(
    r'Effect of Gaussian smoothing on SR$_{1005}$ (990–1020 cm$^{-1}$)',
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT_PATH}')
plt.show()
