"""
Section 5.7 — Functional Data Representation
Produces three figures for the best-performing config: SR_1005, H vs PC (tau*=10, K*=2).

  fig1: Per-class mean functions + individual spectra + difference curve
  fig2: First K* eigenfunctions + cumulative variance explained
  fig3: BLUP shrinkage weight w_k = λ_k / (λ_k + σ²) vs eigenvalue rank / value
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'classical_SVM_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sr_preprocessing import preprocess_all_srs
from fsvm_implement import fpca_face_via_r

OUT = '../eval_result_data/plots'
os.makedirs(OUT, exist_ok=True)

# ── Best config parameters (from Table best-config-fsvm) ──────────────────
BEST_SR     = 'SR_1005'
BEST_TASK   = 'H_vs_PC'
BEST_TAU    = 1.0      # opt_tau  (smoothing parameter λ for fpca.face)
BEST_K      = 2         # opt_K    (FPCs retained)
SR_CENTER   = 1005      # cm⁻¹
WINDOW_HW   = 15        # ±15 cm⁻¹

CLASS_COLORS = {'H': '#2196F3', 'PC': '#E53935'}
CLASS_LABELS = {'H': 'Healthy', 'PC': 'Prostate Cancer'}

# ── Load data ──────────────────────────────────────────────────────────────
df_raw = pd.read_pickle(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data_processed', 'breath_data.pkl')
)


# Replicate notebook cell 7: drop blind rows, rename M/F → H, then filter to task
df_all = df_raw[df_raw['category'] != 'blinddata'].copy()
df_all.loc[df_all['infoP'].isin(['M', 'F']), 'infoP'] = 'H'
# Filter to H vs PC
mask = df_all['infoP'].isin(['H', 'PC'])
df = df_all[mask].drop_duplicates(subset='original_filename').reset_index(drop=True)
print(f"H vs PC training set: n={len(df)}  "
      f"H={sum(df['infoP']=='H')}, PC={sum(df['infoP']=='PC')}")

# ── Extract and preprocess SR_1005 for each patient ───────────────────────
sr_data = []
wn_common = None

for i in range(len(df)):
    spec = np.array(df['intensity_baseline_corrected'].iloc[i])
    wn   = np.array(df['wavenumber'].iloc[i])
    all_srs = preprocess_all_srs(spec, wn)
    sr_info  = all_srs[BEST_SR]
    sr_data.append(sr_info['spectrum'])  # preprocessed (mean-centered + std-normalised)
    if wn_common is None:
        wn_common = sr_info['wavenumbers']

Y = np.vstack(sr_data)          # (n, J)  — preprocessed spectra on SR_1005 grid
J = Y.shape[1]
labels = df['infoP'].values

print(f"Y shape: {Y.shape}, wavenumber axis: {wn_common[0]:.1f}–{wn_common[-1]:.1f} cm⁻¹ ({J} pts)")

# ── Run FPCA (fpca.face via R) on the full H+PC dataset ───────────────────
print(f"\nRunning fpca.face: n={len(Y)}, J={J}, lam={BEST_TAU}, npc=10 ...")
fpca = fpca_face_via_r(Y, npc=10, lam=BEST_TAU)
print(f"FPCA done: npc_returned={fpca.npc}, sigma2={fpca.sigma2:.4f}")
print(f"Eigenvalues (top 10): {fpca.evalues[:10]}")
Z = fpca.efunctions
ZtZ_diag = np.sum(Z ** 2, axis=0)  # ||φ_k||² on discrete grid
approx_w = ZtZ_diag[:10] / (ZtZ_diag[:10] + fpca.sigma2 / fpca.evalues[:10])
print(f"Shrinkage weights (top 10, exact): {approx_w}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Per-class mean functions and difference curve
# ═══════════════════════════════════════════════════════════════════════════

# Per-class mean of preprocessed spectra
mu_H   = Y[labels == 'H',  :].mean(axis=0)
mu_PC  = Y[labels == 'PC', :].mean(axis=0)
diff   = mu_H - mu_PC          # signed difference; we plot both signed and abs

# FPCA grand mean (from fpca.face, on original scale before per-sample normalisation)
mu_fpca = fpca.mu              # grand mean across all spectra in Y (both H and PC combined)

fig = plt.figure(figsize=(10, 7))
gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 0.05, 1.8], hspace=0.08)

# ── Panel A: individual spectra + class means ──────────────────────────────
ax_main = fig.add_subplot(gs[0])

for i in range(len(Y)):
    cls = labels[i]
    ax_main.plot(wn_common, Y[i], color=CLASS_COLORS[cls], alpha=0.15, lw=0.7)

ax_main.plot(wn_common, mu_H,  color=CLASS_COLORS['H'],  lw=2.5, label=f'$\\hat{{\\mu}}_H(t)$')
ax_main.plot(wn_common, mu_PC, color=CLASS_COLORS['PC'], lw=2.5, label=f'$\\hat{{\\mu}}_{{PC}}(t)$')

ax_main.set_ylabel('Normalised Absorbance', fontsize=12)
ax_main.set_title(
    f'SR 1005 ({SR_CENTER}$\\pm${WINDOW_HW} cm$^{{-1}}$) — '
    'Per-Class Mean Functions and Individual Spectra',
    fontsize=12
)
ax_main.legend(fontsize=11, loc='upper right')
ax_main.set_xlim(wn_common[0], wn_common[-1])
ax_main.tick_params(labelbottom=False)
ax_main.axhline(0, color='gray', lw=0.5, ls='--')

abs_diff  = np.abs(diff)
peak_idx  = np.argmax(abs_diff)
peak_wn   = wn_common[peak_idx]
shade_lo  = max(0, peak_idx - 5)
shade_hi  = min(J - 1, peak_idx + 5)

# ── Panel B (spacer — invisible) for breathing room ───────────────────────
ax_gap = fig.add_subplot(gs[1])
ax_gap.axis('off')

# ── Panel C: signed difference μ_H(t) − μ_PC(t) ──────────────────────────
ax_diff = fig.add_subplot(gs[2])

ax_diff.fill_between(wn_common, diff, 0,
                     where=(diff >= 0), color=CLASS_COLORS['H'],  alpha=0.5,
                     label='$\\hat{\\mu}_H > \\hat{\\mu}_{PC}$')
ax_diff.fill_between(wn_common, diff, 0,
                     where=(diff <  0), color=CLASS_COLORS['PC'], alpha=0.5,
                     label='$\\hat{\\mu}_{PC} > \\hat{\\mu}_H$')
ax_diff.plot(wn_common, diff, color='black', lw=1.2)
ax_diff.axhline(0, color='gray', lw=0.6, ls='--')
ax_diff.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
ax_diff.set_ylabel('$\\hat{\\mu}_H - \\hat{\\mu}_{PC}$', fontsize=12)
ax_diff.set_title('Class Mean Difference', fontsize=11)
ax_diff.legend(fontsize=9, loc='upper right')
ax_diff.set_xlim(wn_common[0], wn_common[-1])

plt.tight_layout()
plt.savefig(f'{OUT}/fsvc_5_7_mean_functions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fsvc_5_7_mean_functions.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Eigenfunctions + cumulative variance explained
# ═══════════════════════════════════════════════════════════════════════════

# fpca.face evalues are λ_k × J; proportion of variance explained:
npc_ret  = fpca.npc
evalues  = fpca.evalues[:npc_ret]
pve_each = evalues / evalues.sum()
pve_cum  = np.cumsum(pve_each)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Panel A: first K* eigenfunctions ──────────────────────────────────────
ax = axes[0]
efunc_colors = plt.cm.tab10(np.linspace(0, 0.5, BEST_K))

for k in range(BEST_K):
    phi = fpca.efunctions[:, k]   # (J,)
    # sign convention: make the largest-magnitude lobe positive
    if np.abs(phi.min()) > np.abs(phi.max()):
        phi = -phi
    pct = pve_each[k] * 100
    ax.plot(wn_common, phi, color=efunc_colors[k], lw=2.2,
            label=f'$\\varphi_{k+1}(t)$  ({pct:.1f}% var.)')

ax.axhline(0, color='gray', lw=0.6, ls='--')
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
ax.set_ylabel('Eigenfunction value', fontsize=12)
ax.set_title(f'First $K^* = {BEST_K}$ Eigenfunctions — SR 1005 (H vs PC, $\\tau^*={int(BEST_TAU)}$)',
             fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim(wn_common[0], wn_common[-1])

# ── Panel B: cumulative variance explained ─────────────────────────────────
ax2 = axes[1]
x_ticks = np.arange(1, npc_ret + 1)

bar_colors = ['#2196F3' if k < BEST_K else '#BDBDBD' for k in range(npc_ret)]
ax2.bar(x_ticks, pve_each * 100, color=bar_colors, edgecolor='white', alpha=0.9,
        label='Per-component variance')
ax2_twin = ax2.twinx()
ax2_twin.plot(x_ticks, pve_cum * 100, 'o-', color='black', lw=2, ms=5,
              label='Cumulative PVE')
ax2_twin.axhline(90, color='gray', ls=':', lw=1, label='90% threshold')
ax2_twin.axhline(95, color='gray', ls='--', lw=1, label='95% threshold')
ax2_twin.set_ylim(0, 105)
ax2_twin.set_ylabel('Cumulative Variance Explained (%)', fontsize=11)

# Mark K* with vertical dashed line
ax2.axvline(BEST_K + 0.5, color='red', ls='--', lw=1.5, label=f'$K^*={BEST_K}$')
ax2.set_xlabel('Number of FPCs retained ($K$)', fontsize=12)
ax2.set_ylabel('Per-component Variance Explained (%)', fontsize=11)
ax2.set_title('Cumulative Variance Explained by FPCs', fontsize=11)
ax2.set_xticks(x_ticks)

# Combined legend
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2_twin.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=9, loc='center right')

plt.tight_layout()
plt.savefig(f'{OUT}/fsvc_5_7_eigenfunctions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fsvc_5_7_eigenfunctions.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: BLUP — score scatter + exact shrinkage weights
# ═══════════════════════════════════════════════════════════════════════════
from fsvm_implement import estimate_pc_scores

sigma2 = fpca.sigma2

# Exact shrinkage weights from actual Z'Z diagonal (no J·I approximation)
_, shrinkage_weights = estimate_pc_scores(
    Y, fpca.mu, fpca.sigma2, fpca.evalues, fpca.efunctions,
    return_shrinkage=True,
)

# Use fpca.face BLUP scores directly for training data
scores_all = fpca.scores   # (n, npc)

GREY    = '#AAAAAA'
BLUE    = '#4C72B0'
RED     = '#C44E52'
ACCENT  = '#C44E52'

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax in axes:
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.tick_params(colors='#444444', labelsize=10)

# ── Panel A: FPC score scatter (FPC1 vs FPC2) ─────────────────────────────
ax = axes[0]
for cls, col, lbl in [('H', BLUE, 'Healthy (n=22)'), ('PC', RED, 'Pancreatic Cancer (n=17)')]:
    mask = labels == cls
    ax.scatter(scores_all[mask, 0], scores_all[mask, 1],
               color=col, s=55, alpha=0.8, edgecolors='white', lw=0.5,
               label=lbl, zorder=3)

ax.axhline(0, color=GREY, lw=0.8, ls='--', zorder=1)
ax.axvline(0, color=GREY, lw=0.8, ls='--', zorder=1)
ax.set_xlabel(f'FPC 1 score  ({pve_each[0]*100:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'FPC 2 score  ({pve_each[1]*100:.1f}% variance)', fontsize=11)
ax.set_title('A — BLUP Score Scatter (SR 1005, H vs PC)', fontsize=11,
             fontweight='bold', pad=8)
ax.legend(fontsize=9, frameon=False)
ax.yaxis.grid(True, color='#EEEEEE', lw=0.8, zorder=0)
ax.xaxis.grid(True, color='#EEEEEE', lw=0.8, zorder=0)
ax.set_axisbelow(True)

# ── Panel B: exact shrinkage weights — dot-line plot ──────────────────────
ax2 = axes[1]
ax2.yaxis.grid(True, color='#EEEEEE', lw=0.8, zorder=0)
ax2.set_axisbelow(True)

k_indices = np.arange(1, npc_ret + 1)
dot_colors = [ACCENT if k <= BEST_K else GREY for k in k_indices]

ax2.plot(k_indices, shrinkage_weights[:npc_ret], color=GREY,
         lw=1.4, zorder=2)
ax2.scatter(k_indices, shrinkage_weights[:npc_ret],
            color=dot_colors, s=70, zorder=4, edgecolors='white', lw=0.5)

ax2.axhline(1.0, color='#888888', ls=':', lw=1.2, label='No shrinkage ($w_k=1$)')
ax2.axvspan(0.5, BEST_K + 0.5, color=ACCENT, alpha=0.07,
            label=f'Retained ($K^*={BEST_K}$)')

# Annotate the retained components
for k in range(BEST_K):
    ax2.annotate(f'{shrinkage_weights[k]:.4f}',
                 xy=(k_indices[k], shrinkage_weights[k]),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=8.5, color=ACCENT, fontweight='bold')

ax2.set_xlabel('FPC index $k$', fontsize=11)
ax2.set_ylabel('Shrinkage weight  $w_k$', fontsize=11)
ax2.set_title('B — Exact BLUP Shrinkage Weights', fontsize=11,
              fontweight='bold', pad=8)
ax2.set_xticks(k_indices)
w_min = shrinkage_weights[:npc_ret].min()
ax2.set_ylim(max(0, w_min - 0.03), 1.015)
ax2.legend(fontsize=9, frameon=False, loc='lower left')

plt.suptitle(
    f'BLUP Score Estimation — SR 1005, H vs PC  '
    f'($\\tau^*={int(BEST_TAU)}$, $K^*={BEST_K}$, $\\hat{{\\sigma}}^2={sigma2:.4f}$)',
    fontsize=11, fontweight='bold', y=1.02,
)
plt.tight_layout()
plt.savefig(f'{OUT}/NEW_fsvc_5_7_shrinkage.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: NEW_fsvc_5_7_shrinkage.png")

# ── Print diagnostics for write-up ────────────────────────────────────────
print("\n=== Diagnostics for 5.7 write-up ===")
print(f"Grand mean range: [{mu_fpca.min():.4f}, {mu_fpca.max():.4f}]")
print(f"μ_H range:        [{mu_H.min():.4f}, {mu_H.max():.4f}]")
print(f"μ_PC range:       [{mu_PC.min():.4f}, {mu_PC.max():.4f}]")
print(f"Max |diff| at wn={peak_wn:.1f} cm⁻¹, value={abs_diff[peak_idx]:.4f}")
print(f"Eigenvalues: {np.round(evalues, 4)}")
print(f"PVE per component (%): {np.round(pve_each * 100, 1)}")
print(f"Cumulative PVE (%): {np.round(pve_cum * 100, 1)}")
noise_floor = sigma2 / J
print(f"sigma2: {sigma2:.6f},  J={J},  noise_floor=sigma2/J={noise_floor:.5f}")
print(f"SNR per component: {np.round(evalues/noise_floor, 1)}")
print(f"Shrinkage weights: {np.round(shrinkage_weights[:len(evalues)], 5)}")
print(f"w1={shrinkage_weights[0]:.5f}, w2={shrinkage_weights[1]:.5f}  (K*=2 components)")
