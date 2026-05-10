"""
Regenerate svm_permutation_test.png from saved CSV files.
Reads null distributions and summary stats — no permutations are re-run.
Fixes: legend pinned to upper-right; stats annotation moved to bottom-left.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_NULL    = '../eval_result_data/svm_permutation_null_distributions.csv'
CSV_SUMMARY = '../eval_result_data/svm_permutation_null_distributions_summary.csv'
OUT         = '../eval_result_data/plots/svm_permutation_test.png'

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df_null    = pd.read_csv(CSV_NULL)
df_summary = pd.read_csv(CSV_SUMMARY).set_index('task')

TASK_LABELS = {
    'H_vs_PC':       'Task I — H vs PC',
    'H_vs_KC_BC_PC': 'Task II — H vs KC+BC+PC',
}

# Config labels for subplot titles — sourced from the run that produced the CSVs
TASK_CONFIG_LABEL = {
    'H_vs_PC':       r'SR$_{1005}$, PCA-4, RBF, $C=0.7525$, $\sigma=15$',
    'H_vs_KC_BC_PC': r'SR$_{1170}$, PCA-4, RBF, $C=1.0$, $\sigma=15$',
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, task in zip(axes, TASK_LABELS):
    null  = df_null[df_null['task'] == task]['perm_accuracy'].values
    row   = df_summary.loc[task]
    obs   = row['observed_acc']
    p_val = row['p_value']
    n_perm = int(row['n_permutations'])

    # Histogram
    ax.hist(null, bins=30, color='#90A4AE', edgecolor='white',
            linewidth=0.5, alpha=0.85, label='Null distribution')

    # Observed accuracy line
    ax.axvline(obs, color='#E53935', lw=2.0, ls='-',
               label=f'Observed LOOCV acc = {obs:.3f}')

    # 95th percentile line
    pct95 = np.percentile(null, 95)
    ax.axvline(pct95, color='#FB8C00', lw=1.5, ls='--',
               label=f'95th percentile = {pct95:.3f}')

    # Shade tail beyond observed
    if obs <= null.max():
        ax.axvspan(obs, null.max() + 0.01, alpha=0.15, color='#E53935')

    # ── Legend: pinned upper-right, no frame ──────────────────────────────────
    ax.legend(fontsize=9, frameon=False, loc='upper right')

    # ── p-value: bottom-right ─────────────────────────────────────────────────
    p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
    ax.text(0.97, 0.05, p_str,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold',
            color='#B71C1C' if p_val < 0.05 else '#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    # ── Stats annotation: bottom-left ─────────────────────────────────────────
    ax.text(0.03, 0.05,
            f'Null mean = {null.mean():.3f}\n'
            f'N permutations = {n_perm}',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8.5, color='#555555')

    ax.set_xlabel('LOOCV Accuracy under permuted labels', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'{TASK_LABELS[task]}\n{TASK_CONFIG_LABEL[task]}', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

plt.suptitle(
    'Classical SVM — Permutation Test (LOOCV Accuracy)',
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}")
