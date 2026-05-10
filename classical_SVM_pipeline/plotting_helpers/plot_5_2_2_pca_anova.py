"""
Section 5.2.2 — PCA-Based Exploratory Findings
===============================================
For each spectral region × task:
  - PCA on mean-centred, unit-variance normalised SR (same preprocessing as SVM)
  - Scatter plot: PC1 vs PC2, coloured by class label
  - One-way ANOVA on each of the top 4 PCs with Bonferroni correction

Saves one scatter figure per SR (both tasks on same figure) and one
summary ANOVA table (LaTeX + printed).
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '..')
from sr_preprocessing import extract_sr_window, preprocess_sr

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH = '../../data_processed/breath_data.pkl'
OUT       = '../eval_result_data/plots'

SR_CENTERS = {
    'SR_1005': 1005, 'SR_530':  530,  'SR_1050': 1050, 'SR_1130': 1130,
    'SR_1170': 1170, 'SR_1190': 1190, 'SR_1203': 1203, 'SR_2170': 2170,
}
SR_ORDER = ['SR_1005', 'SR_530', 'SR_1050', 'SR_1130',
            'SR_1170', 'SR_1190', 'SR_1203', 'SR_2170']

TASKS = {
    'H_vs_PC':        ['H', 'PC'],
    'H_vs_KC_BC_PC':  ['H', 'KC', 'BC', 'PC'],
}
TASK_LABELS = {
    'H_vs_PC':       'H vs PC',
    'H_vs_KC_BC_PC': 'H vs KC+BC+PC',
}


# Class colours — healthy grey, cancer red/orange shades
CLASS_COLORS = {
    'H':  '#4878CF',   # blue
    'PC': '#E84040',   # red
    'KC': '#F5A623',   # orange
    'BC': '#7B2D8B',   # purple
}
N_PCS = 4


# ── Load and prepare data ─────────────────────────────────────────────────────
with open(DATA_PATH, 'rb') as f:
    df_raw = pickle.load(f)

# Same deduplication and label cleaning as SVM_notebook.ipynb
df_raw['infoP'] = df_raw['infoP'].apply(lambda x: 'H' if x in ['M', 'F', 'H'] else x)
df_raw = df_raw.drop_duplicates(subset='original_filename').reset_index(drop=True)
df_train = df_raw[df_raw['category'] != 'blinddata'].reset_index(drop=True)

print(f"Training set: {df_train['infoP'].value_counts().to_dict()}")


# ── Extract preprocessed SR features ─────────────────────────────────────────
def get_sr_matrix(df, sr_name):
    """
    Extract mean-centred + unit-variance normalised SR matrix (n_samples × n_points).
    This matches the SVM feature extraction pipeline exactly.
    """
    center = SR_CENTERS[sr_name]
    X_rows = []
    for _, row in df.iterrows():
        wn  = row['wavenumber']
        sp  = row['intensity_baseline_corrected']
        sr_spec, _ = extract_sr_window(sp, wn, center=center, window_width=30.0)
        processed  = preprocess_sr(sr_spec)           # mean-centre + std-normalise
        X_rows.append(processed['preprocessed'])
    return np.array(X_rows)


# ── ANOVA with Bonferroni correction ─────────────────────────────────────────
def anova_on_pcs(scores, labels, n_pcs=N_PCS):
    """
    One-way ANOVA for each PC. Returns DataFrame with F-stat, p-value,
    Bonferroni-corrected p-value, and significance flag.
    """
    results = []
    groups_all = np.unique(labels)
    for k in range(n_pcs):
        groups = [scores[labels == g, k] for g in groups_all]
        F, p   = stats.f_oneway(*groups)
        results.append({'PC': f'Comp{k+1}', 'F': F, 'p': p})
    anova_df = pd.DataFrame(results)
    # Bonferroni correction: multiply by number of tests
    anova_df['p_bonf'] = np.minimum(anova_df['p'] * n_pcs, 1.0)
    anova_df['sig']    = anova_df['p_bonf'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    return anova_df


# ── Main loop: scatter + ANOVA per SR ─────────────────────────────────────────
all_anova_rows = []

for sr_name in SR_ORDER:
    print(f"\nProcessing {sr_name}...")
    X_full = get_sr_matrix(df_train, sr_name)
    y_full = df_train['infoP'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{sr_name}  —  Comp. 1 vs Comp. 2 by class label\n'
                 f'(mean-centred, unit-variance normalised; PCA fit on task subset)',
                 fontsize=12, fontweight='bold')

    for ax, (task_key, task_classes) in zip(axes, TASKS.items()):
        mask   = np.isin(y_full, task_classes)
        X_task = X_full[mask]
        y_task = y_full[mask]

        # Fit PCA on task subset (same as SVM pipeline)
        pca    = PCA(n_components=N_PCS, random_state=42)
        scores = pca.fit_transform(X_task)
        var_exp = pca.explained_variance_ratio_

        # Scatter PC1 vs PC2
        for cls in task_classes:
            idx = y_task == cls
            ax.scatter(scores[idx, 0], scores[idx, 1],
                       c=CLASS_COLORS.get(cls, '#888888'),
                       label=cls, s=55, alpha=0.8,
                       edgecolors='white', linewidths=0.5, zorder=3)

        ax.set_xlabel(f'Comp. 1 ({var_exp[0]*100:.1f}% var.)', fontsize=10)
        ax.set_ylabel(f'Comp. 2 ({var_exp[1]*100:.1f}% var.)', fontsize=10)
        ax.set_title(TASK_LABELS[task_key], fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='#CCCCCC', lw=0.8, zorder=0)
        ax.axvline(0, color='#CCCCCC', lw=0.8, zorder=0)

        # ANOVA
        anova_df = anova_on_pcs(scores, y_task)
        anova_df['sr']   = sr_name
        anova_df['task'] = task_key
        all_anova_rows.append(anova_df)

        # Annotate significant PCs on plot
        sig_pcs = anova_df[anova_df['sig'] != 'ns']['PC'].tolist()  # 'PC' is the column name in anova_df
        if sig_pcs:
            ax.annotate(f"ANOVA sig.: {', '.join(sig_pcs)}",
                        xy=(0.03, 0.97), xycoords='axes fraction',
                        va='top', fontsize=8, color='#444444',
                        bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='none'))

    plt.tight_layout()
    fname = f'{OUT}/svm_5_2_2_pca_scatter_{sr_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ── ANOVA summary: pivot to wide format (PCs as columns) ─────────────────────
anova_all = pd.concat(all_anova_rows, ignore_index=True)

# Wide format: one row per (SR, task), columns for each PC
wide = anova_all.pivot_table(index=['sr', 'task'], columns='PC',
                              values=['F', 'p_bonf', 'sig'],
                              aggfunc='first')
wide.columns = [f'{val}_{pc}' for val, pc in wide.columns]
wide = wide.reset_index()

# Restore SR order, group by task first then SR
sr_order_idx   = {s: i for i, s in enumerate(SR_ORDER)}
task_order_idx = {t: i for i, t in enumerate(TASKS.keys())}
wide['_sr_order']   = wide['sr'].map(sr_order_idx)
wide['_task_order'] = wide['task'].map(task_order_idx)
wide = wide.sort_values(['_task_order', '_sr_order']).drop(
    columns=['_sr_order', '_task_order']).reset_index(drop=True)

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = '../eval_result_data/svm_pca_anova_results.csv'
wide.to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")

# ── LaTeX ANOVA table (PCs as columns) ───────────────────────────────────────
TASK_LABELS_TEX = {
    'H_vs_PC':       r'H vs.\ PC',
    'H_vs_KC_BC_PC': r'H vs.\ KC+BC+PC',
}
PCS = ['Comp1', 'Comp2', 'Comp3', 'Comp4']
PC_COLS = {pc: f'F_{pc}' for pc in PCS}  # maps Comp1 -> F_Comp1 etc.

print("\n\n=== LaTeX ANOVA table (PCs horizontal) ===\n")
print(r'\begin{table}[h!]')
print(r'\centering')
print(r'\small')
print(r'\begin{tabular}{llcccc}')
print(r'\hline')
pc_header = ' & '.join([r'\textbf{' + pc + r'}' for pc in PCS])
print(r'\textbf{SR} & \textbf{Task} & ' + pc_header + r' \\')
print(r'\quad & & ' + ' & '.join([r'\small$F\ (p_\text{Bonf})$'] * 4) + r' \\')
print(r'\hline')

prev_task = None
for _, row in wide.iterrows():
    # insert \hline between task blocks
    if prev_task is not None and row['task'] != prev_task:
        print(r'\hline')
    prev_task  = row['task']
    sr         = row['sr']
    task_label = TASK_LABELS_TEX[row['task']]
    cells = []
    for pc in PCS:
        F     = row[f'F_{pc}']
        p     = row[f'p_bonf_{pc}']
        sig   = row[f'sig_{pc}']
        cell  = f'{F:.2f}\ ({p:.3f})'
        if sig != 'ns':
            cell = r'\textbf{' + cell + r'}' + f'^{{{sig}}}'
        cells.append(f'${cell}$')
    print(f'{sr} & {task_label} & ' + ' & '.join(cells) + r' \\')

print(r'\end{tabular}')
print(r'\caption{One-way ANOVA on the top 4 principal components per spectral '
      r'region and task. Each cell shows $F$-statistic and Bonferroni-corrected '
      r'$p$-value ($p_{\text{Bonf}} = p \times 4$). Bold = significant after '
      r'correction. *** $p<0.001$, ** $p<0.01$, * $p<0.05$. PCA fitted on '
      r'mean-centred, unit-variance normalised spectra within each task subset.}')
print(r'\label{tab:svm_pca_anova}')
print(r'\end{table}')
