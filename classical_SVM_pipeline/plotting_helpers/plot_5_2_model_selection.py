"""
Section 5.2 — Classical SVM Model Selection and Optimal Configurations
Produces three figures (parallel to FSVC plot_5_6_model_selection.py):
  fig1: Heatmap of CV accuracy (SR × task)         — analogous to FSVC fig1
  fig2: opt_sigma distribution across configs       — analogous to FSVC opt_K
  fig3: opt_C distribution + C vs CV accuracy       — analogous to FSVC opt_tau
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

OUT = '../eval_result_data/plots'
df_all = pd.read_csv('../eval_result_data/all_configs_best_params.csv')

# PCA-4 path only
df = df_all[df_all['feature_type'] == 'pca'].copy()
df['sr'] = df['sr_col'].str.replace('_preprocessed', '').str.replace('SR_', 'SR_')
df.loc[df['sr_mode'] == 'concat', 'sr'] = 'concat_all'

SR_ORDER    = ['SR_1005', 'SR_530', 'SR_1050', 'SR_1130', 'SR_1170',
               'SR_1190', 'SR_1203', 'SR_2170', 'concat_all']
TASK_ORDER  = ['H_vs_PC', 'H_vs_KC_BC_PC']
TASK_LABELS = {'H_vs_PC': 'H vs PC', 'H_vs_KC_BC_PC': 'H vs KC+BC+PC'}

BAR_COLOR    = '#4C72B0'
MEDIAN_COLOR = '#C44E52'
GRID_COLOR   = '#E5E5E5'


# ── Figure 1: CV accuracy heatmap ─────────────────────────────────────────────
pivot = (df.pivot_table(values='cv_accuracy', index='sr', columns='task')
           .reindex(index=SR_ORDER, columns=TASK_ORDER))
pivot.columns = [TASK_LABELS[t] for t in TASK_ORDER]

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0.50, vmax=1.0, linewidths=0.4,
            cbar_kws={'label': 'CV Accuracy'}, ax=ax)
ax.set_xlabel('Classification Task', fontsize=11)
ax.set_ylabel('Spectral Region', fontsize=11)
ax.set_title('Classical SVM Joint CV Accuracy — Best (σ, C) per Config\n(PCA-4 features, RBF kernel)',
             fontsize=11)
ax.tick_params(axis='x', rotation=15)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig(f'{OUT}/svm_5_2_cv_accuracy_heatmap.png', dpi=150)
plt.close()
print("Saved: svm_5_2_cv_accuracy_heatmap.png")


# ── Figure 2: opt_sigma distribution ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax in axes:
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.tick_params(colors='#444444', labelsize=10)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# Panel A: overall sigma frequency
sigma_counts = df['best_sigma'].value_counts().sort_index()
med_sigma    = int(df['best_sigma'].median())
axes[0].bar(sigma_counts.index.astype(str), sigma_counts.values,
            color=BAR_COLOR, edgecolor='white', linewidth=0.6,
            width=0.65, zorder=3)
axes[0].axvline(x=str(med_sigma), color=MEDIAN_COLOR, ls='--', lw=1.5,
                label=f'Median = {med_sigma}', zorder=4)
axes[0].set_xlabel('Optimal σ (Gaussian smoothing width)', fontsize=11, color='#333333')
axes[0].set_ylabel('Number of configurations', fontsize=11, color='#333333')
axes[0].set_title('A: Distribution of optimal σ', fontsize=11,
                  fontweight='bold', color='#222222', pad=8)
axes[0].legend(fontsize=9, frameon=False)
axes[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Panel B: sigma by task — boxplot
sigma_by_task = df.groupby('task')['best_sigma'].apply(list).reindex(TASK_ORDER)
positions     = np.arange(len(TASK_ORDER))
bp = axes[1].boxplot(
    [sigma_by_task[t] for t in TASK_ORDER],
    positions=positions, widths=0.42,
    patch_artist=True,
    boxprops=dict(facecolor=BAR_COLOR, alpha=0.55, linewidth=1.2),
    medianprops=dict(color=MEDIAN_COLOR, lw=2.2),
    whiskerprops=dict(color='#555555', lw=1.2, linestyle='--'),
    capprops=dict(color='#555555', lw=1.5),
    flierprops=dict(marker='o', markerfacecolor='#888888',
                    markersize=5, linestyle='none', markeredgewidth=0),
    zorder=3,
)
axes[1].set_xticks(positions)
axes[1].set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], rotation=0, fontsize=10.5)
axes[1].set_ylabel('Optimal σ', fontsize=11, color='#333333')
axes[1].set_title('B: Optimal σ by classification task', fontsize=11,
                  fontweight='bold', color='#222222', pad=8)
axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

for pos, task in zip(positions, TASK_ORDER):
    med_val = int(np.median(sigma_by_task[task]))
    axes[1].text(pos, med_val + 0.5, f'median = {med_val}',
                 va='bottom', ha='center', color=MEDIAN_COLOR,
                 fontsize=9, fontweight='bold')

fig.suptitle('Classical SVM — Gaussian Smoothing Width σ Selected by CV',
             fontsize=12, fontweight='bold', color='#111111', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/svm_5_2_sigma_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: svm_5_2_sigma_distribution.png")


# ── Figure 3: opt_C distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax in axes:
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.tick_params(colors='#444444', labelsize=10)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# Panel A: overall C frequency
c_counts = df['best_C'].value_counts().sort_index()
axes[0].bar(c_counts.index.astype(str), c_counts.values,
            color='darkorange', edgecolor='white', width=0.6, zorder=3)
axes[0].set_xlabel('Optimal C (regularisation)', fontsize=11, color='#333333')
axes[0].set_ylabel('Number of configurations', fontsize=11, color='#333333')
axes[0].set_title(f'A: Distribution of optimal C (all {len(df)} configs)', fontsize=11,
                  fontweight='bold', color='#222222', pad=8)
axes[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Panel B: C vs CV accuracy scatter — does C choice predict accuracy?
c_vals = sorted(df['best_C'].unique())
colors = plt.cm.tab10(np.linspace(0, 0.6, len(c_vals)))
for c_val, col in zip(c_vals, colors):
    sub = df[df['best_C'] == c_val]
    axes[1].scatter(sub['cv_accuracy'], [c_val] * len(sub),
                    color=col, s=60, alpha=0.8, label=f'C={c_val}',
                    edgecolors='white', zorder=3)
axes[1].set_xlabel('CV Accuracy', fontsize=11, color='#333333')
axes[1].set_ylabel('Optimal C', fontsize=11, color='#333333')
axes[1].set_title('B: Does C correlate with CV accuracy?', fontsize=11,
                  fontweight='bold', color='#222222', pad=8)
axes[1].legend(fontsize=9, frameon=False)
for c_val, col in zip(c_vals, colors):
    mean_acc = df[df['best_C'] == c_val]['cv_accuracy'].mean()
    axes[1].axvline(mean_acc, color=col, ls='--', lw=1, alpha=0.6)

fig.suptitle('Classical SVM — Regularisation Parameter C Selected by CV',
             fontsize=12, fontweight='bold', color='#111111', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/svm_5_2_C_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: svm_5_2_C_distribution.png")


# ── Print summary stats for thesis write-up ───────────────────────────────────
print("\n=== Key observations for 5.2.1 write-up ===")
print(f"σ=0 selected in {(df['best_sigma']==0).sum()}/{len(df)} configs "
      f"({100*(df['best_sigma']==0).mean():.0f}%) — no smoothing preferred.")
print(f"C=0.2575 selected in {(df['best_C']==0.2575).sum()}/{len(df)} configs "
      f"({100*(df['best_C']==0.2575).mean():.0f}%) — low regularisation preferred.")
print(f"Mean σ for H_vs_PC:        {df[df['task']=='H_vs_PC']['best_sigma'].mean():.1f}")
print(f"Mean σ for H_vs_KC_BC_PC:  {df[df['task']=='H_vs_KC_BC_PC']['best_sigma'].mean():.1f}")
print(f"Best overall: {df.loc[df['cv_accuracy'].idxmax(), 'sr']} | "
      f"{df.loc[df['cv_accuracy'].idxmax(), 'task']} "
      f"(CV acc = {df['cv_accuracy'].max():.4f})")


# ── LaTeX summary table (best config per task) ────────────────────────────────
eval_df = pd.read_csv('../eval_result_data/svm_evaluation_results.csv')
loocv = eval_df[(eval_df['method'] == 'LOOCV') & (eval_df['pca'] == 1)][
    ['config_id', 'sensitivity', 'specificity', 'balanced_accuracy', 'accuracy']
].rename(columns={'accuracy': 'loocv_accuracy'})

df_merged = df.merge(loocv, on='config_id', how='left')

TASK_LABELS_TEX = {
    'H_vs_PC':        r'H vs.\ PCa',
    'H_vs_KC_BC_PC':  r'H vs.\ KC+BC+PCa',
}

best = (df_merged.sort_values('cv_accuracy', ascending=False)
        .groupby('task').first()
        .reindex(TASK_ORDER))

print("\n=== LaTeX table: best config per task (Classical SVM) ===\n")
print(r'\begin{table}[h!]')
print(r'\centering')
print(r'\small')
print(r'\begin{tabular}{|l|c|c|c|c|c|c|c|}')
print(r'\hline')
print(r'\textbf{Task} & \textbf{SR} & $\boldsymbol{\sigma}$ & $\boldsymbol{C}$ '
      r'& \textbf{CV Acc.} & \textbf{Sens.} & \textbf{Spec.} & \textbf{Bal.\ Acc.} \\')
print(r'\hline')
for task in TASK_ORDER:
    row   = best.loc[task]
    sr    = row['sr']
    sigma = int(row['best_sigma'])
    C     = row['best_C']
    cv    = row['cv_accuracy']
    sens  = row['sensitivity']
    spec  = row['specificity']
    bal   = row['balanced_accuracy']
    label = TASK_LABELS_TEX[task]
    print(f'{label} & {sr} & {sigma} & {C:.4f} & {cv:.3f} & '
          f'{sens:.3f} & {spec:.3f} & {bal:.3f} \\\\')
print(r'\hline \hline')
print(r'\end{tabular}')
print(r'\caption{Optimal classical SVM configuration per task from stratified '
      r'5-fold cross-validation over $(\sigma, C)$. Sensitivity and specificity '
      r'from LOOCV on the selected configuration.}')
print(r'\label{tab:svm_best_config_summary}')
print(r'\end{table}')
