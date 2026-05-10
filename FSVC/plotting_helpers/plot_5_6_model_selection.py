"""
Section 5.6 — FSVC Model Selection and Optimal Configurations
Produces three figures:
  fig1: Heatmap of CV accuracy (SR × task)  — analogous to SVM plot1
  fig2: opt_K distribution across configs   — how many FPCs are selected
  fig3: opt_tau distribution                — does smoothing choice matter?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — remove if running in a notebook
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

OUT = '../eval_result_data/plots'
df  = pd.read_csv('../eval_result_data/fsvc_sr_best_params.csv')
df['sr'] = df['config_id'].str.split('__').str[0]

SR_ORDER   = ['SR_1005','SR_530','SR_1050','SR_1130','SR_1170',
              'SR_1190','SR_1203','SR_2170','concat_all']
TASK_ORDER = ['H_vs_PC','H_vs_KC_BC_PC']
TASK_LABELS = {'H_vs_PC': 'H vs PC', 'H_vs_KC_BC_PC': 'H vs KC+BC+PC'}

# ── Figure 1: CV accuracy heatmap ─────────────────────────────────────────
pivot = (df.pivot_table(values='cv_accuracy', index='sr', columns='task')
           .reindex(index=SR_ORDER, columns=TASK_ORDER))
pivot.columns = [TASK_LABELS[t] for t in TASK_ORDER]

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0.65, vmax=1.0, linewidths=0.4,
            cbar_kws={'label': 'CV Accuracy'}, ax=ax)
ax.set_xlabel('Classification Task', fontsize=11)
ax.set_ylabel('Spectral Region', fontsize=11)
ax.set_title('FSVC Joint CV Accuracy — Best (τ, K, C) per Config', fontsize=12)
ax.tick_params(axis='x', rotation=15)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig(f'{OUT}/fsvc_5_6_cv_accuracy_heatmap.png', dpi=150)
plt.show()
print("Saved: fsvc_5_6_cv_accuracy_heatmap.png")

# ── Figure 2: opt_K distribution ──────────────────────────────────────────
BAR_COLOR    = '#4C72B0'   # muted blue — panel A bars
BOX_COLOR    = '#4C72B0'   # same family — panel B boxes
MEDIAN_COLOR = '#C44E52'   # muted red
GRID_COLOR   = '#E5E5E5'

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax in axes:
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.tick_params(colors='#444444', labelsize=10)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# Panel A: overall K frequency
k_counts = df['opt_K'].value_counts().sort_index()
med_k_all = int(df['opt_K'].median())
axes[0].bar(k_counts.index.astype(str), k_counts.values,
            color=BAR_COLOR, edgecolor='white', linewidth=0.6,
            width=0.65, zorder=3)
axes[0].axvline(x=str(med_k_all), color=MEDIAN_COLOR, ls='--', lw=1.5,
                label=f'Median = {med_k_all}', zorder=4)
axes[0].set_xlabel('Optimal K (FPCs retained)', fontsize=11, color='#333333')
axes[0].set_ylabel('Number of configurations', fontsize=11, color='#333333')
axes[0].set_title('A: Distribution of optimal K', fontsize=11,
                  fontweight='bold', color='#222222', pad=8)
axes[0].legend(fontsize=9, frameon=False)
axes[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Panel B: K by task — boxplot
k_by_task = df.groupby('task')['opt_K'].apply(list).reindex(TASK_ORDER)
positions  = np.arange(len(TASK_ORDER))
bp = axes[1].boxplot(
    [k_by_task[t] for t in TASK_ORDER],
    positions=positions, widths=0.42,
    patch_artist=True,
    boxprops=dict(facecolor=BOX_COLOR, alpha=0.55, linewidth=1.2),
    medianprops=dict(color=MEDIAN_COLOR, lw=2.2),
    whiskerprops=dict(color='#555555', lw=1.2, linestyle='--'),
    capprops=dict(color='#555555', lw=1.5),
    flierprops=dict(marker='o', markerfacecolor='#888888',
                    markersize=5, linestyle='none', markeredgewidth=0),
    zorder=3,
)
axes[1].set_xticks(positions)
axes[1].set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER],
                         rotation=0, fontsize=10.5)
axes[1].set_ylabel('Optimal K', fontsize=11, color='#333333')
axes[1].set_title('B: Optimal K by classification task', fontsize=11,
                  fontweight='bold', color='#222222', pad=8)
axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Annotate median above the red line (inside the box, not off to the side)
for pos, task in zip(positions, TASK_ORDER):
    med_val = int(np.median(k_by_task[task]))
    axes[1].text(pos, med_val + 0.35, f'median = {med_val}',
                 va='bottom', ha='center', color=MEDIAN_COLOR,
                 fontsize=9, fontweight='bold')

fig.suptitle('FSVC — Number of Functional Principal Components Selected',
             fontsize=12, fontweight='bold', color='#111111', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fsvc_5_6_optK_distribution.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved: fsvc_5_6_optK_distribution.png")

# ── Figure 3: opt_tau distribution ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Panel A: overall tau frequency
tau_counts = df['opt_tau'].value_counts().sort_index()
axes[0].bar(tau_counts.index.astype(str), tau_counts.values,
            color='darkorange', edgecolor='white')
axes[0].set_xlabel('Optimal τ (smoothing parameter)', fontsize=11)
axes[0].set_ylabel('Number of configs', fontsize=11)
axes[0].set_title(f'A  — Distribution of optimal τ (all {len(df)} configs)', fontsize=11)
pct_min = 100 * tau_counts.iloc[0] / len(df)
# axes[0].annotate(f'{tau_counts.iloc[0]}/{len(df)} configs\n({pct_min:.0f}%) select τ=0.5',
#                  xy=(0, tau_counts.iloc[0]),
#                  xytext=(0.6, tau_counts.iloc[0] * 0.55),
#                  fontsize=9, color='darkred',
#                  arrowprops=dict(arrowstyle='->', color='darkred'))

# Panel B: tau vs cv_accuracy scatter — does τ choice predict accuracy?
tau_vals = sorted(df['opt_tau'].unique())
colors   = plt.cm.tab10(np.linspace(0, 0.6, len(tau_vals)))
for tau, col in zip(tau_vals, colors):
    sub = df[df['opt_tau'] == tau]
    axes[1].scatter(sub['cv_accuracy'], [tau] * len(sub),
                    color=col, s=60, alpha=0.8, label=f'τ={tau}', edgecolors='white')
axes[1].set_xlabel('CV Accuracy', fontsize=11)
axes[1].set_ylabel('Optimal τ', fontsize=11)
axes[1].set_title('B  — Does τ correlate with CV accuracy?', fontsize=11)
axes[1].legend(fontsize=9)
# Mean accuracy per tau as vertical lines
for tau, col in zip(tau_vals, colors):
    mean_acc = df[df['opt_tau'] == tau]['cv_accuracy'].mean()
    axes[1].axvline(mean_acc, color=col, ls='--', lw=1, alpha=0.6)

plt.suptitle('FSVC — Smoothing Parameter τ Selection', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fsvc_5_6_tau_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fsvc_5_6_tau_distribution.png")

# ── Print summary table for thesis ────────────────────────────────────────
eval_df = pd.read_csv('../eval_result_data/fsvc_sr_evaluation_results.csv')
loocv = eval_df[eval_df['method'] == 'LOOCV'][
    ['config_id', 'sensitivity', 'specificity', 'accuracy']
].rename(columns={'accuracy': 'loocv_accuracy'})
loocv['balanced_accuracy'] = (loocv['sensitivity'] + loocv['specificity']) / 2

df_merged = df.merge(loocv, on='config_id', how='left')

print("\n=== Summary table (best config per task) ===")
best = (df_merged.sort_values('cv_accuracy', ascending=False)
          .groupby('task').first()
          .reindex(TASK_ORDER)
          [['sr','opt_tau','opt_K','opt_C','cv_accuracy','n_samples',
            'loocv_accuracy','sensitivity','specificity','balanced_accuracy']])
best.index = [TASK_LABELS[t] for t in TASK_ORDER]
best_display = best.copy()
for col in ['cv_accuracy','loocv_accuracy','sensitivity','specificity','balanced_accuracy']:
    best_display[col] = best_display[col].map('{:.3f}'.format)
print(best_display.to_string())

print("\n=== Key observations for 5.6 write-up ===")
print(f"τ=0.5 selected in {(df['opt_tau']==0.5).sum()}/{len(df)} configs "
      f"({100*(df['opt_tau']==0.5).mean():.0f}%) — minimal smoothing preferred.")
print(f"K=1  selected in {(df['opt_K']==1).sum()}/{len(df)} configs "
      f"({100*(df['opt_K']==1).mean():.0f}%) — single FPC sufficient for most tasks.")
print(f"Mean K for H_vs_PC: {df[df['task']=='H_vs_PC']['opt_K'].mean():.1f} "
      f"vs H_vs_KC_BC_PC: {df[df['task']=='H_vs_KC_BC_PC']['opt_K'].mean():.1f}")
print(f"Best overall: {df.loc[df['cv_accuracy'].idxmax(), 'config_id']} "
      f"(CV acc = {df['cv_accuracy'].max():.4f})")
