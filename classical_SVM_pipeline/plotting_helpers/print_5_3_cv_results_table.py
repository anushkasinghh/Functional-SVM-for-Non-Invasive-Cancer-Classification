"""
Section 5.3 — Cross-Validation Performance Table
==================================================
Prints a LaTeX table with LOOCV and 9-fold (×10) accuracy, sensitivity,
and specificity for all 18 PCA-4 configurations across both tasks.
Bold = best accuracy per task.
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../eval_result_data/svm_evaluation_results.csv')

# PCA-4 only
df = df[df['pca'] == 1].copy()

SR_ORDER = ['SR_1005', 'SR_530', 'SR_1050', 'SR_1130',
            'SR_1170', 'SR_1190', 'SR_1203', 'SR_2170', 'concat_all']
TASK_ORDER  = ['H_vs_PC', 'H_vs_KC_BC_PC']
TASK_LABELS = {
    'H_vs_PC':       r'H vs.\ PC',
    'H_vs_KC_BC_PC': r'H vs.\ KC+BC+PC',
}

# Clean up sr name
df['sr'] = df['sr_used'].str.replace('SR_', 'SR_')
df.loc[df['sr_type'] == 'concat', 'sr'] = 'concat\_all'

# Pivot to get LOOCV and 9-fold side by side
loocv = df[df['method'] == 'LOOCV'][
    ['sr', 'task', 'accuracy', 'sensitivity', 'specificity']
].rename(columns={'accuracy': 'loocv_acc', 'sensitivity': 'loocv_sens',
                  'specificity': 'loocv_spec'})

kfold = df[df['method'] == '9-fold (x10)'][
    ['sr', 'task', 'accuracy', 'sensitivity', 'specificity',
     'accuracy_std', 'sensitivity_std', 'specificity_std']
].rename(columns={'accuracy': 'kf_acc', 'sensitivity': 'kf_sens',
                  'specificity': 'kf_spec', 'accuracy_std': 'kf_std',
                  'sensitivity_std': 'kf_sens_std', 'specificity_std': 'kf_spec_std'})

merged = loocv.merge(kfold, on=['sr', 'task'])

# Sort by SR order, task order
sr_idx   = {s: i for i, s in enumerate(SR_ORDER)}
task_idx = {t: i for i, t in enumerate(TASK_ORDER)}
merged['_sr']   = merged['sr'].map(sr_idx)
merged['_task'] = merged['task'].map(task_idx)
merged = merged.sort_values(['_task', '_sr']).drop(columns=['_sr', '_task']).reset_index(drop=True)

# Best LOOCV accuracy per task (for bolding)
best_loocv = merged.groupby('task')['loocv_acc'].transform('max')
best_kf    = merged.groupby('task')['kf_acc'].transform('max')


# ── Print LaTeX table ─────────────────────────────────────────────────────────
print(r'\begin{table}[h!]')
print(r'\centering')
print(r'\small')
print(r'\setlength{\tabcolsep}{4pt}')
print(r'\begin{tabular}{llccc|ccc}')
print(r'\hline')
print(r'\multirow{2}{*}{\textbf{Task}} & \multirow{2}{*}{\textbf{SR}} '
      r'& \multicolumn{3}{c|}{\textbf{LOOCV}} '
      r'& \multicolumn{3}{c}{\textbf{9-Fold $\times$10}} \\')
print(r' & & \textbf{Acc.} & \textbf{Sens.} & \textbf{Spec.} '
      r'& \textbf{Acc.\ ($\pm$std)} & \textbf{Sens.\ ($\pm$std)} & \textbf{Spec.\ ($\pm$std)} \\')
print(r'\hline')

prev_task = None
for i, row in merged.iterrows():
    # task block separator
    if prev_task is not None and row['task'] != prev_task:
        print(r'\hline')
    prev_task = row['task']

    is_best = row['loocv_acc'] == best_loocv[i]
    b  = r'\textbf{' if is_best else ''
    be = r'}' if is_best else ''

    task_label   = TASK_LABELS[row['task']]
    sr           = row['sr']
    kf_acc_str  = f"{row['kf_acc']:.3f} ({row['kf_std']:.3f})"
    kf_sens_str = f"{row['kf_sens']:.3f} ({row['kf_sens_std']:.3f})"
    kf_spec_str = f"{row['kf_spec']:.3f} ({row['kf_spec_std']:.3f})"

    print(f"{b}{task_label}{be} & {b}{sr}{be} & "
          f"{b}{row['loocv_acc']:.3f}{be} & "
          f"{b}{row['loocv_sens']:.3f}{be} & "
          f"{b}{row['loocv_spec']:.3f}{be} & "
          f"{b}{kf_acc_str}{be} & "
          f"{b}{kf_sens_str}{be} & "
          f"{b}{kf_spec_str}{be} \\\\")

print(r'\hline')
print(r'\end{tabular}')
print(r'\caption{SVM LOOCV and repeated 9-fold CV (10 repetitions) performance '
      r'for all 18 configurations across both tasks. '
      r'Bold rows indicate the best LOOCV accuracy per task. '
      r'9-fold CV reports mean $\pm$ std over 10 repetitions for all metrics. '
      r'PCA-4 features, RBF kernel.}')
print(r'\label{tab:svm_cv_results}')
print(r'\end{table}')
