"""
LaTeX table generation for Classical SVM results (Section 5.2).

Prints two tables:
  Table 1: All configurations — best (σ, C, CV Acc.) per SR × task
           Parallel to FSVC table showing all (τ, K, C) configs.
  Table 2: Best config per task summary with LOOCV sens/spec/bal. acc.
           Parallel to FSVC tab:best_config_summary.
"""

import pandas as pd
import numpy as np

df_all = pd.read_csv('../eval_result_data/all_configs_best_params.csv')

# PCA-4 path only
df = df_all[df_all['feature_type'] == 'pca'].copy()
df['sr'] = df['sr_col'].str.replace('_preprocessed', '').str.replace('SR_', 'SR_')
df.loc[df['sr_mode'] == 'concat', 'sr'] = 'concat\_all'

SR_ORDER   = ['SR_1005', 'SR_530', 'SR_1050', 'SR_1130', 'SR_1170',
              'SR_1190', 'SR_1203', 'SR_2170', r'concat\_all']
TASK_ORDER = ['H_vs_PC', 'H_vs_KC_BC_PC']
TASK_LABELS_TEX = {
    'H_vs_PC':       r'H vs.\ PCa',
    'H_vs_KC_BC_PC': r'H vs.\ KC+BC+PCa',
}


# ── Table 1: All configurations ───────────────────────────────────────────────
print("=" * 70)
print("TABLE 1: All configurations")
print("=" * 70)
print()
print(r'\begin{table}[h!]')
print(r'\centering')
print(r'\small')
print(r'\begin{tabular}{llccccc}')
print(r'\hline')
print(r'\textbf{Task} & \textbf{SR} & $\boldsymbol{n}$ & $\boldsymbol{\sigma}$'
      r' & $\boldsymbol{C}$ & $\boldsymbol{\gamma}$ & \textbf{CV Acc.} \\')
print(r'\hline')

for task in TASK_ORDER:
    task_df  = df[df['task'] == task].copy()
    best_acc = task_df['cv_accuracy'].max()

    singles = task_df[task_df['sr_mode'] == 'single'].copy()
    singles['_order'] = singles['sr'].map({s: i for i, s in enumerate(SR_ORDER)})
    singles = singles.sort_values('_order')
    concat  = task_df[task_df['sr_mode'] == 'concat']
    rows    = pd.concat([singles, concat], ignore_index=True)

    label = TASK_LABELS_TEX[task]
    print(r'\multirow{' + str(len(rows)) + r'}{*}{' + label + r'}')
    for _, row in rows.iterrows():
        sr    = row['sr']
        sigma = int(row['best_sigma'])
        C     = row['best_C']
        gamma = row['best_gamma']
        cv    = row['cv_accuracy']
        n     = int(row['n_samples'])
        if cv == best_acc:
            sr_str = r'\textbf{' + sr + r'}'
            cv_str = r'\textbf{' + f'{cv:.3f}' + r'}'
        else:
            sr_str = sr
            cv_str = f'{cv:.3f}'
        print(f'  & {sr_str} & {n} & {sigma} & {C:.4f} & {gamma} & {cv_str} \\\\')
    print(r'\hline')

print(r'\end{tabular}')
print(r'\caption{Best classical SVM 5-fold cross-validation parameters for each '
      r'configuration. $n$ = total number of samples, $\sigma$ = Gaussian '
      r'smoothing width, $C$ = regularisation parameter, $\gamma$ = RBF kernel '
      r'bandwidth (\texttt{scale} = $1/(n\_features \cdot \text{Var}(X))$), '
      r'CV Acc.\ = stratified 5-fold CV accuracy. PCA-4 features, RBF kernel. '
      r'Bold = best per task.}')
print(r'\label{tab:svm_all_configs}')
print(r'\end{table}')


# ── Table 2: Best config per task summary ─────────────────────────────────────
print()
print("=" * 70)
print("TABLE 2: Best config per task (with LOOCV sens/spec)")
print("=" * 70)
print()

eval_df = pd.read_csv('../eval_result_data/svm_evaluation_results.csv')
loocv = eval_df[(eval_df['method'] == 'LOOCV') & (eval_df['pca'] == 1)][
    ['config_id', 'sensitivity', 'specificity', 'balanced_accuracy']
]
df_merged = df.merge(loocv, on='config_id', how='left')
best = (df_merged.sort_values('cv_accuracy', ascending=False)
        .groupby('task').first()
        .reindex(TASK_ORDER))

print(r'\begin{table}[h!]')
print(r'\centering')
print(r'\small')
print(r'\begin{tabular}{|l|c|c|c|c|c|c|c|c|}')
print(r'\hline')
print(r'\textbf{Task} & \textbf{SR} & $\boldsymbol{n}$ & $\boldsymbol{\sigma}$'
      r' & $\boldsymbol{C}$ & $\boldsymbol{\gamma}$ & \textbf{CV Acc.} & \textbf{Sens.} '
      r'& \textbf{Spec.} \\')
print(r'\hline')
for task in TASK_ORDER:
    row   = best.loc[task]
    label = TASK_LABELS_TEX[task]
    print(f'{label} & {row["sr"]} & {int(row["n_samples"])} & '
          f'{int(row["best_sigma"])} & {row["best_C"]:.4f} & {row["best_gamma"]} & '
          f'{row["cv_accuracy"]:.3f} & {row["sensitivity"]:.3f} & '
          f'{row["specificity"]:.3f} \\\\')
print(r'\hline \hline')
print(r'\end{tabular}')
print(r'\caption{Optimal classical SVM configuration per task from stratified '
      r'5-fold cross-validation over $(\sigma, C, \gamma)$. $n$ = total samples, '
      r'$\sigma$ = Gaussian smoothing width, $C$ = regularisation parameter, '
      r'$\gamma$ = RBF kernel bandwidth. '
      r'Sensitivity and specificity from LOOCV on the selected configuration.}')
print(r'\label{tab:svm_best_config_summary}')
print(r'\end{table}')
