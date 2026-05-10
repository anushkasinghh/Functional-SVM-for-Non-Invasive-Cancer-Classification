"""
Generate LaTeX blind test table for classical SVM (PCA-4 configs).
Format matches FSVC blind test table.
"""

import pandas as pd
import numpy as np

DATA     = '../eval_result_data/svm_blind_results_all.csv'
SR_ORDER = ['SR_1005','SR_530','SR_1050','SR_1130','SR_1170','SR_1190','SR_1203','SR_2170','all']
TASK_ORDER  = ['H_vs_PC', 'H_vs_KC_BC_PC']
TASK_LABELS = {'H_vs_PC': r'H vs.\ PC', 'H_vs_KC_BC_PC': r'H vs.\ KC+BC+PC'}
SR_LABELS   = {s: s.replace('_', r'\_') for s in SR_ORDER}
SR_LABELS['all'] = r'\textit{concat\_all}'

df = pd.read_csv(DATA).set_index(['task', 'sr_used'])

def fmt_ci(lo, hi):
    return f'[{lo:.2f},\\,{hi:.2f}]'

def is_degenerate(row):
    # spec=0: all healthy predicted as cancer
    # spec=1 and sens=0: all cancer predicted as healthy
    return row['specificity'] == 0.0 or (row['specificity'] == 1.0 and row['sensitivity'] == 0.0)

def bold(s):
    return r'\textbf{' + s + '}'

lines = []
lines.append(r'\begin{table}[h!]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\setlength{\tabcolsep}{3pt}')
lines.append(r'\begin{tabular}{|l|cccc|cc|}')
lines.append(r'\hline')
lines.append(
    r'\textbf{SR} & '
    r'\textbf{LOOCV Acc.} & \textbf{Blind Acc.} & \textbf{Gap} & \textbf{MCC} & '
    r'\textbf{Blind Sens.\ [95\% CI]} & \textbf{Blind Spec.\ [95\% CI]} \\'
)
lines.append(r'\hline')

for task in TASK_ORDER:
    lines.append(r'\multicolumn{7}{|l|}{\textit{' + TASK_LABELS[task] + r'}} \\')
    lines.append(r'\hline')

    # best non-degenerate row = highest blind accuracy
    task_rows = []
    for sr in SR_ORDER:
        if (task, sr) in df.index:
            task_rows.append((sr, df.loc[(task, sr)]))
    non_deg = [(sr, r) for sr, r in task_rows if not is_degenerate(r)]
    best_sr = max(non_deg, key=lambda x: x[1]['accuracy'])[0] if non_deg else None

    for sr, row in task_rows:
        deg   = is_degenerate(row)
        loocv = f'{row["loocv_accuracy"]:.3f}'
        acc   = f'{row["accuracy"]:.3f}'
        gap   = f'{row["gap"]:+.3f}'
        mcc   = f'{row["mcc"]:.3f}'
        sens  = f'{row["sensitivity"]:.3f}\ {fmt_ci(row["sens_ci_lo"], row["sens_ci_hi"])}'
        spec  = f'{row["specificity"]:.3f}\ {fmt_ci(row["spec_ci_lo"], row["spec_ci_hi"])}'

        if sr == best_sr:
            loocv = bold(loocv)
            acc   = bold(acc)
            gap   = bold(gap)
            mcc   = bold(mcc)

        dag = r'$^\dagger$' if deg else ''
        lines.append(
            f'{SR_LABELS[sr]}{dag} & '
            f'${loocv}$ & ${acc}$ & ${gap}$ & ${mcc}$ & ${sens}$ & ${spec}$ \\\\'
        )

    lines.append(r'\hline')

lines.append(r'\end{tabular}')
lines.append(
    r'\caption{Classical SVM (PCA-4) blind test performance for all 18 configurations. '
    r'Gap = LOOCV accuracy $-$ blind accuracy. '
    r'$^\dagger$Degenerate: classifier assigns all patients to one class. '
    r'Confidence intervals are Clopper-Pearson exact 95\% CIs. '
    r'The blind set contains only 4 healthy patients in both tasks; '
    r'specificity estimates should be interpreted with caution. '
    r'Bold = best non-degenerate configuration by MCC per task.}'
)
lines.append(r'\label{tab:svm_blind_results}')
lines.append(r'\end{table}')

print('\n'.join(lines))
