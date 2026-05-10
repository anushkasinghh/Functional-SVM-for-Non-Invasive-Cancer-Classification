"""
Generate LaTeX table: LOOCV + 9-fold CV results for all 18 FSVC configurations.
Outputs to stdout — redirect to a .tex file if needed.
"""

import pandas as pd
import numpy as np

DATA = '../eval_result_data/fsvc_sr_evaluation_results.csv'

SR_ORDER   = ['SR_1005','SR_530','SR_1050','SR_1130','SR_1170',
              'SR_1190','SR_1203','SR_2170','concat_all']
TASK_ORDER = ['H_vs_PC', 'H_vs_KC_BC_PC']
TASK_LABELS = {
    'H_vs_PC':        r'H vs.\ PC',
    'H_vs_KC_BC_PC':  r'H vs.\ KC+BC+PC',
}
SR_LABELS = {s: s.replace('_', r'\_') for s in SR_ORDER}
SR_LABELS['concat_all'] = r'\textit{concat\_all}'

df    = pd.read_csv(DATA)
loocv = df[df['method'] == 'LOOCV'].copy()
kfold = df[df['method'] == '9-fold (x10)'].copy()
for d in (loocv, kfold):
    d['sr'] = d['config_id'].str.split('__').str[0]

def fmt(val, std=None):
    if std is not None:
        return f'{val:.3f}$\\pm${std:.3f}'
    return f'{val:.3f}'

def bold(s):
    return r'\textbf{' + s + '}'

# Find best LOOCV accuracy per task (for bolding)
best_loocv = (loocv.groupby('task')['accuracy'].idxmax()
                   .apply(lambda i: loocv.loc[i, 'sr']))

lines = []
lines.append(r'\begin{table}[h!]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\setlength{\tabcolsep}{4pt}')
lines.append(r'\begin{tabular}{|l|ccc|ccc|}')
lines.append(r'\hline')
lines.append(
    r'\textbf{SR} & '
    r'\multicolumn{3}{c|}{\textbf{LOOCV}} & '
    r'\multicolumn{3}{c|}{\textbf{9-fold CV (mean$\pm$std, $\times$10)}} \\'
)
lines.append(
    r' & \textbf{Acc.} & \textbf{Sens.} & \textbf{Spec.} '
    r'& \textbf{Acc.} & \textbf{Sens.} & \textbf{Spec.} \\'
)
lines.append(r'\hline')

for task in TASK_ORDER:
    lines.append(
        r'\multicolumn{7}{|l|}{\textit{' + TASK_LABELS[task] + r'}} \\'
    )
    lines.append(r'\hline')

    l = loocv[loocv['task'] == task].set_index('sr')
    k = kfold[kfold['task'] == task].set_index('sr')
    best_sr = best_loocv[task]

    for sr in SR_ORDER:
        lr = l.loc[sr]
        kr = k.loc[sr]

        acc_l  = fmt(lr['accuracy'])
        sens_l = fmt(lr['sensitivity'])
        spec_l = fmt(lr['specificity'])

        acc_k  = fmt(kr['accuracy'],    kr['accuracy_std'])
        sens_k = fmt(kr['sensitivity'], kr['sensitivity_std'])
        spec_k = fmt(kr['specificity'], kr['specificity_std'])

        # Bold entire LOOCV row for best config per task
        if sr == best_sr:
            acc_l  = bold(acc_l)
            sens_l = bold(sens_l)
            spec_l = bold(spec_l)

        row = (f'{SR_LABELS[sr]} & '
               f'{acc_l} & {sens_l} & {spec_l} & '
               f'{acc_k} & {sens_k} & {spec_k} \\\\')
        lines.append(row)

    lines.append(r'\hline')

lines.append(r'\end{tabular}')
lines.append(
    r'\caption{FSVC LOOCV and repeated 9-fold CV (10 repetitions) accuracy, '
    r'sensitivity, and specificity for all 18 configurations across both tasks. '
    r'Bold rows indicate the best LOOCV accuracy per task.}'
)
lines.append(r'\label{tab:FSVM_cv_results}')
lines.append(r'\end{table}')

print('\n'.join(lines))
