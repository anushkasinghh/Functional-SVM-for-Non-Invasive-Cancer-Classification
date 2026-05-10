"""
Generate LaTeX table: FSVC blind test performance for all 18 configurations.
Outputs to stdout — redirect to a .tex file if needed.

Gap = LOOCV accuracy minus blind accuracy.
Degenerate: predicts cancer for all patients (sensitivity=1, specificity=0).
"""

import pandas as pd
import numpy as np
from scipy.stats import beta as beta_dist

BLIND_DATA = '../eval_result_data/fsvc_sr_blind_results.csv'
EVAL_DATA  = '../eval_result_data/fsvc_sr_evaluation_results.csv'

# Blind set composition (caption: 4 healthy in both tasks)
N_NEG_BLIND = 4
BLIND_N_POS = {'H_vs_PC': 11, 'H_vs_KC_BC_PC': 19}  # 15-4=11, 23-4=19

SR_ORDER = ['SR_1005', 'SR_530', 'SR_1050', 'SR_1130', 'SR_1170',
            'SR_1190', 'SR_1203', 'SR_2170', 'concat_all']
TASK_ORDER = ['H_vs_PC', 'H_vs_KC_BC_PC']
TASK_LABELS = {
    'H_vs_PC':       r'H vs.\ PC',
    'H_vs_KC_BC_PC': r'H vs.\ KC+BC+PC',
}
SR_LABELS = {s: s.replace('_', r'\_') for s in SR_ORDER}
SR_LABELS['concat_all'] = r'\textit{concat\_all}'


def clopper_pearson(k, n, alpha=0.05):
    """Exact Clopper-Pearson 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    lo = beta_dist.ppf(alpha / 2,     k,     n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)     if k < n else 1.0
    return (lo, hi)


def fmt_acc(val):
    return f'{val:.3f}'



def bold(s):
    return r'\textbf{' + s + '}'


# ── Load data ──────────────────────────────────────────────────────────────────
blind = pd.read_csv(BLIND_DATA)
blind['sr'] = blind['config_id'].str.split('__').str[0]

loocv = pd.read_csv(EVAL_DATA)
loocv = loocv[loocv['method'] == 'LOOCV'].copy()
loocv['sr'] = loocv['config_id'].str.split('__').str[0]

# ── Pre-compute Clopper-Pearson CIs and MCC for blind set ────────────────────
def add_blind_stats(row):
    task  = row['task']
    n_pos = BLIND_N_POS[task]
    n_neg = N_NEG_BLIND

    k_sens = round(row['sensitivity'] * n_pos)
    k_spec = round(row['specificity'] * n_neg)

    sens_lo, sens_hi = clopper_pearson(k_sens, n_pos)
    spec_lo, spec_hi = clopper_pearson(k_spec, n_neg)

    row['sens_ci_lo'] = sens_lo
    row['sens_ci_hi'] = sens_hi
    row['spec_ci_lo'] = spec_lo
    row['spec_ci_hi'] = spec_hi

    TP = k_sens
    TN = k_spec
    FN = n_pos - TP
    FP = n_neg - TN
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    row['mcc'] = (TP * TN - FP * FN) / denom if denom > 0 else 0.0

    # Verify: reconstructed accuracy must match the CSV accuracy column
    reconstructed_acc = (TP + TN) / row['n_blind']
    if not np.isclose(reconstructed_acc, row['accuracy'], atol=1e-6):
        raise ValueError(
            f"Accuracy mismatch for {row['config_id']}: "
            f"reconstructed={reconstructed_acc:.6f}, csv={row['accuracy']:.6f}. "
            f"Check n_pos/n_neg assumptions."
        )

    return row

blind = blind.apply(add_blind_stats, axis=1)

# ── Identify degenerate configs (blind sens=1, spec=0) ────────────────────────
def is_degenerate(row):
    return np.isclose(row['sensitivity'], 1.0) and np.isclose(row['specificity'], 0.0)

blind['degenerate'] = blind.apply(is_degenerate, axis=1)

# ── Best blind accuracy per task (for bolding) ────────────────────────────────
best_blind = {}
for task in TASK_ORDER:
    sub = blind[blind['task'] == task]
    best_sr = sub.loc[sub['accuracy'].idxmax(), 'sr']
    best_blind[task] = best_sr

# ── Build LaTeX ───────────────────────────────────────────────────────────────
lines = []
lines.append(r'\begin{table}[ht]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\setlength{\tabcolsep}{4pt}')
lines.append(r'\begin{tabular}{|l|c|c|c|c|c|c|}')
lines.append(r'\hline')
lines.append(
    r'\textbf{SR} & \textbf{LOOCV Acc.} & \textbf{Blind Acc.} & \textbf{Gap} '
    r'& \textbf{Blind Sens.} & \textbf{Blind Spec.} & \textbf{MCC} \\'
)
lines.append(r'\hline')

for task in TASK_ORDER:
    # Task header
    lines.append(
        r'\multicolumn{7}{|l|}{\textit{' + TASK_LABELS[task] + r'}} \\'
    )
    lines.append(r'\hline')

    b = blind[blind['task'] == task].set_index('sr')
    l = loocv[loocv['task'] == task].set_index('sr')

    for sr in SR_ORDER:
        br = b.loc[sr]
        lr = l.loc[sr]

        loocv_acc = lr['accuracy']
        blind_acc = br['accuracy']
        gap       = loocv_acc - blind_acc

        sens_str = fmt_acc(br['sensitivity'])
        spec_str = fmt_acc(br['specificity'])

        dagger = r'$^\dagger$' if br['degenerate'] else ''

        acc_blind_str = fmt_acc(blind_acc) + dagger
        gap_str       = f'{gap:+.3f}'

        mcc_str = fmt_acc(br['mcc'])

        # Bold the row with best blind accuracy per task
        if sr == best_blind[task] and not br['degenerate']:
            acc_blind_str = bold(fmt_acc(blind_acc)) + dagger
            gap_str       = bold(gap_str)
            sens_str      = bold(sens_str)
            spec_str      = bold(spec_str)
            mcc_str       = bold(mcc_str)

        row = (
            f'{SR_LABELS[sr]} & '
            f'{fmt_acc(loocv_acc)} & '
            f'{acc_blind_str} & '
            f'{gap_str} & '
            f'{sens_str} & '
            f'{spec_str} & '
            f'{mcc_str} \\\\'
        )
        lines.append(row)

    lines.append(r'\hline')

lines.append(r'\end{tabular}')
lines.append(
    r'\caption{FSVC blind test performance for all 18 configurations. '
    r'Gap = LOOCV accuracy minus blind accuracy. '
    r'$^\dagger$Degenerate: predicts cancer for all patients. '
    r'The blind set contains only 4 healthy patients in both tasks; '
    r'specificity estimates should be interpreted with caution.}'
)
lines.append(r'\label{tab:FSVM_blind_results}')
lines.append(r'\end{table}')

print('\n'.join(lines))
