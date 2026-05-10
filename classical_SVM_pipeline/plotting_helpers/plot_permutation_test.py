"""
Permutation test for Classical SVM LOOCV accuracy.

For each task, selects the best config by LOOCV accuracy (PCA-4, single SR)
from svm_evaluation_results.csv, then re-runs LOOCV with N_PERM randomly
shuffled label vectors. The fraction of permutations that match or exceed
the observed LOOCV accuracy is the permutation p-value.

Config selection is done by LOOCV accuracy (not 9-fold CV), so the
permutation test and the model selection criterion are consistent.

Note: PCA and Gaussian smoothing are unsupervised — they do not use class
labels. Only the SVM's training labels are permuted, which is correct.

Outputs
-------
  plots/svm_permutation_test.png
  svm_permutation_null_distributions.csv
"""

import sys, os
sys.path.insert(0, '..')

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from SVM_implement import SVMBreathClassifier
from sr_preprocessing import extract_sr_window, preprocess_sr

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = '../../data_processed/breath_data.pkl'
OUT          = '../eval_result_data/plots'
CSV_OUT      = '../eval_result_data/svm_permutation_null_distributions.csv'
N_PERM       = 1000
RNG_SEED     = 42

os.makedirs(OUT, exist_ok=True)

SR_CENTERS = {
    'SR_1005': 1005, 'SR_530': 530,   'SR_1050': 1050, 'SR_1130': 1130,
    'SR_1170': 1170, 'SR_1190': 1190, 'SR_1203': 1203, 'SR_2170': 2170,
}

# Class membership per task (fixed by experiment design)
TASK_CLASSES = {
    'H_vs_PC':       ['H', 'PC'],
    'H_vs_KC_BC_PC': ['H', 'KC', 'BC', 'PC'],
}

# ── Load best configs dynamically from LOOCV evaluation results ───────────────
# Selects the single-SR, PCA-4 config with highest LOOCV accuracy per task.
# This keeps model selection and permutation test on the same metric.
EVAL_CSV = '../eval_result_data/svm_evaluation_results.csv'

eval_df = pd.read_csv(EVAL_CSV)
if eval_df.empty:
    raise RuntimeError(
        f"{EVAL_CSV} is empty — rerun SVM_notebook.ipynb (cells 29 onward) first."
    )

loocv_pca = eval_df[
    (eval_df['method'] == 'LOOCV') &
    (eval_df['sr_type'] == 'single') &
    (eval_df['pca'].astype(bool))
].copy()

BEST_CONFIGS = {}
for task in TASK_CLASSES:
    task_rows = loocv_pca[loocv_pca['task'] == task]
    if task_rows.empty:
        raise RuntimeError(f"No LOOCV PCA-4 single-SR results found for task '{task}'.")
    best = task_rows.loc[task_rows['accuracy'].idxmax()]
    print(f"[{task}] Best LOOCV: {best['sr_used']}  "
          f"acc={best['accuracy']:.4f}  "
          f"sigma={best['sigma']}  C={best['C']}  gamma={best['gamma']}")
    BEST_CONFIGS[task] = {
        'sr':      best['sr_used'],
        'classes': TASK_CLASSES[task],
        'params': {
            'sigma':            best['sigma'],
            'kernel':           best['kernel'],
            'C':                best['C'],
            'gamma':            best['gamma'],
            'feature_type':     'pca',
            'n_pca_components': 4,
        },
        'observed_loocv': best['accuracy'],
    }

TASK_LABELS = {
    'H_vs_PC':       'Task I — H vs PC',
    'H_vs_KC_BC_PC': 'Task II — H vs KC+BC+PC',
}

# ── Load data ─────────────────────────────────────────────────────────────────
with open(DATA_PATH, 'rb') as f:
    df_raw = pickle.load(f)

df_raw['infoP'] = df_raw['infoP'].apply(lambda x: 'H' if x in ['M', 'F', 'H'] else x)
df_raw = df_raw.drop_duplicates(subset='original_filename').reset_index(drop=True)
df_train = df_raw[df_raw['category'] != 'blinddata'].reset_index(drop=True)


def get_sr_matrix(df, sr_name):
    center = SR_CENTERS[sr_name]
    rows = []
    for _, row in df.iterrows():
        sr_spec, _ = extract_sr_window(
            row['intensity_baseline_corrected'],
            row['wavenumber'],
            center=center, window_width=30.0
        )
        rows.append(preprocess_sr(sr_spec)['preprocessed'])
    return np.array(rows)


def loocv_accuracy(X, y, params):
    """Run LOOCV and return accuracy only (fast path for permutation loop)."""
    clf = SVMBreathClassifier()
    result = clf.loocv_validation(X, y, params)
    return result['accuracy']


# ── Run permutation tests ─────────────────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)
results = {}

for task, cfg in BEST_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"  SR={cfg['sr']}, params={cfg['params']}")
    print(f"  Observed LOOCV accuracy: {cfg['observed_loocv']:.4f}")

    # Build X, y for this task
    mask    = df_train['infoP'].isin(cfg['classes'])
    df_task = df_train[mask].reset_index(drop=True)
    X       = get_sr_matrix(df_task, cfg['sr'])
    y_orig  = df_task['infoP'].values.copy()
    # Collapse multi-cancer labels
    y = np.where(y_orig == 'H', 'H', 'cancer')

    n = len(y)
    print(f"  n={n}, classes={dict(zip(*np.unique(y, return_counts=True)))}")

    # Recompute observed accuracy to verify it matches the CSV value
    obs_acc = loocv_accuracy(X, y, cfg['params'])
    delta = abs(obs_acc - cfg['observed_loocv'])
    print(f"  Recomputed LOOCV: {obs_acc:.4f}  (CSV: {cfg['observed_loocv']:.4f}"
          f"{'  ✓' if delta < 0.01 else f'  WARNING: delta={delta:.4f}'})")

    # Permutation loop
    null_accs = np.empty(N_PERM)
    for p in range(N_PERM):
        y_perm       = rng.permutation(y)
        null_accs[p] = loocv_accuracy(X, y_perm, cfg['params'])
        if (p + 1) % 100 == 0:
            print(f"  Permutation {p+1}/{N_PERM}  "
                  f"(running mean null acc = {null_accs[:p+1].mean():.3f})")

    # p-value: proportion of permutations >= observed (including observed itself)
    p_value = (np.sum(null_accs >= obs_acc) + 1) / (N_PERM + 1)

    print(f"\n  Null distribution: mean={null_accs.mean():.3f}, "
          f"std={null_accs.std():.3f}, max={null_accs.max():.3f}")
    print(f"  p-value = {p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'NOT significant'} at alpha=0.05)")

    results[task] = {
        'observed_acc': obs_acc,
        'null_accs':    null_accs,
        'p_value':      p_value,
    }

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (task, res) in zip(axes, results.items()):
    cfg      = BEST_CONFIGS[task]
    null     = res['null_accs']
    obs      = res['observed_acc']
    p_val    = res['p_value']

    ax.hist(null, bins=30, color='#90A4AE', edgecolor='white',
            linewidth=0.5, alpha=0.85, label='Null distribution')
    ax.axvline(obs, color='#E53935', lw=2.0, ls='-',
               label=f'Observed LOOCV acc = {obs:.3f}')
    pct95 = np.percentile(null, 95)
    ax.axvline(pct95, color='#FB8C00', lw=1.5, ls='--',
               label=f'95th percentile = {pct95:.3f}')

    # Shade the tail beyond observed
    if obs <= null.max():
        ax.axvspan(obs, null.max() + 0.01, alpha=0.15, color='#E53935')

    p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
    ax.text(0.97, 0.05, p_str,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold',
            color='#B71C1C' if p_val < 0.05 else '#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.set_xlabel('LOOCV Accuracy under permuted labels', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(
        f'{TASK_LABELS[task]}\n'
        f'{cfg["sr"]}, PCA-4, RBF, $C={cfg["params"]["C"]}$, '
        f'$\\sigma={cfg["params"]["sigma"]}$',
        fontsize=10
    )
    ax.legend(fontsize=9, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)

    null_mean = null.mean()
    ax.text(0.03, 0.05,
            f'Null mean = {null_mean:.3f}\n'
            f'N permutations = {N_PERM}',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8.5, color='#555555')

plt.suptitle(
    'Classical SVM — Permutation Test (LOOCV Accuracy)',
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
out_path = os.path.join(OUT, 'svm_permutation_test.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")

# ── Save null distributions to CSV ───────────────────────────────────────────
rows = []
for task, res in results.items():
    for acc in res['null_accs']:
        rows.append({'task': task, 'perm_accuracy': acc})

df_null = pd.DataFrame(rows)

# Append summary rows
summary_rows = []
for task, res in results.items():
    summary_rows.append({
        'task':            task,
        'observed_acc':    res['observed_acc'],
        'p_value':         res['p_value'],
        'null_mean':       res['null_accs'].mean(),
        'null_std':        res['null_accs'].std(),
        'null_max':        res['null_accs'].max(),
        'n_permutations':  N_PERM,
    })

df_summary = pd.DataFrame(summary_rows)

with open(CSV_OUT.replace('.csv', '_summary.csv'), 'w') as f:
    df_summary.to_csv(f, index=False)
df_null.to_csv(CSV_OUT, index=False)

print(f"Saved: {CSV_OUT}")
print(f"Saved: {CSV_OUT.replace('.csv', '_summary.csv')}")

print("\n=== Summary ===")
for row in summary_rows:
    sig = 'SIGNIFICANT' if row['p_value'] < 0.05 else 'not significant'
    print(f"  {row['task']}: obs={row['observed_acc']:.4f}, "
          f"p={row['p_value']:.4f} ({sig})")
