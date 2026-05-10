"""
Section 5.8 — Validation and Diagnostics: Decision Score Distributions
Produces one figure with two panels (one per task) for SR_1005 (best config).

For each task, re-runs LOOCV and collects the SVM decision function value
for every patient. The decision function is the signed distance from the
hyperplane: positive = predicted cancer, negative = predicted healthy.

Panel layout per task:
  - Strip plot of decision scores grouped by true class
  - Misclassified patients marked separately
  - Decision boundary at 0
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'classical_SVM_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from sr_preprocessing import preprocess_all_srs
from fsvm_implement import fpca_face_via_r, estimate_pc_scores, compute_gamma_automatic, run_fpca

OUT = '../eval_result_data/plots'
os.makedirs(OUT, exist_ok=True)

# ── Best config (from Table best-config-fsvm) ──────────────────────────────
# BEST_SR    = 'SR_1005'
CONFIGS = {
    'H_vs_PC':        {'BEST_SR': 'SR_1190', 'opt_tau': 5.0, 'opt_K': 1, 'opt_C': 0.7525,    'classes': ['H', 'PC']},
    'H_vs_KC_BC_PC':  {'BEST_SR': 'SR_1005', 'opt_tau': 10.0, 'opt_K': 2, 'opt_C': 1.000, 'classes': ['H', 'PC', 'KC', 'BC']},
}
TASK_LABELS = {
    'H_vs_PC':       'Task I — H vs PC',
    'H_vs_KC_BC_PC': 'Task II — H vs PC+KC+BC',
}
CLASS_COLORS = {
    'H':      '#2196F3',
    'PC':     '#E53935',
    'cancer': '#E53935',
}

# ── Load and preprocess data ───────────────────────────────────────────────
df_raw = pd.read_pickle(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data_processed', 'breath_data.pkl')
)
df_all = df_raw[df_raw['category'] != 'blinddata'].copy()
df_all.loc[df_all['infoP'].isin(['M', 'F']), 'infoP'] = 'H'
df_all = df_all.reset_index(drop=True)

# Preprocess all SRs once
sr_results = [
    preprocess_all_srs(
        np.array(df_all['intensity_baseline_corrected'].iloc[i]),
        np.array(df_all['wavenumber'].iloc[i])
    )
    for i in range(len(df_all))
]
for sr in sr_results[0].keys():
    df_all[f'{sr}_preprocessed'] = [sr_results[i][sr]['spectrum'] for i in range(len(df_all))]

print(f"Training set: n={len(df_all)},  infoP: {dict(df_all['infoP'].value_counts())}")


# ── Modified LOOCV that returns per-patient decision scores ────────────────
def loocv_with_scores(X, y, opt_tau, opt_K, opt_C, use_r=True):
    """
    LOOCV collecting SVM decision function value for each left-out patient.

    Returns
    -------
    df_scores : pd.DataFrame
        Columns: true_label, pred_label, correct, decision_score
    """
    n = len(y)
    records = []

    for i in range(n):
        mask   = np.arange(n) != i
        X_tr, y_tr = X[mask], y[mask]
        X_te        = X[[i]]

        # FPCA on training fold
        npc  = min(opt_K + 2, X_tr.shape[0] - 1, X_tr.shape[1] - 1)
        fpca = run_fpca(X_tr, npc=npc, lam=opt_tau)

        # Training scores (direct from fpca.face for train, BLUP for test)
        S_tr = fpca.scores[:, :opt_K]
        S_te = estimate_pc_scores(
            X_te, fpca.mu, fpca.sigma2, fpca.evalues, fpca.efunctions
        )[:, :opt_K]

        # Fit SVM
        gamma = compute_gamma_automatic(S_tr)
        svm   = SVC(kernel='rbf', C=opt_C, gamma=gamma, decision_function_shape='ovr')
        svm.fit(S_tr, y_tr)

        pred   = svm.predict(S_te)[0]
        # decision_function: positive = predicted as svm.classes_[1]
        # We want positive = cancer, so orient so that H is class 0
        raw_score = svm.decision_function(S_te)[0]
        classes   = list(svm.classes_)
        # SVC: decision_function > 0 → classes[1]
        # We want positive = cancer (non-H), so flip only when classes[1] == 'H'
        if classes[1] == 'H':
            raw_score = -raw_score

        records.append({
            'true_label':    y[i],
            'pred_label':    pred,
            'correct':       y[i] == pred,
            'decision_score': raw_score,
        })
        if (i + 1) % 10 == 0:
            print(f"  LOOCV {i+1}/{n}")

    return pd.DataFrame(records)


# ── Run LOOCV for each task ────────────────────────────────────────────────
results = {}

for task, cfg in CONFIGS.items():
    print(f"\nRunning LOOCV: {task} ...")
    classes = cfg['classes']
    mask    = df_all['infoP'].isin(classes)
    df_task = df_all[mask].reset_index(drop=True)

    y = df_task['infoP'].values.copy()
    if len(classes) > 2:
        y = np.where(y == 'H', 'H', 'cancer')

    X = np.array(list(df_task[f'{cfg["BEST_SR"]}_preprocessed']))
    print(f"  n={len(X)}, classes={np.unique(y, return_counts=True)}")

    df_scores = loocv_with_scores(
        X, y,
        opt_tau=cfg['opt_tau'],
        opt_K=cfg['opt_K'],
        opt_C=cfg['opt_C'],
    )
    # attach patient identifiers from df_task
    df_scores['patient_id']        = df_task['patient_id'].values
    df_scores['original_filename'] = df_task['original_filename'].values
    df_scores['infoP_original']    = df_task['infoP'].values   # pre-collapse label
    df_scores['category']          = df_task['category'].values
    df_scores['task']              = task
    results[task] = df_scores

    acc  = df_scores['correct'].mean()
    sens = df_scores[df_scores['true_label'] != 'H']['correct'].mean()
    spec = df_scores[df_scores['true_label'] == 'H']['correct'].mean()
    print(f"  Acc={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")
    misclf = df_scores[~df_scores['correct']]
    print(f"  Misclassified ({len(misclf)}): {dict(misclf['true_label'].value_counts())}")


# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, (task, df_sc) in zip(axes, results.items()):
    cfg = CONFIGS[task]

    # Separate by true label
    true_classes = sorted(df_sc['true_label'].unique(),
                          key=lambda c: (0 if c == 'H' else 1))
    class_colors_local = {'H': '#2196F3', 'PC': '#E53935', 'cancer': '#E53935'}

    jitter_rng = np.random.default_rng(42)
    y_positions = {'H': 0, 'PC': 1, 'cancer': 1}

    for cls in true_classes:
        sub = df_sc[df_sc['true_label'] == cls]
        correct   = sub[sub['correct']]
        incorrect = sub[~sub['correct']]

        ypos = y_positions[cls]
        jitter_c = jitter_rng.uniform(-0.12, 0.12, len(correct))
        jitter_e = jitter_rng.uniform(-0.12, 0.12, len(incorrect))

        ax.scatter(correct['decision_score'],   ypos + jitter_c,
                   color=class_colors_local[cls], s=55, alpha=0.75,
                   edgecolors='white', linewidths=0.5, zorder=3)
        ax.scatter(incorrect['decision_score'], ypos + jitter_e,
                   color='black', s=80, marker='X', alpha=0.9, zorder=5,
                   label='_nolegend_')

    # Decision boundary
    ax.axvline(0, color='black', ls='-', lw=1.2, label='Decision boundary', zorder=2)

    # Grey zone
    GREY_ZONE = 0.2
    ax.axvspan(-GREY_ZONE, GREY_ZONE, alpha=0.12, color='#888888',
               label=f'Grey zone ($|d|<{GREY_ZONE}$)', zorder=0)

    # Shading
    ax.axvspan(-2, 0, alpha=0.04, color='#2196F3', zorder=0)
    ax.axvspan(0, 2, alpha=0.04, color='#E53935', zorder=0)

    # Axis labels and ticks
    ax.set_yticks([0, 1])
    ax.set_yticklabels(true_classes, fontsize=12)
    ax.set_xlabel('SVM Decision Score (distance from hyperplane)', fontsize=11)
    ax.set_title(
        f'{TASK_LABELS[task]}\n'
        f'{cfg["BEST_SR"]}, $\\tau^*={int(cfg["opt_tau"])}$, $K^*={cfg["opt_K"]}$, $C^*={cfg["opt_C"]}$',
        fontsize=11
    )
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 1.5)

    # Annotation: counts
    for cls in true_classes:
        sub     = df_sc[df_sc['true_label'] == cls]
        n_wrong = (~sub['correct']).sum()
        ypos    = y_positions[cls]
        ax.text(1.95, ypos + 0.22,
                f'n={len(sub)}\n{n_wrong} error{"s" if n_wrong != 1 else ""}',
                ha='right', va='center', fontsize=8.5,
                color=class_colors_local[cls])

    # Legend
    legend_handles = [
        mpatches.Patch(color='#2196F3', alpha=0.85, label='Healthy (H)'),
        mpatches.Patch(color='#E53935', alpha=0.85,
                       label='PC' if task == 'H_vs_PC' else 'Cancer (PC+KC+BC)'),
        plt.scatter([], [], color='black', marker='X', s=60,
                    label='Misclassified'),
        plt.Line2D([0], [0], color='black', ls='-', lw=1.2,
                   label='Decision boundary'),
        mpatches.Patch(color='#888888', alpha=0.3,
                       label=f'Grey zone ($|d|<{GREY_ZONE}$)'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper left')

plt.suptitle(
    'FSVC LOOCV — Per-Patient Decision Scores',
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig(f'{OUT}/fsvc_5_8_decision_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: fsvc_5_8_decision_scores.png")

# ── Save per-patient results to CSV ───────────────────────────────────────
all_results = pd.concat(results.values(), ignore_index=True)
csv_cols = ['task', 'patient_id', 'original_filename', 'infoP_original',
            'true_label', 'pred_label', 'correct', 'decision_score', 'category']
all_results[csv_cols].to_csv(
    '../eval_result_data/fsvc_loocv_per_patient.csv', index=False
)
print("Saved: fsvc_loocv_per_patient.csv")

# ── Diagnostics for write-up ───────────────────────────────────────────────
print("\n=== Misclassified patients ===")
for task, df_sc in results.items():
    print(f"\n{TASK_LABELS[task]}")
    misclf = df_sc[~df_sc['correct']][
        ['patient_id', 'original_filename', 'infoP_original',
         'true_label', 'pred_label', 'decision_score']
    ].sort_values('decision_score')
    print(misclf.to_string(index=False))
    print(f"  Score range correct:   [{df_sc[df_sc['correct']]['decision_score'].min():.3f}, "
          f"{df_sc[df_sc['correct']]['decision_score'].max():.3f}]")
    for cls in sorted(df_sc['true_label'].unique()):
        sub = df_sc[df_sc['true_label'] == cls]
        print(f"  {cls}: mean={sub['decision_score'].mean():.3f} "
              f"std={sub['decision_score'].std():.3f}")
