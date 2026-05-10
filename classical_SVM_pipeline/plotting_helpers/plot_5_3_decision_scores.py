"""
Section 5.3 — Decision Function Outputs and Misclassification Patterns
=======================================================================
For each task (H_vs_PC and H_vs_KC_BC_PC), using the best LOOCV config:
  Left panel  : Per-patient LOOCV decision scores, sorted by [class, score].
                Grey zone (|d| < 0.2) shaded. Misclassified patients circled.
  Right panel : PCA scatter (Comp1 vs Comp2) with misclassified patients
                highlighted — links decision-space errors to feature-space position.

Saves one figure per task:
  svm_5_3_decision_scores_H_vs_PC.png
  svm_5_3_decision_scores_H_vs_KC_BC_PC.png
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

sys.path.insert(0, '..')
from SVM_implement import SVMBreathClassifier
from sr_preprocessing import extract_sr_window, preprocess_sr

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = '../../data_processed/breath_data.pkl'
CONFIGS_PATH = '../eval_result_data/all_configs_best_params.csv'
OUT          = '../eval_result_data/plots'
GREY_ZONE    = 0.2   # |decision| < GREY_ZONE → ambiguous

SR_CENTERS = {
    'SR_1005': 1005, 'SR_530':  530,  'SR_1050': 1050, 'SR_1130': 1130,
    'SR_1170': 1170, 'SR_1190': 1190, 'SR_1203': 1203, 'SR_2170': 2170,
}

TASKS = {
    'H_vs_PC':       ['H', 'PC'],
    'H_vs_KC_BC_PC': ['H', 'KC', 'BC', 'PC'],
}
TASK_LABELS = {
    'H_vs_PC':       'H vs PC',
    'H_vs_KC_BC_PC': 'H vs KC+BC+PC',
}

CLASS_COLORS = {
    'H':      '#4878CF',
    'PC':     '#E84040',
    'KC':     '#F5A623',
    'BC':     '#7B2D8B',
    'cancer': '#D94040',   # collapsed multi-cancer label
}

# ── Load data ─────────────────────────────────────────────────────────────────
with open(DATA_PATH, 'rb') as f:
    df_raw = pickle.load(f)

df_raw['infoP'] = df_raw['infoP'].apply(lambda x: 'H' if x in ['M', 'F', 'H'] else x)
df_raw = df_raw.drop_duplicates(subset='original_filename').reset_index(drop=True)
df_train = df_raw[df_raw['category'] != 'blinddata'].reset_index(drop=True)

# ── Load best configs (PCA-4, best CV acc per task) ───────────────────────────
configs = pd.read_csv(CONFIGS_PATH)
configs = configs[(configs['feature_type'] == 'pca') & (configs['status'] == 'done')]
best_configs = (configs.sort_values('cv_accuracy', ascending=False)
                .groupby('task').first()
                .reset_index())


def get_sr_matrix(df, sr_name):
    center = SR_CENTERS[sr_name]
    rows = []
    for _, row in df.iterrows():
        sr_spec, _ = extract_sr_window(row['intensity_baseline_corrected'],
                                       row['wavenumber'],
                                       center=center, window_width=30.0)
        rows.append(preprocess_sr(sr_spec)['preprocessed'])
    return np.array(rows)


def run_loocv_decisions(X, y, params):
    """Return (decisions, y_pred_bin, y_true_bin) with cancer collapsed."""
    clf = SVMBreathClassifier()
    r = clf.loocv_validation(X, y, params)
    decisions = r['decisions']
    y_pred_bin = np.where(decisions > 0, 'cancer', 'H')
    y_true_bin = np.where(y == 'H', 'H', 'cancer')
    return decisions, y_pred_bin, y_true_bin, r['y_pred']


# ── Plot function ─────────────────────────────────────────────────────────────
def make_figure(task, task_classes, cfg_row):
    sr_name = cfg_row['sr_col'].replace('_preprocessed', '')
    params  = {
        'sigma':            cfg_row['best_sigma'],
        'kernel':           cfg_row['best_kernel'],
        'C':                cfg_row['best_C'],
        'gamma':            cfg_row['best_gamma'],
        'degree':           cfg_row['best_degree'] if cfg_row['best_kernel'] == 'poly' else None,
        'feature_type':     'pca',
        'n_pca_components': int(cfg_row['n_pca_components']),
    }

    # Filter to task classes
    mask    = df_train['infoP'].isin(task_classes)
    df_task = df_train[mask].reset_index(drop=True)
    y_orig  = df_task['infoP'].values
    # Collapse for multi-cancer task
    y_svm   = np.where(y_orig == 'H', 'H', 'cancer')

    print(f"\n[{task}] Best SR: {sr_name} | σ={params['sigma']} C={params['C']}")
    X = get_sr_matrix(df_task, sr_name)

    # ── Run LOOCV ─────────────────────────────────────────────────────────────
    decisions, y_pred_bin, y_true_bin, _ = run_loocv_decisions(X, y_svm, params)
    wrong_mask = y_true_bin != y_pred_bin
    n_wrong = wrong_mask.sum()
    n_total = len(y_svm)
    print(f"  Misclassified: {n_wrong}/{n_total}  ({100*n_wrong/n_total:.1f}%)")
    print(f"  In grey zone (|d|<{GREY_ZONE}): {(np.abs(decisions) < GREY_ZONE).sum()}")

    # ── PCA scores for scatter panel ──────────────────────────────────────────
    # Fit PCA on full task set (for visualisation only — not LOOCV)
    pca_vis = PCA(n_components=4, random_state=42)
    scores  = pca_vis.fit_transform(X)
    var_exp = pca_vis.explained_variance_ratio_

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax_strip, ax_pca) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f'LOOCV Decision Scores and Misclassification Patterns — {TASK_LABELS[task]}\n'
        f'Best config: {sr_name}, PCA-4, RBF, C={params["C"]:.4f}, σ={int(params["sigma"])}',
        fontsize=11, fontweight='bold'
    )

    # ── Panel A: Jitter strip plot (x = class, y = decision score) ──────────────
    jitter_rng  = np.random.default_rng(42)
    y_positions = {'H': 0, 'cancer': 1}

    for cls in ['H', 'cancer']:
        sub_mask  = y_true_bin == cls
        correct   = np.where(sub_mask & ~wrong_mask)[0]
        incorrect = np.where(sub_mask &  wrong_mask)[0]
        color     = CLASS_COLORS['H'] if cls == 'H' else CLASS_COLORS['cancer']
        ypos      = y_positions[cls]

        jc = jitter_rng.uniform(-0.15, 0.15, len(correct))
        ji = jitter_rng.uniform(-0.15, 0.15, len(incorrect))

        # x = decision score (horizontal), y = class position + jitter (vertical)
        ax_strip.scatter(decisions[correct],   ypos + jc,
                         color=color, s=50, alpha=0.80,
                         edgecolors='white', linewidths=0.5, zorder=3,
                         label='Healthy (true)' if cls == 'H' else 'Cancer (true)')
        if len(incorrect):
            ax_strip.scatter(decisions[incorrect], ypos + ji,
                             color='black', marker='X', s=80, alpha=0.95,
                             zorder=5,
                             label=f'Misclassified (n={n_wrong})' if cls == 'H' else '_nolegend_')

    # Decision boundary (vertical line at 0) and grey zone (vertical band)
    ax_strip.axvline(0, color='black', lw=1.2, ls='-', zorder=2,
                     label='Decision boundary')
    ax_strip.axvspan(-GREY_ZONE, GREY_ZONE, alpha=0.12, color='#888888',
                     label=f'Grey zone ($|d|<{GREY_ZONE}$)', zorder=0)

    ax_strip.set_yticks([0, 1])
    ax_strip.set_yticklabels(['Healthy', 'Cancer'], fontsize=11)
    ax_strip.set_xlabel('SVM decision function value', fontsize=10)
    ax_strip.set_title('A: Per-patient LOOCV decision scores', fontsize=10,
                       fontweight='bold', pad=6)
    ax_strip.legend(fontsize=8.5, frameon=False, loc='upper left')
    ax_strip.spines[['top', 'right']].set_visible(False)
    ax_strip.set_ylim(-0.5, 1.5)

    # Count annotations (right edge, one per class row)
    xmax = ax_strip.get_xlim()[1]
    for cls in ['H', 'cancer']:
        sub   = y_true_bin == cls
        n_sub = sub.sum()
        n_err = (sub & wrong_mask).sum()
        ypos  = y_positions[cls]
        color = CLASS_COLORS['H'] if cls == 'H' else CLASS_COLORS['cancer']
        ax_strip.text(xmax * 0.97,  ypos + 0.22,
                      f'n={n_sub}, {n_err} error{"s" if n_err != 1 else ""}',
                      ha='right', va='center', fontsize=8.5, color=color)

    # ── Panel B: PCA scatter with misclassified highlighted ───────────────────
    for cls in (['H', 'cancer'] if len(task_classes) > 2 else task_classes):
        if cls == 'cancer':
            idx = np.where(y_true_bin == 'cancer')[0]
        else:
            idx = np.where(y_orig == cls)[0]
        ax_pca.scatter(scores[idx, 0], scores[idx, 1],
                       c=CLASS_COLORS.get(cls, '#888888'),
                       label=cls, s=50, alpha=0.75,
                       edgecolors='white', linewidths=0.4, zorder=3)

    # Overlay misclassified with open black circles
    if wrong_mask.any():
        ax_pca.scatter(scores[wrong_mask, 0], scores[wrong_mask, 1],
                       s=200, facecolors='none', edgecolors='black',
                       lw=2.0, zorder=5, label=f'Misclassified (n={n_wrong})')

    ax_pca.set_xlabel(f'Comp. 1 ({var_exp[0]*100:.1f}% var.)', fontsize=10)
    ax_pca.set_ylabel(f'Comp. 2 ({var_exp[1]*100:.1f}% var.)', fontsize=10)
    ax_pca.set_title('B: PCA score plot — misclassified patients highlighted',
                     fontsize=10, fontweight='bold', pad=6)
    ax_pca.legend(fontsize=8.5, frameon=False)
    ax_pca.spines[['top', 'right']].set_visible(False)
    ax_pca.axhline(0, color='#DDDDDD', lw=0.7, zorder=0)
    ax_pca.axvline(0, color='#DDDDDD', lw=0.7, zorder=0)

    plt.tight_layout()
    fname = f'{OUT}/svm_5_3_decision_scores_{task}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    # ── Print misclassification detail for write-up ───────────────────────────
    print(f"\n  Misclassification summary ({task}):")
    for pi in np.where(wrong_mask)[0]:
        in_grey = abs(decisions[pi]) < GREY_ZONE
        print(f"    Patient {pi:3d}: true={y_true_bin[pi]:6s}  "
              f"pred={y_pred_bin[pi]:6s}  d={decisions[pi]:+.3f}"
              f"{'  [GREY ZONE]' if in_grey else ''}")
    grey_wrong = wrong_mask & (np.abs(decisions) < GREY_ZONE)
    print(f"  Misclassified in grey zone: {grey_wrong.sum()}/{n_wrong}")
    print(f"  Misclassified outside grey zone: {(wrong_mask & ~(np.abs(decisions)<GREY_ZONE)).sum()}/{n_wrong}")


# ── Run for both tasks ────────────────────────────────────────────────────────
for task, task_classes in TASKS.items():
    cfg_rows = best_configs[best_configs['task'] == task]
    if cfg_rows.empty:
        print(f"No best config found for {task}, skipping.")
        continue
    make_figure(task, task_classes, cfg_rows.iloc[0])
