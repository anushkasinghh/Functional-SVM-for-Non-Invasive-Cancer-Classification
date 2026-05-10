# """
# SVM vs FSVM — Decision Score Scatter (Task I & II)
# ===================================================

# One figure, two panels. Each dot is one patient.

#   x-axis : SVM decision score   (positive → predicts cancer)
#   y-axis : FSVM decision score  (positive → predicts cancer)

# The dashed lines at x=0 and y=0 are the decision boundaries.
# The four quadrants show what each model predicted:

#   top-right    both predict cancer
#   bottom-left  both predict H
#   top-left     SVM→H,      FSVM→cancer
#   bottom-right SVM→cancer, FSVM→H

# Dot colour = true label. Black ring = misclassified by ≥1 model.
# Distance from an axis = that model's confidence.

# Run after generating svm_loocv_per_patient.csv from SVM_notebook Step 3b.
# Output: src/figures/confidence_scatter_both_tasks.png
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # ── Paths ─────────────────────────────────────────────────────────────────────
# BASE     = os.path.dirname(os.path.abspath(__file__))
# SVM_CSV  = os.path.join(BASE, '..', 'classical_SVM_pipeline',
#                          'eval_result_data', 'svm_loocv_per_patient.csv')
# FSVM_CSV = os.path.join(BASE, '..', 'FSVC',
#                          'eval_result_data', 'fsvc_loocv_per_patient.csv')
# OUT_DIR  = os.path.join(BASE, 'figures')
# os.makedirs(OUT_DIR, exist_ok=True)

# # ── Colours ───────────────────────────────────────────────────────────────────
# C_H      = '#1565C0'   # blue        — healthy
# C_CANCER = '#BF360C'   # burnt red   — cancer
# C_RING   = '#1A1A1A'   # near-black  — misclassification ring

# Q_BOTH_CANCER = '#FDE8E8'   # warm red tint    — top-right
# Q_BOTH_H      = '#DDEEFF'   # cool blue tint   — bottom-left
# Q_DISAGREE    = '#F4F4EE'   # neutral cream    — off-diagonals

# # ── Task config ───────────────────────────────────────────────────────────────
# TASKS = {
#     'H_vs_PC': {
#         'panel': 'A',
#         'title': 'Task I — H vs PC',
#         'cancer_label': 'PC',
#     },
#     'H_vs_KC_BC_PC': {
#         'panel': 'B',
#         'title': 'Task II — H vs KC+BC+PC',
#         'cancer_label': 'cancer',
#     },
# }

# # ── Load data ─────────────────────────────────────────────────────────────────
# svm_all  = pd.read_csv(SVM_CSV)
# fsvm_all = pd.read_csv(FSVM_CSV)

# # ── Figure layout ─────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
# fig.subplots_adjust(top=0.88, bottom=0.18, left=0.08, right=0.97, wspace=0.30)


# def draw_panel(ax, task, meta):
#     svm_t  = svm_all[svm_all['task']  == task].reset_index(drop=True)
#     fsvm_t = fsvm_all[fsvm_all['task'] == task].reset_index(drop=True)

#     df = svm_t[['patient_id', 'true_label', 'decision_score', 'correct']].merge(
#         fsvm_t[['patient_id', 'decision_score', 'correct']],
#         on='patient_id', suffixes=('_svm', '_fsvm'),
#     )

#     xs = df['decision_score_svm'].values
#     ys = df['decision_score_fsvm'].values

#     # Symmetric axis limits with padding
#     pad  = 0.25
#     lim  = max(np.abs(xs).max(), np.abs(ys).max()) * (1 + pad)
#     ax.set_xlim(-lim, lim)
#     ax.set_ylim(-lim, lim)

#     # ── Quadrant shading ──────────────────────────────────────────────────────
#     ax.fill_between([ 0,  lim], [ 0,  0],  [ lim,  lim], color=Q_BOTH_CANCER, zorder=0)
#     ax.fill_between([-lim, 0], [-lim,-lim], [ 0,    0],  color=Q_BOTH_H,      zorder=0)
#     ax.fill_between([-lim, 0], [ 0,   0],  [ lim,  lim], color=Q_DISAGREE,    zorder=0)
#     ax.fill_between([ 0,  lim], [-lim,-lim],[ 0,    0],  color=Q_DISAGREE,    zorder=0)

#     # ── Decision boundary lines ───────────────────────────────────────────────
#     ax.axvline(0, color='#37474F', lw=1.6, ls='--', zorder=2, alpha=0.75)
#     ax.axhline(0, color='#37474F', lw=1.6, ls='--', zorder=2, alpha=0.75)

#     # ── Quadrant labels ───────────────────────────────────────────────────────
#     qc   = lim * 0.62   # push labels toward quadrant corners, away from dots
#     qbox = dict(boxstyle='round,pad=0.45', facecolor='white',
#                 edgecolor='#B0BEC5', linewidth=1.0, alpha=0.93)
#     qlabels = [
#         ( qc,  qc, 'Both →\nCancer'),
#         (-qc,  qc, 'SVM → H\nFSVM → Cancer'),
#         (-qc, -qc, 'Both →\nH'),
#         ( qc, -qc, 'SVM → Cancer\nFSVM → H'),
#     ]
#     for qx, qy, qtxt in qlabels:
#         ax.text(qx, qy, qtxt,
#                 ha='center', va='center',
#                 fontsize=11.5, fontweight='bold', color='#37474F',
#                 linespacing=1.6, bbox=qbox, zorder=3)

#     # ── Dots — true label colour ──────────────────────────────────────────────
#     for label, color in [('H', C_H), (meta['cancer_label'], C_CANCER)]:
#         sub = df[df['true_label'] == label]
#         ax.scatter(sub['decision_score_svm'], sub['decision_score_fsvm'],
#                    color=color, s=120, alpha=0.88,
#                    edgecolors='white', linewidths=0.8, zorder=4)

#     # ── Ring on misclassified patients (≥1 model wrong) ───────────────────────
#     wrong = df[~df['correct_svm'] | ~df['correct_fsvm']]
#     ax.scatter(wrong['decision_score_svm'], wrong['decision_score_fsvm'],
#                s=260, facecolors='none',
#                edgecolors=C_RING, linewidths=2.2, zorder=5)

#     # ── Equal-confidence diagonal ─────────────────────────────────────────────
#     ax.plot([-lim, lim], [-lim, lim],
#             color='#90A4AE', lw=1.0, ls=':', zorder=1)

#     # ── Axis formatting ───────────────────────────────────────────────────────
#     ax.set_xlabel('SVM decision score', fontsize=14, labelpad=11)
#     ax.set_ylabel('FSVM decision score', fontsize=14, labelpad=11)
#     ax.tick_params(axis='both', labelsize=12)
#     ax.spines[['top', 'right']].set_visible(False)
#     ax.set_aspect('equal', adjustable='box')

#     # ── Panel title ───────────────────────────────────────────────────────────
#     n = len(df)
#     ax.set_title(f'{meta["panel"]}: {meta["title"]}  ($n={n}$)',
#                  fontsize=15, fontweight='bold', pad=14)

#     # ── Stats box — top-left corner (never overlaps dots in disagreement zone) ─
#     wrong_both = (~df['correct_svm'] & ~df['correct_fsvm']).sum()
#     stats = (f"SVM errors:   {(~df['correct_svm']).sum()}\n"
#              f"FSVM errors:  {(~df['correct_fsvm']).sum()}\n"
#              f"Both wrong:   {wrong_both}")
#     ax.text(0.03, 0.97, stats,
#             transform=ax.transAxes,
#             ha='left', va='top',
#             fontsize=11, color='#37474F', linespacing=1.9,
#             family='monospace',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
#                       edgecolor='#CFD8DC', linewidth=1.1),
#             zorder=6)

#     return df


# # ── Draw panels ───────────────────────────────────────────────────────────────
# for ax, (task, meta) in zip(axes, TASKS.items()):
#     df = draw_panel(ax, task, meta)
#     print(f"\n{meta['title']}  (n={len(df)})")
#     print(f"  SVM errors : {(~df['correct_svm']).sum()}")
#     print(f"  FSVM errors: {(~df['correct_fsvm']).sum()}")
#     print(f"  Both wrong : {(~df['correct_svm'] & ~df['correct_fsvm']).sum()}")
#     print(f"  SVM only   : {( df['correct_fsvm'] & ~df['correct_svm']).sum()}")
#     print(f"  FSVM only  : {(~df['correct_fsvm'] &  df['correct_svm']).sum()}")

# # ── Figure title ──────────────────────────────────────────────────────────────
# # Sits well above the panels thanks to top=0.84 in subplots_adjust.
# fig.suptitle('LOOCV Decision Scores — SVM vs FSVM',
#              fontsize=17, fontweight='bold', y=0.975)

# # ── Shared legend — two rows at the bottom ────────────────────────────────────
# # ncol=3 gives two rows for five items (3 + 2), avoiding the single long row
# # that caused crowding previously.
# legend_handles = [
#     # ── Row 1: dot colours — class-specific names ──────────────────────────
#     mpatches.Patch(color=C_H,      label='Healthy (H)'),
#     mpatches.Patch(color=C_CANCER, label='PC  (Task I)'),
#     mpatches.Patch(color=C_CANCER, label='KC+BC+PC  (Task II)',
#                    hatch='////', edgecolor='white', linewidth=0.5),
#     # ── Row 2: markers & lines ────────────────────────────────────────────
#     plt.scatter([], [], s=120, facecolors='none', edgecolors=C_RING,
#                 linewidths=2.0, label='Misclassified by ≥1 model'),
#     plt.Line2D([0], [0], color='#37474F', lw=1.6, ls='--',
#                label='Decision boundary  ($d = 0$)'),
#     plt.Line2D([0], [0], color='#90A4AE', lw=1.0, ls=':',
#                label='Equal confidence  ($d_\\mathrm{SVM} = d_\\mathrm{FSVM}$)'),
# ]
# fig.legend(handles=legend_handles,
#            fontsize=12,
#            loc='lower center',
#            ncol=3,
#            bbox_to_anchor=(0.5, 0.00),
#            frameon=True, framealpha=0.97,
#            edgecolor='#CFD8DC',
#            handlelength=2.4,
#            handletextpad=0.8,
#            columnspacing=2.2,
#            borderpad=0.9)

# out_path = os.path.join(OUT_DIR, 'confidence_scatter_both_tasks.png')
# plt.savefig(out_path, dpi=180, bbox_inches='tight')
# plt.close()
# print(f'\nSaved: {out_path}')










"""
SVM vs FSVM — Decision Score Scatter (Task I & II)
===================================================

One figure, two panels. Each dot is one patient.

  x-axis : SVM decision score   (positive → predicts cancer)
  y-axis : FSVM decision score  (positive → predicts cancer)

The dashed lines at x=0 and y=0 are the decision boundaries.
The four quadrants show what each model predicted:

  top-right    both predict cancer
  bottom-left  both predict H
  top-left     SVM→H,      FSVM→cancer
  bottom-right SVM→cancer, FSVM→H

Dot colour = true label. Black ring = misclassified by ≥1 model.
Distance from an axis = that model's confidence.

Run after generating svm_loocv_per_patient.csv from SVM_notebook Step 3b.
Output: src/figures/confidence_scatter_both_tasks.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
SVM_CSV  = os.path.join(BASE, '..', 'classical_SVM_pipeline',
                         'eval_result_data', 'svm_loocv_per_patient.csv')
FSVM_CSV = os.path.join(BASE, '..', 'FSVC',
                         'eval_result_data', 'fsvc_loocv_per_patient.csv')
OUT_DIR  = os.path.join(BASE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C_H      = '#1565C0'   # blue        — healthy
C_CANCER = '#BF360C'   # burnt red   — cancer
C_RING   = '#1A1A1A'   # near-black  — misclassification ring

Q_BOTH_CANCER = '#FDE8E8'   # warm red tint    — top-right
Q_BOTH_H      = '#DDEEFF'   # cool blue tint   — bottom-left
Q_DISAGREE    = '#EEEEDD'   # FIX: warmer/more saturated — off-diagonals (was #F4F4EE, nearly white)

# ── Task config ───────────────────────────────────────────────────────────────
TASKS = {
    'H_vs_PC': {
        'panel': 'A',
        'title': 'Task I — H vs PC',
        'cancer_label': 'PC',
    },
    'H_vs_KC_BC_PC': {
        'panel': 'B',
        'title': 'Task II — H vs KC+BC+PC',
        'cancer_label': 'cancer',
    },
}

# ── Load data ─────────────────────────────────────────────────────────────────
svm_all  = pd.read_csv(SVM_CSV)
fsvm_all = pd.read_csv(FSVM_CSV)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
# FIX: increased top from 0.88→0.90 and wspace from 0.30→0.38
fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.97, wspace=0.38)


def draw_panel(ax, task, meta):
    svm_t  = svm_all[svm_all['task']  == task].reset_index(drop=True)
    fsvm_t = fsvm_all[fsvm_all['task'] == task].reset_index(drop=True)

    df = svm_t[['patient_id', 'true_label', 'decision_score', 'correct']].merge(
        fsvm_t[['patient_id', 'decision_score', 'correct']],
        on='patient_id', suffixes=('_svm', '_fsvm'),
    )

    xs = df['decision_score_svm'].values
    ys = df['decision_score_fsvm'].values

    # Symmetric axis limits with padding
    pad  = 0.25
    lim  = max(np.abs(xs).max(), np.abs(ys).max()) * (1 + pad)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # ── Quadrant shading ──────────────────────────────────────────────────────
    ax.fill_between([ 0,  lim], [ 0,  0],  [ lim,  lim], color=Q_BOTH_CANCER, zorder=0)
    ax.fill_between([-lim, 0], [-lim,-lim], [ 0,    0],  color=Q_BOTH_H,      zorder=0)
    ax.fill_between([-lim, 0], [ 0,   0],  [ lim,  lim], color=Q_DISAGREE,    zorder=0)
    ax.fill_between([ 0,  lim], [-lim,-lim],[ 0,    0],  color=Q_DISAGREE,    zorder=0)

    # ── Decision boundary lines ───────────────────────────────────────────────
    # FIX: lw 1.6→2.0, alpha 0.75→1.0 — fully opaque, thicker, easier to read near dense clusters
    ax.axvline(0, color='#37474F', lw=2.0, ls='--', zorder=2, alpha=1.0)
    ax.axhline(0, color='#37474F', lw=2.0, ls='--', zorder=2, alpha=1.0)

    # ── Quadrant labels ───────────────────────────────────────────────────────
    # FIX: push labels to extreme corners (0.80/0.78 × lim) to avoid overlap with data
    # Previously at 0.62 × lim which landed in the middle of dense clusters (especially Panel B)
    qc_x = lim * 0.80
    qc_y = lim * 0.78

    qbox = dict(boxstyle='round,pad=0.45', facecolor='white',
                edgecolor='#B0BEC5', linewidth=1.0, alpha=0.93)
    qlabels = [
        ( qc_x,  qc_y, 'Both →\nCancer'),
        (-qc_x,  qc_y, 'SVM → H\nFSVM → Cancer'),
        (-qc_x, -qc_y, 'Both →\nH'),
        ( qc_x, -qc_y, 'SVM → Cancer\nFSVM → H'),
    ]
    for qx, qy, qtxt in qlabels:
        ax.text(qx, qy, qtxt,
                ha='center', va='center',
                fontsize=12, fontweight='bold', color='#37474F',   # FIX: 11.5→12 (clean integer)
                linespacing=1.6, bbox=qbox, zorder=3)

    # ── Dots — true label colour ──────────────────────────────────────────────
    # FIX: edgecolors 'white'→'#CCCCCC' so dots are visually separated when overlapping
    for label, color in [('H', C_H), (meta['cancer_label'], C_CANCER)]:
        sub = df[df['true_label'] == label]
        ax.scatter(sub['decision_score_svm'], sub['decision_score_fsvm'],
                   color=color, s=120, alpha=0.88,
                   edgecolors='#CCCCCC', linewidths=0.8, zorder=4)

    # ── Ring on misclassified patients (≥1 model wrong) ───────────────────────
    wrong = df[~df['correct_svm'] | ~df['correct_fsvm']]
    ax.scatter(wrong['decision_score_svm'], wrong['decision_score_fsvm'],
               s=260, facecolors='none',
               edgecolors=C_RING, linewidths=2.2, zorder=5)

    # ── Equal-confidence diagonal ─────────────────────────────────────────────
    ax.plot([-lim, lim], [-lim, lim],
            color='#90A4AE', lw=1.0, ls=':', zorder=1)

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax.set_xlabel('SVM decision score', fontsize=14, labelpad=11)
    ax.set_ylabel('FSVM decision score', fontsize=14, labelpad=11)
    ax.tick_params(axis='both', labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_aspect('equal', adjustable='box')

    # ── Panel title ───────────────────────────────────────────────────────────
    n = len(df)
    ax.set_title(f'{meta["panel"]}: {meta["title"]}  ($n={n}$)',
                 fontsize=15, fontweight='bold', pad=14)

    # ── Stats box — FIX: added Agreement count for interpretive value ──────────
    n_agree    = (df['correct_svm'] == df['correct_fsvm']).sum()
    wrong_both = (~df['correct_svm'] & ~df['correct_fsvm']).sum()
    stats = (f"SVM errors:   {(~df['correct_svm']).sum()}\n"
             f"FSVM errors:  {(~df['correct_fsvm']).sum()}\n"
             f"Both wrong:   {wrong_both}\n"
             f"Agreement:    {n_agree}/{n}")
    ax.text(0.03, 0.97, stats,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=11, color='#37474F', linespacing=1.9,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#CFD8DC', linewidth=1.1),
            zorder=6)

    return df


# ── Draw panels ───────────────────────────────────────────────────────────────
for ax, (task, meta) in zip(axes, TASKS.items()):
    df = draw_panel(ax, task, meta)
    n_agree    = (df['correct_svm'] == df['correct_fsvm']).sum()
    wrong_both = (~df['correct_svm'] & ~df['correct_fsvm']).sum()
    print(f"\n{meta['title']}  (n={len(df)})")
    print(f"  SVM errors : {(~df['correct_svm']).sum()}")
    print(f"  FSVM errors: {(~df['correct_fsvm']).sum()}")
    print(f"  Both wrong : {wrong_both}")
    print(f"  SVM only   : {( df['correct_fsvm'] & ~df['correct_svm']).sum()}")
    print(f"  FSVM only  : {(~df['correct_fsvm'] &  df['correct_svm']).sum()}")
    print(f"  Agreement  : {n_agree}/{len(df)}")

# ── Figure title ──────────────────────────────────────────────────────────────
# FIX: y=0.975→0.96 to prevent clipping at the very top edge
fig.suptitle('LOOCV Decision Scores — SVM vs FSVM',
             fontsize=17, fontweight='bold', y=0.96)

# ── Shared legend — FIX: single cancer patch (no misleading hatch), ncol=5 ────
# Previously two red patches (plain + hatched) for PC vs KC+BC+PC, but the
# dots in the plot have no hatching — the hatch was purely decorative and
# confusing. Panel titles already distinguish tasks, so one entry is enough.
legend_handles = [
    mpatches.Patch(color=C_H,      label='Healthy (H)'),
    mpatches.Patch(color=C_CANCER, label='Cancer (PC / KC+BC+PC)'),
    plt.scatter([], [], s=120, facecolors='none', edgecolors=C_RING,
                linewidths=2.0, label='Misclassified by ≥1 model'),
    plt.Line2D([0], [0], color='#37474F', lw=1.6, ls='--',
               label='Decision boundary  ($d = 0$)'),
    plt.Line2D([0], [0], color='#90A4AE', lw=1.0, ls=':',
               label='Equal confidence  ($d_\\mathrm{SVM} = d_\\mathrm{FSVM}$)'),
]
fig.legend(handles=legend_handles,
           fontsize=12,
           loc='lower center',
           ncol=5,           # FIX: 3→5, all items fit in one clean row now
           bbox_to_anchor=(0.5, 0.00),
           frameon=True, framealpha=0.97,
           edgecolor='#CFD8DC',
           handlelength=2.4,
           handletextpad=0.8,
           columnspacing=2.2,
           borderpad=0.9)

out_path = os.path.join(OUT_DIR, 'confidence_scatter_both_tasks.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_path}')