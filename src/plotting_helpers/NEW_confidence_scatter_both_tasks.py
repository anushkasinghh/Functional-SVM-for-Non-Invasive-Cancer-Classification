"""
SVM vs FSVM — Decision Score Scatter (Task I & II)
===================================================

One figure, two panels. Each dot is one patient.

  x-axis : SVM decision score   (positive → predicts cancer)
  y-axis : FSVM decision score  (positive → predicts cancer)

Quadrant labelling: corner badge strips (FSVM ✓/✗  SVM ✓/✗) replace all
floating text boxes. Each corner badge sits inside its quadrant and shows
what each model predicts in that region.

  top-right    FSVM✓  SVM✓   — both predict cancer correctly
  top-left     FSVM✓  SVM✗   — FSVM→cancer, SVM→H
  bottom-left  FSVM✓  SVM✓   — both predict H correctly
  bottom-right FSVM✗  SVM✓   — SVM→cancer, FSVM→H

Dot colour = true label. Black ring = misclassified by ≥1 model.
Distance from an axis = that model's confidence.

Output: src/figures/confidence_scatter_both_tasks.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
SVM_CSV  = os.path.join(BASE, '..', 'classical_SVM_pipeline',
                         'eval_result_data', 'svm_loocv_per_patient.csv')
FSVM_CSV = os.path.join(BASE, '..', 'FSVC',
                         'eval_result_data', 'fsvc_loocv_per_patient.csv')
OUT_DIR  = os.path.join(BASE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C_H       = '#1565C0'   # blue       — healthy
C_CANCER  = '#BF360C'   # burnt red  — cancer
C_RING    = '#1A1A1A'   # near-black — misclassification ring
C_CORRECT = '#2E7D32'   # dark green — ✓
C_WRONG   = '#C62828'   # dark red   — ✗

Q_BOTH_CANCER = '#FDE8E8'   # warm red tint   — top-right
Q_BOTH_H      = '#DDEEFF'   # cool blue tint  — bottom-left
Q_DISAGREE    = '#EEEEDD'   # warm cream      — off-diagonals

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
fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.97, wspace=0.38)


# ── Helper: draw a two-row FSVM/SVM badge at one quadrant corner ──────────────
def draw_corner_badge(ax, x_sign, y_sign, fsvm_correct, svm_correct):
    """
    Place a compact badge in axes-fraction space near the corner defined by
    (x_sign, y_sign), where +1 means the positive half and -1 the negative half.

    Badge layout (two rows):
        FSVM  ✓ or ✗
        SVM   ✓ or ✗
    """
    # Badge dimensions and inset from the axes edge (all in axes-fraction)
    bw   = 0.110   # badge width
    bh   = 0.105   # badge height
    inset = 0.022  # gap from axis edge inward toward centre

    # Horizontal: if x_sign=+1 (right half) badge is near the right edge,
    # if x_sign=-1 (left half) badge is near the left edge.
    if x_sign > 0:
        origin_x = 1.0 - bw - inset   # right-aligned
    else:
        origin_x = inset               # left-aligned

    # Vertical: if y_sign=+1 (top half) badge is near the top edge,
    # if y_sign=-1 (bottom half) badge is near the bottom edge.
    if y_sign > 0:
        origin_y = 1.0 - bh - inset   # top-aligned
    else:
        origin_y = inset               # bottom-aligned

    # Background
    bg = FancyBboxPatch(
        (origin_x, origin_y), bw, bh,
        boxstyle='round,pad=0.008',
        transform=ax.transAxes,
        facecolor='white', edgecolor='#90A4AE',
        linewidth=1.0, alpha=0.95, zorder=7,
        clip_on=False,
    )
    ax.add_patch(bg)

    # Two rows: FSVM on top, SVM on bottom
    cx      = origin_x + bw / 2          # horizontal centre of badge
    row_gap = bh * 0.22                  # half-row offset from vertical centre
    cy      = origin_y + bh / 2

    icon_fs = 10.5   # ✓/✗ size
    lbl_fs  = 8.0    # "FSVM"/"SVM" label size

    rows = [
        ('FSVM', fsvm_correct, cy + row_gap),
        ('SVM',  svm_correct,  cy - row_gap),
    ]

    for model_name, is_correct, ry in rows:
        icon  = '✓' if is_correct else '✗'
        color = C_CORRECT if is_correct else C_WRONG

        # Label on the left
        ax.text(origin_x + 0.007, ry, model_name,
                transform=ax.transAxes,
                ha='left', va='center',
                fontsize=lbl_fs, color='#37474F',
                fontweight='bold', zorder=8, clip_on=False)

        # Icon on the right
        ax.text(origin_x + bw - 0.007, ry, icon,
                transform=ax.transAxes,
                ha='right', va='center',
                fontsize=icon_fs, color=color,
                fontweight='bold', zorder=8, clip_on=False)


# ── Main panel drawing function ───────────────────────────────────────────────
def draw_panel(ax, task, meta):
    svm_t  = svm_all[svm_all['task']  == task].reset_index(drop=True)
    fsvm_t = fsvm_all[fsvm_all['task'] == task].reset_index(drop=True)

    df = svm_t[['patient_id', 'true_label', 'decision_score', 'correct']].merge(
        fsvm_t[['patient_id', 'decision_score', 'correct']],
        on='patient_id', suffixes=('_svm', '_fsvm'),
    )

    xs = df['decision_score_svm'].values
    ys = df['decision_score_fsvm'].values

    # Symmetric axis limits
    pad = 0.25
    lim = max(np.abs(xs).max(), np.abs(ys).max()) * (1 + pad)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # ── Quadrant shading ──────────────────────────────────────────────────────
    ax.fill_between([ 0,  lim], [ 0,  0],  [ lim,  lim], color=Q_BOTH_CANCER, zorder=0)
    ax.fill_between([-lim, 0], [-lim,-lim], [ 0,    0],  color=Q_BOTH_H,      zorder=0)
    ax.fill_between([-lim, 0], [ 0,   0],  [ lim,  lim], color=Q_DISAGREE,    zorder=0)
    ax.fill_between([ 0,  lim], [-lim,-lim],[ 0,    0],  color=Q_DISAGREE,    zorder=0)

    # ── Decision boundary lines ───────────────────────────────────────────────
    ax.axvline(0, color='#37474F', lw=2.0, ls='--', zorder=2, alpha=1.0)
    ax.axhline(0, color='#37474F', lw=2.0, ls='--', zorder=2, alpha=1.0)

    # ── Corner badges ─────────────────────────────────────────────────────────
    # (x_sign, y_sign, fsvm_correct, svm_correct)
    # "correct" here means: the model's prediction is consistent with the
    # canonical class of that quadrant (cancer for right/top, H for left/bottom).
    #
    #  top-right  (+,+): both predict cancer  → FSVM✓  SVM✓
    #  top-left   (-,+): FSVM→cancer, SVM→H  → FSVM✓  SVM✗
    #  bottom-left(-,-): both predict H       → FSVM✓  SVM✓
    #  bottom-right(+,-):SVM→cancer, FSVM→H  → FSVM✗  SVM✓
    quadrant_badges = [
        ( 1,  1, True,  True),
        (-1,  1, True,  False),
        (-1, -1, True,  True),
        ( 1, -1, False, True),
    ]
    for x_sign, y_sign, fsvm_c, svm_c in quadrant_badges:
        draw_corner_badge(ax, x_sign, y_sign, fsvm_c, svm_c)

    # ── Dots ──────────────────────────────────────────────────────────────────
    for label, color in [('H', C_H), (meta['cancer_label'], C_CANCER)]:
        sub = df[df['true_label'] == label]
        ax.scatter(sub['decision_score_svm'], sub['decision_score_fsvm'],
                   color=color, s=120, alpha=0.88,
                   edgecolors='#CCCCCC', linewidths=0.8, zorder=4)

    # ── Misclassification ring ────────────────────────────────────────────────
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

    # ── Stats box ─────────────────────────────────────────────────────────────
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
fig.suptitle('LOOCV Decision Scores — SVM vs FSVM',
             fontsize=17, fontweight='bold', y=0.96)

# ── Legend — also add ✓/✗ colour explanation ──────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_H,      label='Healthy (H)'),
    mpatches.Patch(color=C_CANCER, label='Cancer (PC / KC+BC+PC)'),
    plt.scatter([], [], s=120, facecolors='none', edgecolors=C_RING,
                linewidths=2.0, label='Misclassified by ≥1 model'),
    plt.Line2D([0], [0], color='#37474F', lw=1.6, ls='--',
               label='Decision boundary  ($d = 0$)'),
    plt.Line2D([0], [0], color='#90A4AE', lw=1.0, ls=':',
               label='Equal confidence  ($d_\\mathrm{SVM} = d_\\mathrm{FSVM}$)'),
    # Decode the badge colours
    mpatches.Patch(color=C_CORRECT, label='✓  correct prediction (badge)'),
    mpatches.Patch(color=C_WRONG,   label='✗  incorrect prediction (badge)'),
]
fig.legend(handles=legend_handles,
           fontsize=11.5,
           loc='lower center',
           ncol=4,
           bbox_to_anchor=(0.5, 0.00),
           frameon=True, framealpha=0.97,
           edgecolor='#CFD8DC',
           handlelength=2.2,
           handletextpad=0.8,
           columnspacing=2.0,
           borderpad=0.9)

out_path = os.path.join(OUT_DIR, 'confidence_scatter_both_tasks.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_path}')
