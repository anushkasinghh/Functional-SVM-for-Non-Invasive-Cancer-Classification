"""
SVM vs FSVM — Model Agreement Matrix
=====================================

Produces one agreement matrix figure per task:
  svm_vs_fsvc_agreement_H_vs_PC.png
  svm_vs_fsvc_agreement_H_vs_KC_BC_PC.png

Each figure is a 2×2 matrix: rows = SVM outcome, columns = FSVM outcome.
Cells are semantically coloured (green / amber / red) with a large bold
count and label. A right-margin bracket calls out the disagreement cells.

Run after generating svm_loocv_per_patient.csv from SVM_notebook Step 3b.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

SVM_CSV  = os.path.join(BASE, '..', 'classical_SVM_pipeline',
                         'eval_result_data', 'svm_loocv_per_patient.csv')
FSVM_CSV = os.path.join(BASE, '..', 'FSVC',
                         'eval_result_data', 'fsvc_loocv_per_patient.csv')
OUT_DIR  = os.path.join(BASE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Task metadata ─────────────────────────────────────────────────────────────
# title, subtitle config line, output filename
TASKS = {
    'H_vs_PC': {
        'title':    'Task I: H vs PC',
        'subtitle': 'SVM: SR 1005, PCA-4, RBF, C=0.7525   ·   FSVM: SR 1005, τ*=5, K*=1, C=0.7525',
        'outfile':  'svm_vs_fsvm_agreement_H_vs_PC.png',
    },
    'H_vs_KC_BC_PC': {
        'title':    'Task II: H vs KC+BC+PC',
        'subtitle': 'SVM: SR 1170, PCA-4, RBF, C=0.505   ·   FSVM: SR 1005, τ*=10, K*=2, C=1.0',
        'outfile':  'svm_vs_fsvm_agreement_H_vs_KC_BC_PC.png',
    },
}

# ── Cell visual definitions ───────────────────────────────────────────────────
# (svm_correct, fsvc_correct) → (grid_row, grid_col, bg, fg, label, icon)
CELLS = {
    (True,  True):  (0, 0, '#E8F5E9', '#1B5E20', 'Both correct',   '✓✓'),
    (True,  False): (0, 1, '#FFF8E1', '#E65100', 'SVM ✓  FSVM ✗', '✓✗'),
    (False, True):  (1, 0, '#FFF8E1', '#E65100', 'SVM ✗  FSVM ✓', '✗✓'),
    (False, False): (1, 1, '#FFEBEE', '#B71C1C', 'Both wrong',     '✗✗'),
}

CELL_SIZE = 0.88
CELL_PAD  = 0.06
CORNER_R  = 0.06

def cell_origin(row, col):
    x = col * (CELL_SIZE + CELL_PAD)
    y = (1 - row) * (CELL_SIZE + CELL_PAD)
    return x, y


# ── Drawing function ──────────────────────────────────────────────────────────
def draw_matrix(ax, cell_data, n_disagree, n_total):
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.35, 2.35)
    ax.axis('off')
    ax.set_aspect('equal')

    # Cells
    for key, d in cell_data.items():
        x0, y0 = cell_origin(d['row'], d['col'])
        cx = x0 + CELL_SIZE / 2

        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), CELL_SIZE, CELL_SIZE,
            boxstyle=f'round,pad=0,rounding_size={CORNER_R}',
            facecolor=d['bg'], edgecolor='white', linewidth=3,
            transform=ax.transData, zorder=2,
        ))

        # Icon — top-right corner, ghost opacity
        ax.text(x0 + CELL_SIZE - 0.04, y0 + CELL_SIZE - 0.04,
                d['icon'], ha='right', va='top', fontsize=10,
                color=d['fg'], alpha=0.45, transform=ax.transData, zorder=3)

        # Count — hero element
        ax.text(cx, y0 + CELL_SIZE * 0.65, str(d['count']),
                ha='center', va='center', fontsize=34, fontweight='bold',
                color=d['fg'], transform=ax.transData, zorder=3)

        # Label
        ax.text(cx, y0 + CELL_SIZE * 0.32, d['label'].upper(),
                ha='center', va='center', fontsize=11, fontweight='bold',
                alpha=0.8, color=d['fg'], family='monospace',
                transform=ax.transData, zorder=3)

    # Column headers
    HEADER_Y = 2.05
    for col, (symbol, label, color) in enumerate([
        ('✓', 'FSVM correct', '#2E7D32'),
        ('✗', 'FSVM wrong',   '#B71C1C'),
    ]):
        cx = col * (CELL_SIZE + CELL_PAD) + CELL_SIZE / 2
        ax.text(cx, HEADER_Y + 0.12, symbol, ha='center', va='bottom',
                fontsize=16, color=color, transform=ax.transData)
        ax.text(cx, HEADER_Y, label, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color, transform=ax.transData)

    # Row headers
    HEADER_X = -0.12
    for row, (symbol, label, color) in enumerate([
        ('✓', 'SVM correct', '#2E7D32'),
        ('✗', 'SVM wrong',   '#B71C1C'),
    ]):
        _, y0 = cell_origin(row, 0)
        cy = y0 + CELL_SIZE / 2
        ax.text(HEADER_X, cy + 0.06, symbol, ha='right', va='center',
                fontsize=16, color=color, transform=ax.transData)
        ax.text(HEADER_X, cy - 0.10, label, ha='right', va='center',
                fontsize=9, fontweight='bold', color=color, transform=ax.transData)

    # Disagreement bracket on the right margin
    bx = 2.12
    _, y_top_cell = cell_origin(0, 1)
    _, y_bot_cell = cell_origin(1, 1)
    y_top = y_top_cell + CELL_SIZE * 0.5 + 0.15
    y_bot = y_bot_cell + CELL_SIZE * 0.5 - 0.15

    ax.annotate('', xy=(bx, y_bot), xytext=(bx, y_top),
                arrowprops=dict(arrowstyle='-', color='#999999', lw=1.5))
    ax.plot([bx - 0.03, bx], [y_top, y_top], color='#999999', lw=1.5)
    ax.plot([bx - 0.03, bx], [y_bot, y_bot], color='#999999', lw=1.5)
    ax.text(bx + 0.04, (y_top + y_bot) / 2,
            f'{n_disagree} patients\nmodels disagree',
            ha='left', va='center', fontsize=8, color='#555555',
            linespacing=1.5, transform=ax.transData)


# ── Load CSVs once ────────────────────────────────────────────────────────────
svm_all  = pd.read_csv(SVM_CSV)
fsvc_all = pd.read_csv(FSVM_CSV)

# ── Loop over tasks ───────────────────────────────────────────────────────────
for task, meta in TASKS.items():
    svm_t  = svm_all[svm_all['task']  == task].reset_index(drop=True)
    fsvc_t = fsvc_all[fsvc_all['task'] == task].reset_index(drop=True)

    merged = svm_t[['patient_id', 'correct']].merge(
        fsvc_t[['patient_id', 'correct']],
        on='patient_id', suffixes=('_svm', '_fsvc'),
    )

    cell_data = {}
    for key, (row, col, bg, fg, label, icon) in CELLS.items():
        svm_ok, fsvc_ok = key
        sub = merged[
            (merged['correct_svm']  == svm_ok) &
            (merged['correct_fsvc'] == fsvc_ok)
        ]
        cell_data[key] = {
            'row': row, 'col': col,
            'bg': bg, 'fg': fg,
            'label': label, 'icon': icon,
            'count': len(sub),
        }

    n_disagree = cell_data[(True, False)]['count'] + cell_data[(False, True)]['count']
    n_total    = len(merged)

    fig, ax = plt.subplots(figsize=(7, 6.2))
    draw_matrix(ax, cell_data, n_disagree, n_total)

    fig.text(0.5, 0.97, f'Model Agreement — {meta["title"]}  (LOOCV)',
             ha='center', va='top', fontsize=12, fontweight='bold', color='#212121')
    fig.text(0.5, 0.93, f'n = {n_total} patients   ·   {meta["subtitle"]}',
             ha='center', va='top', fontsize=8.5, color='#666666')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(OUT_DIR, meta['outfile'])
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\n{meta['title']} (n={n_total}):")
    for key, d in cell_data.items():
        print(f"  {d['label']:20s}: {d['count']:2d}")
