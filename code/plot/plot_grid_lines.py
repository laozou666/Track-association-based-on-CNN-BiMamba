"""
Line chart of validation accuracy for the 2-D grid search:
    X-axis: dropout rate
    Lines : different n_cnn_layers (1, 2, 3, 4)
    n_mamba_layers = 1 (fixed)

Pulls results from both the ablation folder and the grid_search folder.
"""

import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

ABL_DIR  = '/home/yangcq/track_association/output/ablation'
GRID_DIR = '/home/yangcq/track_association/output/grid_search'
FIG_DIR  = '/home/yangcq/track_association/output/figures'

# (n_cnn, dropout) -> result.json file (picks whichever exists)
TABLE = {
    # n_cnn = 1   (from ablation study)
    (1, 0.1): f'{ABL_DIR}/drop01_result.json',
    (1, 0.2): f'{ABL_DIR}/cnn1_result.json',
    (1, 0.3): f'{ABL_DIR}/drop03_result.json',
    (1, 0.4): f'{ABL_DIR}/drop04_result.json',
    # n_cnn = 2
    (2, 0.1): f'{GRID_DIR}/g_cnn2_drop01_result.json',
    (2, 0.2): f'{ABL_DIR}/cnn2_result.json',
    (2, 0.3): f'{GRID_DIR}/g_cnn2_drop03_result.json',
    (2, 0.4): f'{GRID_DIR}/g_cnn2_drop04_result.json',
    # n_cnn = 3
    (3, 0.1): f'{GRID_DIR}/g_cnn3_drop01_result.json',
    (3, 0.2): f'{ABL_DIR}/cnn3_result.json',
    (3, 0.3): f'{GRID_DIR}/g_cnn3_drop03_result.json',
    (3, 0.4): f'{GRID_DIR}/g_cnn3_drop04_result.json',
    # n_cnn = 4
    (4, 0.1): f'{GRID_DIR}/g_cnn4_drop01_result.json',
    (4, 0.2): f'{GRID_DIR}/g_cnn4_drop02_result.json',
    (4, 0.3): f'{GRID_DIR}/g_cnn4_drop03_result.json',
    (4, 0.4): f'{GRID_DIR}/g_cnn4_drop04_result.json',
}

N_CNN_LIST   = [1, 2, 3, 4]
DROPOUT_LIST = [0.1, 0.2, 0.3, 0.4]

# Collect accuracies
accs = {}
for (n_cnn, drop), path in TABLE.items():
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        accs[(n_cnn, drop)] = r['best_val_acc'] * 100
    else:
        print(f'Missing: {path}')

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 6))

colors  = {1: '#1f77b4', 2: '#d62728', 3: '#2ca02c', 4: '#9467bd'}
markers = {1: 'o',       2: 's',       3: '^',       4: 'D'}
labels  = {1: '1 CNN Layer', 2: '2 CNN Layers', 3: '3 CNN Layers', 4: '4 CNN Layers'}

for n_cnn in N_CNN_LIST:
    y = [accs.get((n_cnn, d), np.nan) for d in DROPOUT_LIST]
    ax.plot(DROPOUT_LIST, y,
            color=colors[n_cnn], marker=markers[n_cnn], markersize=10,
            linewidth=2.2, label=labels[n_cnn])
    # annotate each point
    for d, v in zip(DROPOUT_LIST, y):
        if not np.isnan(v):
            ax.annotate(f'{v:.2f}', xy=(d, v), xytext=(0, 8),
                        textcoords='offset points',
                        ha='center', fontsize=8.5, color=colors[n_cnn])

# Styling to match the reference chart
ax.set_xlabel('Dropout Rate',           fontsize=13)
ax.set_ylabel('Validation Accuracy (%)', fontsize=13)
ax.set_title('Grid Search: Val Accuracy vs Dropout for Different CNN Depths\n'
             '(n_mamba = 1)', fontsize=13, fontweight='bold')

ax.set_xticks(DROPOUT_LIST)
ax.set_xticklabels([f'{d:.1f}' for d in DROPOUT_LIST])
ax.set_ylim(86, 94)
ax.grid(True, alpha=0.35, linestyle='--')
ax.legend(loc='lower left', fontsize=11, framealpha=0.95)

# Highlight the global best cell
best_key = max(accs, key=accs.get)
bn, bd   = best_key
ax.scatter(bd, accs[best_key], s=350,
           facecolors='none', edgecolors='gold', linewidths=3, zorder=5)
ax.annotate(f'Best: n_cnn={bn}, drop={bd}\n{accs[best_key]:.2f}%',
            xy=(bd, accs[best_key]),
            xytext=(20, -35), textcoords='offset points',
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='goldenrod', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.4', fc='#fff8d8',
                      ec='goldenrod', lw=1))

plt.tight_layout()
out = f'{FIG_DIR}/grid_search_lines.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ---- Print summary table ----
print(f'\n{"n_cnn":<7}', end='')
for d in DROPOUT_LIST:
    print(f'{d:>9.1f}', end='')
print()
print('-' * (7 + 9 * len(DROPOUT_LIST)))
for n_cnn in N_CNN_LIST:
    print(f'{n_cnn:<7}', end='')
    for d in DROPOUT_LIST:
        v = accs.get((n_cnn, d), np.nan)
        print(f'{v:>8.2f}%' if not np.isnan(v) else f'{"--":>9}', end='')
    print()
print(f'\nGlobal best: n_cnn={bn}, dropout={bd} -> {accs[best_key]:.2f}%')
