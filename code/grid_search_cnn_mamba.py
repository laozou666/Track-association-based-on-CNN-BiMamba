"""
2-D grid search over (n_cnn_layers, dropout) with n_mamba=1 fixed.

Grid:
  n_cnn   in {2, 3, 4}
  dropout in {0.1, 0.2, 0.3, 0.4}
  n_mamba = 1

Total 12 cells; reuse results from ablation for (2,0.2) and (3,0.2).

Usage:
  python grid_search_cnn_mamba.py --mode launch          # run all in parallel on 4 GPUs
  python grid_search_cnn_mamba.py --mode plot            # plot heatmap after done
  python grid_search_cnn_mamba.py --mode train --config g_cnn3_drop03 --gpu 0   # single cell
"""

import os, sys, json, argparse, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# Reuse training code from ablation script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_cnn_mamba import train_config, AblationCNNMamba   # noqa

GRID_RESULTS_DIR  = '/home/yangcq/track_association/output/grid_search'
ABL_RESULTS_DIR   = '/home/yangcq/track_association/output/ablation'
FIG_DIR           = '/home/yangcq/track_association/output/figures'
os.makedirs(GRID_RESULTS_DIR, exist_ok=True)

N_CNN_LIST    = [2, 3, 4]
DROPOUT_LIST  = [0.1, 0.2, 0.3, 0.4]
N_MAMBA_FIXED = 1

# Build configs & identify reusable ablation results
def cell_name(n_cnn, drop):
    return f'g_cnn{n_cnn}_drop{int(drop*10):02d}'

REUSE_MAP = {   # (n_cnn, dropout) -> already-existing ablation result file
    (2, 0.2): 'cnn2',
    (3, 0.2): 'cnn3',
}

CONFIGS = []
for n_cnn in N_CNN_LIST:
    for drop in DROPOUT_LIST:
        CONFIGS.append({
            'name':    cell_name(n_cnn, drop),
            'n_cnn':   n_cnn,
            'n_mamba': N_MAMBA_FIXED,
            'dropout': drop,
        })

CONFIGS_TO_RUN = [c for c in CONFIGS if (c['n_cnn'], c['dropout']) not in REUSE_MAP]

# GPU assignment: 10 configs -> 4 GPUs  (2-3 per GPU)
GPU_ASSIGN = {0: [], 1: [], 2: [], 3: []}
for i, c in enumerate(CONFIGS_TO_RUN):
    GPU_ASSIGN[i % 4].append(c['name'])


# -------------------------------------------------------------
# Launcher
# -------------------------------------------------------------
def launch_all():
    script = os.path.abspath(__file__)
    procs = []
    for gpu_id, cfg_names in GPU_ASSIGN.items():
        if not cfg_names:
            continue
        cmds = []
        for cn in cfg_names:
            cmds.append(
                f"CUDA_VISIBLE_DEVICES={gpu_id} "
                f"conda run -n track_association python3 {script} "
                f"--mode train --config {cn} --gpu {gpu_id}"
            )
        shell_cmd = ' && '.join(cmds)
        log_file  = f'/tmp/grid_gpu{gpu_id}.log'
        p = subprocess.Popen(
            ['bash', '-c', f'({shell_cmd}) > {log_file} 2>&1'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        procs.append((gpu_id, cfg_names, p, log_file))
        print(f'GPU {gpu_id}: {cfg_names}  -> {log_file}')

    print(f'\nTotal new configs: {len(CONFIGS_TO_RUN)} (reusing {len(CONFIGS)-len(CONFIGS_TO_RUN)})')
    print('Waiting for all to finish...')
    for gpu_id, cfg_names, p, _ in procs:
        p.wait()
        print(f'GPU {gpu_id} done: {cfg_names}')
    print('\nAll grid cells finished.')
    plot_grid()


# -------------------------------------------------------------
# Plotting
# -------------------------------------------------------------
def plot_grid():
    grid = np.full((len(N_CNN_LIST), len(DROPOUT_LIST)), np.nan)

    for i, n_cnn in enumerate(N_CNN_LIST):
        for j, drop in enumerate(DROPOUT_LIST):
            # Check reuse first
            if (n_cnn, drop) in REUSE_MAP:
                p = f'{ABL_RESULTS_DIR}/{REUSE_MAP[(n_cnn, drop)]}_result.json'
            else:
                p = f'{GRID_RESULTS_DIR}/{cell_name(n_cnn, drop)}_result.json'
            if os.path.exists(p):
                with open(p) as f:
                    r = json.load(f)
                grid[i, j] = r['best_val_acc'] * 100
            else:
                print(f'  Missing: {p}')

    # ---- Heatmap ----
    fig, ax = plt.subplots(figsize=(8, 6))
    vmin = np.nanmin(grid); vmax = np.nanmax(grid)
    im = ax.imshow(grid, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')

    # Text annotations
    for i in range(len(N_CNN_LIST)):
        for j in range(len(DROPOUT_LIST)):
            val = grid[i, j]
            if not np.isnan(val):
                color = 'white' if val > (vmin + vmax) / 2 else 'black'
                ax.text(j, i, f'{val:.2f}%', ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')

    # Highlight best cell
    if not np.all(np.isnan(grid)):
        bi, bj = np.unravel_index(np.nanargmax(grid), grid.shape)
        ax.add_patch(plt.Rectangle((bj-0.5, bi-0.5), 1, 1, fill=False,
                                    edgecolor='lime', linewidth=3))

    ax.set_xticks(range(len(DROPOUT_LIST)))
    ax.set_xticklabels([f'{d:.1f}' for d in DROPOUT_LIST])
    ax.set_yticks(range(len(N_CNN_LIST)))
    ax.set_yticklabels([str(n) for n in N_CNN_LIST])
    ax.set_xlabel('Dropout', fontsize=12)
    ax.set_ylabel('n_cnn_layers', fontsize=12)
    ax.set_title('2-D Grid Search: Val Accuracy\n(n_mamba=1 fixed)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Val Accuracy (%)')
    plt.tight_layout()
    out = f'{FIG_DIR}/grid_search_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # ---- Print table ----
    header = "n_cnn \\ drop"
    print(f'\n{header:<12}', end='')
    for d in DROPOUT_LIST:
        print(f'{d:>8.1f}', end='')
    print()
    print('-' * (12 + 8 * len(DROPOUT_LIST)))
    for i, n_cnn in enumerate(N_CNN_LIST):
        print(f'{n_cnn:<12}', end='')
        for j in range(len(DROPOUT_LIST)):
            v = grid[i, j]
            print(f'{v:>7.2f}%' if not np.isnan(v) else f'{"--":>8}', end='')
        print()

    if not np.all(np.isnan(grid)):
        bi, bj = np.unravel_index(np.nanargmax(grid), grid.shape)
        print(f'\nBest cell: n_cnn={N_CNN_LIST[bi]}, dropout={DROPOUT_LIST[bj]}, '
              f'val_acc={grid[bi, bj]:.2f}%')


# -------------------------------------------------------------
# Entry
# -------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['launch', 'train', 'plot'], default='launch')
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.mode == 'launch':
        launch_all()
    elif args.mode == 'train':
        import torch
        cfg = next((c for c in CONFIGS if c['name'] == args.config), None)
        if cfg is None:
            print(f'Unknown config: {args.config}')
            sys.exit(1)

        # Temporarily redirect save dir by monkey-patching
        import ablation_cnn_mamba as abl
        abl.RESULTS_DIR = GRID_RESULTS_DIR

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        abl.train_config(cfg, device)

    elif args.mode == 'plot':
        plot_grid()
