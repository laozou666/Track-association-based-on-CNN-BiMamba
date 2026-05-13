"""
Fine-tune BiMamba layer count on top of the grid-search optimum.

Fixed:     n_cnn = 4,  dropout = 0.4
Variable:  n_mamba in {2, 3, 4}
Reused:    n_mamba = 1 result already exists from grid search
           (output/grid_search/g_cnn4_drop04_result.json, val_acc = 0.9238)

Per user instruction this script uses ONLY the single most idle GPU
(sequential training, no parallel launching across multiple GPUs).

Usage:
  python tune_mamba_layers.py            # auto pick idle GPU + train all
  python tune_mamba_layers.py --plot     # plot after training
"""

import os, sys, json, argparse, subprocess, time
# -----------------------------------------------------------
# Pick the single most idle GPU BEFORE importing torch
# -----------------------------------------------------------
def pick_idle_gpu():
    try:
        q = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            text=True).strip().splitlines()
        rows = []
        for line in q:
            idx, util, mem = [x.strip() for x in line.split(',')]
            rows.append((int(idx), int(util), int(mem)))
        # sort by (utilization, memory used) ascending
        rows.sort(key=lambda r: (r[1], r[2]))
        print(f"GPU status (idx, util%, mem MiB):")
        for r in rows:
            print(f"   GPU {r[0]}: util={r[1]:>3}%  mem={r[2]:>5} MiB")
        picked = rows[0][0]
        print(f"Picked GPU: {picked}")
        return picked
    except Exception as e:
        print(f"Could not query nvidia-smi ({e}), fallback to GPU 0")
        return 0


# --------------------- parse args ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true',
                    help='Plot only, skip training')
parser.add_argument('--gpu', type=int, default=None,
                    help='Override auto-picked GPU')
args = parser.parse_args()

if not args.plot:
    gpu_id = args.gpu if args.gpu is not None else pick_idle_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# -----------------------------------------------------------
# Now import torch (and reuse training infra from ablation)
# -----------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_cnn_mamba as ab       # reuse model + train_config
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False


# --------------------- paths ---------------------
RESULTS_DIR  = '/home/yangcq/track_association/output/mamba_tuning'
FIG_DIR      = '/home/yangcq/track_association/output/figures'
GRID_DIR     = '/home/yangcq/track_association/output/grid_search'
os.makedirs(RESULTS_DIR, exist_ok=True)

# redirect ablation's RESULTS_DIR to our folder so result files land here
ab.RESULTS_DIR = RESULTS_DIR


# --------------------- configs ---------------------
# n_cnn and dropout are fixed to the grid-search best (92.38%)
FIXED_N_CNN   = 4
FIXED_DROPOUT = 0.4

CONFIGS = [
    {'name': 'm2', 'n_cnn': FIXED_N_CNN, 'n_mamba': 2, 'dropout': FIXED_DROPOUT},
    {'name': 'm3', 'n_cnn': FIXED_N_CNN, 'n_mamba': 3, 'dropout': FIXED_DROPOUT},
    {'name': 'm4', 'n_cnn': FIXED_N_CNN, 'n_mamba': 4, 'dropout': FIXED_DROPOUT},
]


def train_all(device):
    for cfg in CONFIGS:
        result_path = f'{RESULTS_DIR}/{cfg["name"]}_result.json'
        if os.path.exists(result_path):
            with open(result_path) as f:
                r = json.load(f)
            print(f"[SKIP] {cfg['name']} already trained "
                  f"(best_val_acc={r['best_val_acc']:.4f})")
            continue
        t0 = time.time()
        ab.train_config(cfg, device)
        print(f"[DONE] {cfg['name']} in {time.time()-t0:.1f}s\n")


# ============================================================
#                        PLOTTING
# ============================================================
def collect_points():
    """Collect (n_mamba, val_acc, params) for all n_mamba values."""
    pts = []

    # n_mamba = 1  -> reuse grid-search result
    grid1 = f'{GRID_DIR}/g_cnn4_drop04_result.json'
    if os.path.exists(grid1):
        with open(grid1) as f:
            r = json.load(f)
        pts.append((1, r['best_val_acc'], r.get('n_params', None), 'grid_search'))

    # n_mamba = 2, 3, 4  -> results from this run
    for cfg in CONFIGS:
        p = f'{RESULTS_DIR}/{cfg["name"]}_result.json'
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            pts.append((cfg['n_mamba'], r['best_val_acc'],
                        r.get('n_params', None), 'tuning'))

    pts.sort(key=lambda t: t[0])
    return pts


def plot_tuning():
    pts = collect_points()
    if not pts:
        print("No results to plot yet."); return

    xs    = [p[0] for p in pts]
    accs  = [p[1] * 100 for p in pts]
    prms  = [p[2] for p in pts]

    best_idx = int(np.argmax(accs))
    best_n   = xs[best_idx]
    best_acc = accs[best_idx]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # --- line: val_acc vs n_mamba ---
    ax1.plot(xs, accs, 'o-', color='#2a7de1', linewidth=2.2,
             markersize=10, markerfacecolor='#2a7de1', label='Val Accuracy')
    for x, a in zip(xs, accs):
        ax1.annotate(f'{a:.2f}%', xy=(x, a), xytext=(0, 10),
                     textcoords='offset points', ha='center',
                     fontsize=10, color='#2a7de1', fontweight='bold')

    # highlight best
    ax1.scatter([best_n], [best_acc], s=260, marker='*',
                color='#ff7b00', zorder=5,
                label=f'Best (n_mamba={best_n}, {best_acc:.2f}%)')

    ax1.set_xlabel('Number of BiMamba Layers', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, color='#2a7de1')
    ax1.set_xticks(xs)
    ax1.tick_params(axis='y', labelcolor='#2a7de1')
    ax1.grid(True, alpha=0.35, linestyle='--')
    ymin = min(accs) - 1.0
    ymax = max(accs) + 1.5
    ax1.set_ylim(ymin, ymax)

    # --- second y-axis: parameter count ---
    if all(p is not None for p in prms):
        ax2 = ax1.twinx()
        ax2.plot(xs, [p/1e3 for p in prms], 's--', color='#888',
                 linewidth=1.4, markersize=6, alpha=0.85,
                 label='Parameters (K)')
        ax2.set_ylabel('Parameters (K)', fontsize=11, color='#555')
        ax2.tick_params(axis='y', labelcolor='#555')

    title = (f'BiMamba Layer Tuning  '
             f'(n_cnn = {FIXED_N_CNN}, dropout = {FIXED_DROPOUT} fixed)')
    ax1.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    out = f'{FIG_DIR}/mamba_layer_tuning.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    # also print a table
    print(f"\n{'n_mamba':<10}{'val_acc':<12}{'params':<12}{'source'}")
    print('-' * 50)
    for n, a, p, src in pts:
        p_str = f'{p/1e3:.1f}K' if p else 'N/A'
        print(f'{n:<10}{a*100:<12.4f}{p_str:<12}{src}')


# ============================================================
#                        MAIN
# ============================================================
if __name__ == '__main__':
    if args.plot:
        plot_tuning()
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        train_all(device)
        plot_tuning()
