"""
Train a "pure BiMamba" model (no CNN frontend) with the tuned dropout.
Used to demonstrate the contribution of the CNN module by comparing:
    n_cnn=0, n_mamba=1, dropout=0.4   (this run)   vs.
    n_cnn=4, n_mamba=1, dropout=0.4   (tuned BiMamba, 92.38%)

Results are written to output/pure_bimamba/ and a comparison bar chart to
output/figures/pure_vs_cnn_bimamba.png.
"""

import os, sys, json, subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def pick_idle_gpu():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits']
        ).decode().strip().splitlines()
        return int(np.argmin([int(x) for x in out]))
    except Exception as e:
        print(f"[WARN] nvidia-smi failed ({e}); fallback GPU 0")
        return 0


# Pick GPU BEFORE importing torch-heavy ablation module
GPU = pick_idle_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
print(f"[gpu] using physical GPU {GPU}")

import ablation_cnn_mamba as ab  # noqa: E402

RESULTS_DIR = '/home/yangcq/track_association/output/pure_bimamba'
FIG_DIR     = '/home/yangcq/track_association/output/figures'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Redirect the ablation module's output folder to our own
ab.RESULTS_DIR = RESULTS_DIR

BI_TUNED_RESULT = ('/home/yangcq/track_association/output/grid_search/'
                   'g_cnn4_drop04_result.json')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    cfg = {
        'name':   'pure_bimamba_d04',
        'n_cnn':   0,
        'n_mamba': 1,
        'dropout': 0.4,
    }
    print(f"[config] {cfg}")
    res_pure = ab.train_config(cfg, device)

    # --- Compare against tuned CNN-BiMamba (reuse existing result) ---
    with open(BI_TUNED_RESULT) as f:
        res_bi = json.load(f)

    # Normalise
    rows = [
        {
            'name':         'Pure BiMamba (n_cnn=0)',
            'n_params':     res_pure['n_params'],
            'best_val_acc': res_pure['best_val_acc'],
            'best_epoch':   res_pure['best_epoch'],
        },
        {
            'name':         'CNN-BiMamba tuned (n_cnn=4)',
            'n_params':     res_bi['n_params'],
            'best_val_acc': res_bi['best_val_acc'],
            'best_epoch':   res_bi['best_epoch'],
        },
    ]

    # --- Save tables ---
    csv_path = f'{RESULTS_DIR}/pure_vs_cnn_summary.csv'
    with open(csv_path, 'w') as f:
        f.write('name,n_params,val_acc,best_epoch\n')
        for r in rows:
            f.write(f"{r['name']},{r['n_params']},"
                    f"{r['best_val_acc']:.6f},{r['best_epoch']}\n")

    md_path = f'{RESULTS_DIR}/pure_vs_cnn_summary.md'
    with open(md_path, 'w') as f:
        f.write('# Pure BiMamba vs CNN-BiMamba (tuned)\n\n')
        f.write('Fixed: `n_mamba=1, dropout=0.4`\n\n')
        f.write('| Variant | Params | Val Acc | Best Epoch |\n')
        f.write('|---------|--------|---------|------------|\n')
        for r in rows:
            f.write(f"| {r['name']} | {r['n_params']:,} | "
                    f"{r['best_val_acc']*100:.2f}% | {r['best_epoch']} |\n")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names  = [r['name'] for r in rows]
    accs   = [r['best_val_acc'] * 100 for r in rows]
    params = [r['n_params'] / 1e3 for r in rows]
    colors = ['#70AD47', '#ED7D31']

    ax = axes[0]
    bars = ax.bar(names, accs, color=colors, edgecolor='black',
                  linewidth=0.7, width=0.5)
    ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=10)
    ax.set_ylabel('Best Val Accuracy (%)', fontsize=11)
    ax.set_title('Effect of CNN Frontend on BiMamba', fontsize=12)
    ax.set_ylim(max(0, min(accs) - 4), min(100, max(accs) + 4))
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    for r, c in zip(rows, colors):
        ax.scatter(r['n_params'] / 1e3, r['best_val_acc'] * 100,
                   s=180, c=c, edgecolor='black', linewidth=0.8)
        ax.annotate(r['name'],
                    (r['n_params'] / 1e3, r['best_val_acc'] * 100),
                    xytext=(8, 6), textcoords='offset points', fontsize=9)
    ax.set_xlabel('Parameters (K)', fontsize=11)
    ax.set_ylabel('Best Val Accuracy (%)', fontsize=11)
    ax.set_title('Params vs Accuracy', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f'{FIG_DIR}/pure_vs_cnn_bimamba.png'
    plt.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[plot] saved {fig_path}")
    print(f"[csv ] saved {csv_path}")
    print(f"[md  ] saved {md_path}")

    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    for r in rows:
        print(f"  {r['name']:32s}  "
              f"params={r['n_params']:>7,}  "
              f"val_acc={r['best_val_acc']*100:.2f}%")


if __name__ == '__main__':
    main()
