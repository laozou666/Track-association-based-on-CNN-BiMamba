"""
Direction Ablation for CNN-Mamba
--------------------------------
Compare Forward-only / Backward-only / Bi-directional Mamba
under the tuned optimal hyperparameters (n_cnn=4, n_mamba=1, dropout=0.4).

Only trains the two uni-directional variants; reuses the existing tuned BiMamba
result from output/grid_search/g_cnn4_drop04_*.
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# Make sure we can import ablation_cnn_mamba as a module
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import ablation_cnn_mamba as ab  # reuses CNNBlock, LinearEmbedder, constants, data loader


# ===================== Paths =====================
RESULTS_DIR = '/home/yangcq/track_association/output/direction_ablation'
FIG_DIR     = '/home/yangcq/track_association/output/figures'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Tuned optimal config from 2-D grid search (BiMamba baseline: 92.38%)
FIXED_N_CNN   = 4
FIXED_N_MAMBA = 1
FIXED_DROPOUT = 0.4

# Reuse existing BiMamba baseline instead of re-training
BI_RESULT_PATH = '/home/yangcq/track_association/output/grid_search/g_cnn4_drop04_result.json'
BI_WEIGHT_PATH = '/home/yangcq/track_association/output/grid_search/g_cnn4_drop04_best.pth'


# ================================================================
#  Uni-directional BiMamba replacement
# ================================================================
class UniMambaSeqLayer(nn.Module):
    """
    Single-direction Mamba layer with residual connection.
    direction = 'fwd' : process x in original order.
    direction = 'bwd' : reverse the sequence before Mamba, reverse again after.

    Keeps the same output dimensions as BiMambaSeqLayer so the surrounding
    architecture (CNN embedder, classifier head) stays identical.
    """
    def __init__(self, d_model, d_state, d_conv, expand, dropout, direction='fwd'):
        super().__init__()
        assert direction in ('fwd', 'bwd')
        from mamba_ssm import Mamba
        self.direction = direction
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model, d_state, d_conv, expand)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        if self.direction == 'fwd':
            h = self.mamba(self.norm(x))
        else:
            h_in = torch.flip(x, dims=[1])
            h    = torch.flip(self.mamba(self.norm(h_in)), dims=[1])
        return self.drop(x + h)


class DirectionalCNNMamba(nn.Module):
    """Same as ab.AblationCNNMamba but Mamba layers are uni-directional."""
    def __init__(self, input_dim=4, d_model=ab.D_MODEL,
                 n_cnn=FIXED_N_CNN, n_mamba=FIXED_N_MAMBA,
                 d_state=ab.D_STATE, d_conv=ab.D_CONV, expand=ab.EXPAND,
                 dropout=FIXED_DROPOUT, direction='fwd'):
        super().__init__()

        if n_cnn == 0:
            self.embedder = ab.LinearEmbedder(input_dim, d_model)
        else:
            blocks = [ab.CNNBlock(input_dim, d_model, dropout * 0.5)]
            for _ in range(n_cnn - 1):
                blocks.append(ab.CNNBlock(d_model, d_model, dropout * 0.5))
            self.embedder = nn.Sequential(*blocks)

        self.mamba_layers = nn.ModuleList([
            UniMambaSeqLayer(d_model, d_state, d_conv, expand,
                             dropout * 0.5, direction=direction)
            for _ in range(n_mamba)
        ])

        enc_dim    = d_model * 2
        fusion_dim = enc_dim * 4
        hidden     = enc_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def _encode(self, x):
        h = self.embedder(x)
        for layer in self.mamba_layers:
            h = layer(h)
        return torch.cat([h.mean(1), h.max(1).values], dim=-1)

    def forward(self, traj_a, traj_b):
        g_a = self._encode(traj_a)
        g_b = self._encode(traj_b)
        comb = torch.cat([g_a, g_b, g_a - g_b, torch.abs(g_a - g_b)], dim=-1)
        return self.classifier(comb).squeeze(-1)


# ================================================================
#  GPU selection (pick the single idlest one)
# ================================================================
def pick_idle_gpu():
    """Return the index of the GPU with the least used memory."""
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
        ).decode().strip().splitlines()
        used = [int(x) for x in out]
        return int(np.argmin(used))
    except Exception as e:
        print(f"[WARN] Could not query nvidia-smi ({e}), falling back to GPU 0")
        return 0


# ================================================================
#  Training (mirrors ab.train_config but plugs in DirectionalCNNMamba)
# ================================================================
def train_directional(direction, device):
    cfg_name = f'mamba_{direction}'
    print(f"\n{'='*60}")
    print(f"  Direction-ablation:  {cfg_name}  "
          f"(n_cnn={FIXED_N_CNN}, n_mamba={FIXED_N_MAMBA}, "
          f"dropout={FIXED_DROPOUT})")
    print(f"{'='*60}", flush=True)

    train_ld, val_ld, pos_w = ab.load_data()
    model = DirectionalCNNMamba(direction=direction).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}", flush=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    optimizer = optim.Adam(model.parameters(), lr=ab.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    best_val_acc  = 0.0
    best_epoch    = -1
    patience_cnt  = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    t0 = time.time()
    for epoch in range(ab.EPOCHS):
        tr_loss, _      = ab.run_epoch(model, train_ld, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = ab.run_epoch(model, val_ld,   criterion, optimizer, device, train=False)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        marker = ''
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_val_acc  = vl_acc
            best_epoch    = epoch + 1
            torch.save(model.state_dict(), f'{RESULTS_DIR}/{cfg_name}_best.pth')
            patience_cnt = 0
            marker = '  ✓'
        else:
            patience_cnt += 1

        print(f"  Epoch {epoch+1:02d}/{ab.EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"val_acc={vl_acc:.4f}{marker}", flush=True)

        if patience_cnt >= ab.PATIENCE:
            print(f"  Early stop at epoch {epoch+1}", flush=True)
            break

    elapsed = time.time() - t0
    result = {
        'name':          cfg_name,
        'direction':     direction,
        'n_cnn':         FIXED_N_CNN,
        'n_mamba':       FIXED_N_MAMBA,
        'dropout':       FIXED_DROPOUT,
        'n_params':      n_params,
        'best_val_acc':  best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch':    best_epoch,
        'train_seconds': elapsed,
        'history':       history,
    }
    with open(f'{RESULTS_DIR}/{cfg_name}_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  >> {cfg_name}: Best val_acc={best_val_acc:.4f}  "
          f"(epoch {best_epoch})  [{elapsed:.1f}s]", flush=True)
    return result


# ================================================================
#  Aggregate: reuse existing BiMamba + plot 3-way comparison
# ================================================================
def _load_bi_result():
    """Load the existing tuned BiMamba result (from grid_search) and
    normalise it into the same JSON shape as our directional runs."""
    with open(BI_RESULT_PATH) as f:
        g = json.load(f)
    return {
        'name':          'mamba_bi',
        'direction':     'bi',
        'n_cnn':         g['n_cnn'],
        'n_mamba':       g['n_mamba'],
        'dropout':       g['dropout'],
        'n_params':      g['n_params'],
        'best_val_acc':  g['best_val_acc'],
        'best_val_loss': g['best_val_loss'],
        'best_epoch':    g['best_epoch'],
        'history':       g.get('history', {}),
        'source':        BI_RESULT_PATH,
    }


def plot_comparison():
    results = []
    for d in ('fwd', 'bwd'):
        p = f'{RESULTS_DIR}/mamba_{d}_result.json'
        if os.path.exists(p):
            with open(p) as f:
                results.append(json.load(f))
    results.append(_load_bi_result())

    # --- persist summary ---
    summary = [
        {
            'direction':    r['direction'],
            'name':         r['name'],
            'n_params':     r['n_params'],
            'best_val_acc': r['best_val_acc'],
            'best_val_loss':r['best_val_loss'],
            'best_epoch':   r['best_epoch'],
        }
        for r in results
    ]
    with open(f'{RESULTS_DIR}/direction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # --- CSV + Markdown tables ---
    csv_path = f'{RESULTS_DIR}/direction_summary.csv'
    with open(csv_path, 'w') as f:
        f.write('direction,params,best_val_acc,best_val_loss,best_epoch\n')
        for r in summary:
            f.write(f"{r['direction']},{r['n_params']},"
                    f"{r['best_val_acc']:.6f},{r['best_val_loss']:.6f},"
                    f"{r['best_epoch']}\n")

    md_path = f'{RESULTS_DIR}/direction_summary.md'
    with open(md_path, 'w') as f:
        f.write('# Mamba Direction Ablation\n\n')
        f.write(f'Fixed config: `n_cnn={FIXED_N_CNN}, n_mamba={FIXED_N_MAMBA}, '
                f'dropout={FIXED_DROPOUT}`\n\n')
        f.write('| Direction | Params | Val Acc | Val Loss | Best Epoch |\n')
        f.write('|-----------|--------|---------|----------|------------|\n')
        for r in summary:
            f.write(f"| {r['direction']} | {r['n_params']:,} | "
                    f"{r['best_val_acc']*100:.2f}% | "
                    f"{r['best_val_loss']:.4f} | {r['best_epoch']} |\n")

    # --- Figure: 2-panel (accuracy bars + params vs acc scatter) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    label_map = {'fwd': 'Forward-only', 'bwd': 'Backward-only', 'bi': 'BiMamba'}
    color_map = {'fwd': '#4472C4', 'bwd': '#70AD47', 'bi': '#ED7D31'}
    order     = ['fwd', 'bwd', 'bi']
    results   = sorted(results, key=lambda r: order.index(r['direction']))

    # Panel 1: accuracy bars
    ax = axes[0]
    xs     = [label_map[r['direction']] for r in results]
    accs   = [r['best_val_acc'] * 100 for r in results]
    colors = [color_map[r['direction']] for r in results]
    bars = ax.bar(xs, accs, color=colors, edgecolor='black', linewidth=0.7, width=0.55)
    ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=10)
    ax.set_ylabel('Best Val Accuracy (%)', fontsize=11)
    ax.set_title('Mamba Direction Ablation  (val accuracy)', fontsize=12)
    ylo = max(0, min(accs) - 3)
    yhi = min(100, max(accs) + 3)
    ax.set_ylim(ylo, yhi)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: params vs acc
    ax = axes[1]
    for r in results:
        ax.scatter(r['n_params'] / 1e3, r['best_val_acc'] * 100,
                   s=160, c=color_map[r['direction']],
                   edgecolor='black', linewidth=0.8,
                   label=label_map[r['direction']])
        ax.annotate(label_map[r['direction']],
                    (r['n_params'] / 1e3, r['best_val_acc'] * 100),
                    xytext=(8, 6), textcoords='offset points', fontsize=10)
    ax.set_xlabel('Parameters (K)', fontsize=11)
    ax.set_ylabel('Best Val Accuracy (%)', fontsize=11)
    ax.set_title('Params vs Accuracy', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = f'{FIG_DIR}/direction_ablation.png'
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[plot] saved {out_png}")
    print(f"[csv ] saved {csv_path}")
    print(f"[md  ] saved {md_path}")

    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    for r in summary:
        print(f"  {label_map[r['direction']]:15s}  "
              f"params={r['n_params']:>7,}  "
              f"val_acc={r['best_val_acc']*100:.2f}%")


# ================================================================
#  Entry
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'plot'], default='train',
                        help='train: run fwd+bwd training then plot; '
                             'plot: only aggregate existing results')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU index. If omitted, picks the idlest one.')
    args = parser.parse_args()

    if args.mode == 'plot':
        plot_comparison()
        return

    # --- pick GPU (user requested single-GPU mode) ---
    gpu = args.gpu if args.gpu is not None else pick_idle_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print(f"[gpu] using physical GPU {gpu}")

    import torch  # reimport after CUDA_VISIBLE_DEVICES
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    if not os.path.exists(BI_WEIGHT_PATH):
        print(f"[WARN] existing BiMamba weights not found at {BI_WEIGHT_PATH};"
              f" plot will still run but BiMamba row may be missing.")

    for d in ('fwd', 'bwd'):
        train_directional(d, device)

    plot_comparison()


if __name__ == '__main__':
    main()
