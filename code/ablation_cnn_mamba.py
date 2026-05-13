"""
Ablation study for CNN + Bidirectional Mamba track association model.

Ablation dimensions:
  1. n_cnn_layers : 0, 1, 2, 3
  2. n_mamba_layers: 1, 2, 3
  3. dropout       : 0.1, 0.2, 0.3, 0.4

Usage:
  # Train a single config (called internally):
  python ablation_cnn_mamba.py --mode train --config cnn1 --gpu 0

  # Launch all configs in parallel across 4 GPUs (default):
  python ablation_cnn_mamba.py --mode launch

  # Plot results after all training is done:
  python ablation_cnn_mamba.py --mode plot
"""

import os, sys, json, argparse, subprocess, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ===================== Paths =====================
DATA_DIR    = '/home/yangcq/track_association/data/final_dataset'
RESULTS_DIR = '/home/yangcq/track_association/output/ablation'
FIG_DIR     = '/home/yangcq/track_association/output/figures'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,     exist_ok=True)

# ===================== Ablation configs =====================
CONFIGS = [
    # --- CNN layers ablation (n_mamba=1, dropout=0.2) ---
    {'name': 'cnn0',   'n_cnn': 0, 'n_mamba': 1, 'dropout': 0.2},
    {'name': 'cnn1',   'n_cnn': 1, 'n_mamba': 1, 'dropout': 0.2},  # baseline
    {'name': 'cnn2',   'n_cnn': 2, 'n_mamba': 1, 'dropout': 0.2},
    {'name': 'cnn3',   'n_cnn': 3, 'n_mamba': 1, 'dropout': 0.2},
    # --- Mamba layers ablation (n_cnn=1, dropout=0.2) ---
    {'name': 'mamba2', 'n_cnn': 1, 'n_mamba': 2, 'dropout': 0.2},
    {'name': 'mamba3', 'n_cnn': 1, 'n_mamba': 3, 'dropout': 0.2},
    # --- Dropout ablation (n_cnn=1, n_mamba=1) ---
    {'name': 'drop01', 'n_cnn': 1, 'n_mamba': 1, 'dropout': 0.1},
    {'name': 'drop03', 'n_cnn': 1, 'n_mamba': 1, 'dropout': 0.3},
    {'name': 'drop04', 'n_cnn': 1, 'n_mamba': 1, 'dropout': 0.4},
    # --- Combined optimal config ---
    {'name': 'best',   'n_cnn': 3, 'n_mamba': 2, 'dropout': 0.4},
]

# GPU assignment: each GPU runs its list sequentially
GPU_ASSIGN = {
    0: ['cnn0',   'cnn1'],
    1: ['cnn2',   'cnn3'],
    2: ['mamba2', 'mamba3'],
    3: ['drop01', 'drop03', 'drop04'],
}

# ===================== Training hyper-params =====================
EPOCHS     = 25
BATCH_SIZE = 64
LR         = 2e-4
D_MODEL    = 64
D_STATE    = 32
D_CONV     = 4
EXPAND     = 2
PATIENCE   = 7


# ================================================================
# Model Components
# ================================================================
class LinearEmbedder(nn.Module):
    """n_cnn=0: simple linear projection."""
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):          # x: (B, T, input_dim)
        return F.gelu(self.norm(self.proj(x)))


class CNNBlock(nn.Module):
    """1-D CNN residual block: (B,T,in_ch) -> (B,T,out_ch)."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 5, padding=2)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.proj  = nn.Conv1d(in_ch,  out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x   = x.permute(0, 2, 1)          # (B, C, T)
        res = self.proj(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = F.gelu(self.bn2(self.conv2(out)) + res)
        return out.permute(0, 2, 1)        # (B, T, C)


class BiMambaSeqLayer(nn.Module):
    """BiMamba layer: (B,T,d) -> (B,T,d) with residual connection."""
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        from mamba_ssm import Mamba
        self.norm_fwd  = nn.LayerNorm(d_model)
        self.norm_bwd  = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model, d_state, d_conv, expand)
        self.mamba_bwd = Mamba(d_model, d_state, d_conv, expand)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x):
        fwd = self.mamba_fwd(self.norm_fwd(x))
        bwd = torch.flip(
            self.mamba_bwd(self.norm_bwd(torch.flip(x, dims=[1]))),
            dims=[1]
        )
        return self.drop(x + (fwd + bwd) * 0.5)


class AblationCNNMamba(nn.Module):
    """
    Ablation-friendly CNN + stacked BiMamba model.
    n_cnn    : 0 = linear proj, >=1 = that many CNN residual blocks
    n_mamba  : number of stacked BiMamba sequence layers
    dropout  : dropout rate
    """
    def __init__(self, input_dim=4, d_model=D_MODEL,
                 n_cnn=1, n_mamba=1,
                 d_state=D_STATE, d_conv=D_CONV, expand=EXPAND,
                 dropout=0.2):
        super().__init__()

        # --- Embedder ---
        if n_cnn == 0:
            self.embedder = LinearEmbedder(input_dim, d_model)
        else:
            blocks = [CNNBlock(input_dim, d_model, dropout * 0.5)]
            for _ in range(n_cnn - 1):
                blocks.append(CNNBlock(d_model, d_model, dropout * 0.5))
            self.embedder = nn.Sequential(*blocks)

        # --- Stacked BiMamba ---
        self.mamba_layers = nn.ModuleList([
            BiMambaSeqLayer(d_model, d_state, d_conv, expand, dropout * 0.5)
            for _ in range(n_mamba)
        ])

        # --- Classifier  (cross-track interaction: [a,b,a-b,|a-b|]) ---
        enc_dim    = d_model * 2   # mean-pool + max-pool
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
        h = self.embedder(x)              # (B, T, d_model)
        for layer in self.mamba_layers:
            h = layer(h)
        return torch.cat([h.mean(1), h.max(1).values], dim=-1)  # (B, 2*d_model)

    def forward(self, traj_a, traj_b):
        g_a = self._encode(traj_a)
        g_b = self._encode(traj_b)
        comb = torch.cat([g_a, g_b, g_a - g_b, torch.abs(g_a - g_b)], dim=-1)
        return self.classifier(comb).squeeze(-1)


# ================================================================
# Training
# ================================================================
def load_data():
    t1_tr = torch.FloatTensor(np.load(f'{DATA_DIR}/track1_train.npy'))
    t2_tr = torch.FloatTensor(np.load(f'{DATA_DIR}/track2_train.npy'))
    lb_tr = torch.FloatTensor(np.load(f'{DATA_DIR}/labels_train.npy'))
    t1_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/track1_val.npy'))
    t2_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/track2_val.npy'))
    lb_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/labels_val.npy'))
    pos_w = torch.tensor([(lb_tr == 0).sum() / (lb_tr == 1).sum()])
    train_ds = TensorDataset(t1_tr, t2_tr, lb_tr)
    val_ds   = TensorDataset(t1_vl, t2_vl, lb_vl)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=128,        shuffle=False, num_workers=4, pin_memory=True)
    return train_ld, val_ld, pos_w


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, correct, total = 0., 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for t1, t2, lb in loader:
            t1, t2, lb = t1.to(device), t2.to(device), lb.to(device)
            out = model(t1, t2)
            loss = criterion(out, lb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(lb)
            correct    += ((torch.sigmoid(out) > 0.5) == lb.bool()).sum().item()
            total      += len(lb)
    return total_loss / total, correct / total


def train_config(cfg, device):
    print(f"\n{'='*55}")
    print(f"  Config: {cfg['name']}  |  n_cnn={cfg['n_cnn']}  "
          f"n_mamba={cfg['n_mamba']}  dropout={cfg['dropout']}")
    print(f"{'='*55}")

    train_ld, val_ld, pos_w = load_data()
    model = AblationCNNMamba(
        n_cnn=cfg['n_cnn'], n_mamba=cfg['n_mamba'], dropout=cfg['dropout']
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience_cnt  = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        tr_loss, _        = run_epoch(model, train_ld, criterion, optimizer, device, train=True)
        vl_loss, vl_acc   = run_epoch(model, val_ld,   criterion, optimizer, device, train=False)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        marker = ''
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_val_acc  = vl_acc
            best_epoch    = epoch + 1
            torch.save(model.state_dict(),
                       f'{RESULTS_DIR}/{cfg["name"]}_best.pth')
            patience_cnt = 0
            marker = '  ✓'
        else:
            patience_cnt += 1

        print(f"  Epoch {epoch+1:02d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"val_acc={vl_acc:.4f}{marker}")

        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    result = {
        'name':         cfg['name'],
        'n_cnn':        cfg['n_cnn'],
        'n_mamba':      cfg['n_mamba'],
        'dropout':      cfg['dropout'],
        'n_params':     n_params,
        'best_val_acc': best_val_acc,
        'best_val_loss':best_val_loss,
        'best_epoch':   best_epoch,
        'history':      history,
    }
    with open(f'{RESULTS_DIR}/{cfg["name"]}_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  >> Best val_acc={best_val_acc:.4f}  (epoch {best_epoch})")
    return result


# ================================================================
# Plotting
# ================================================================
def plot_ablation():
    # Load all results
    results = {}
    for cfg in CONFIGS:
        p = f'{RESULTS_DIR}/{cfg["name"]}_result.json'
        if os.path.exists(p):
            with open(p) as f:
                results[cfg['name']] = json.load(f)
        else:
            print(f"  Missing: {p}")

    if not results:
        print("No results found.")
        return

    # ---- Figure 1: 3-panel ablation bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('CNN + BiMamba Ablation Study', fontsize=14, fontweight='bold')

    bar_kw = dict(width=0.5, edgecolor='black', linewidth=0.7)

    # Panel 1: CNN layers
    ax = axes[0]
    names  = ['cnn0', 'cnn1', 'cnn2', 'cnn3']
    labels = ['0 (Linear)', '1', '2', '3']
    accs   = [results[n]['best_val_acc'] * 100 for n in names if n in results]
    xlabels= [labels[i] for i, n in enumerate(names) if n in results]
    bars = ax.bar(xlabels, accs, color='#4472C4', **bar_kw)
    ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9)
    ax.set_title('Number of CNN Layers\n(n_mamba=1, dropout=0.2)', fontsize=10)
    ax.set_xlabel('n_cnn_layers')
    ax.set_ylabel('Best Val Accuracy (%)')
    ax.set_ylim(max(0, min(accs) - 5), min(100, max(accs) + 5))
    ax.grid(axis='y', alpha=0.3)
    # Highlight best
    best_idx = accs.index(max(accs))
    bars[best_idx].set_color('#ED7D31')

    # Panel 2: Mamba layers
    ax = axes[1]
    names  = ['cnn1', 'mamba2', 'mamba3']
    labels = ['1', '2', '3']
    accs   = [results[n]['best_val_acc'] * 100 for n in names if n in results]
    xlabels= [labels[i] for i, n in enumerate(names) if n in results]
    bars = ax.bar(xlabels, accs, color='#4472C4', **bar_kw)
    ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9)
    ax.set_title('Number of BiMamba Layers\n(n_cnn=1, dropout=0.2)', fontsize=10)
    ax.set_xlabel('n_mamba_layers')
    ax.set_ylabel('Best Val Accuracy (%)')
    ax.set_ylim(max(0, min(accs) - 5), min(100, max(accs) + 5))
    ax.grid(axis='y', alpha=0.3)
    best_idx = accs.index(max(accs))
    bars[best_idx].set_color('#ED7D31')

    # Panel 3: Dropout
    ax = axes[2]
    names  = ['drop01', 'cnn1', 'drop03', 'drop04']
    labels = ['0.1', '0.2', '0.3', '0.4']
    accs   = [results[n]['best_val_acc'] * 100 for n in names if n in results]
    xlabels= [labels[i] for i, n in enumerate(names) if n in results]
    bars = ax.bar(xlabels, accs, color='#4472C4', **bar_kw)
    ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9)
    ax.set_title('Dropout Rate\n(n_cnn=1, n_mamba=1)', fontsize=10)
    ax.set_xlabel('dropout')
    ax.set_ylabel('Best Val Accuracy (%)')
    ax.set_ylim(max(0, min(accs) - 5), min(100, max(accs) + 5))
    ax.grid(axis='y', alpha=0.3)
    best_idx = accs.index(max(accs))
    bars[best_idx].set_color('#ED7D31')

    plt.tight_layout()
    out1 = f'{FIG_DIR}/ablation_bar.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out1}")

    # ---- Figure 2: training curves for each ablation dimension ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CNN + BiMamba Ablation — Validation Accuracy Curves',
                 fontsize=13, fontweight='bold')

    color_map = plt.cm.tab10.colors

    def plot_curves(ax, name_list, label_list, title):
        for i, (n, lbl) in enumerate(zip(name_list, label_list)):
            if n not in results:
                continue
            accs = [v * 100 for v in results[n]['history']['val_acc']]
            ax.plot(range(1, len(accs)+1), accs,
                    color=color_map[i], lw=1.8, marker='o', markersize=3,
                    label=lbl)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Accuracy (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plot_curves(axes[0],
                ['cnn0','cnn1','cnn2','cnn3'],
                ['n_cnn=0','n_cnn=1','n_cnn=2','n_cnn=3'],
                'CNN Layers Ablation')
    plot_curves(axes[1],
                ['cnn1','mamba2','mamba3'],
                ['n_mamba=1','n_mamba=2','n_mamba=3'],
                'Mamba Layers Ablation')
    plot_curves(axes[2],
                ['drop01','cnn1','drop03','drop04'],
                ['dropout=0.1','dropout=0.2','dropout=0.3','dropout=0.4'],
                'Dropout Rate Ablation')

    plt.tight_layout()
    out2 = f'{FIG_DIR}/ablation_curves.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out2}")

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print(f"{'Config':<10} {'n_cnn':>6} {'n_mamba':>8} {'dropout':>8} "
          f"{'Val Acc':>9} {'Params':>10} {'Epoch':>6}")
    print('-'*60)
    for cfg in CONFIGS:
        n = cfg['name']
        if n not in results:
            continue
        r = results[n]
        print(f"{n:<10} {r['n_cnn']:>6} {r['n_mamba']:>8} {r['dropout']:>8.1f} "
              f"{r['best_val_acc']*100:>8.2f}% {r['n_params']:>10,} {r['best_epoch']:>6}")
    print('='*60)


# ================================================================
# Launcher
# ================================================================
def launch_all():
    procs = []
    script = os.path.abspath(__file__)

    for gpu_id, cfg_names in GPU_ASSIGN.items():
        # Build sequential shell command for this GPU
        cmds = []
        for cn in cfg_names:
            cmds.append(
                f"CUDA_VISIBLE_DEVICES={gpu_id} "
                f"conda run -n track_association python3 {script} "
                f"--mode train --config {cn} --gpu {gpu_id}"
            )
        shell_cmd = ' && '.join(cmds)
        log_file  = f'/tmp/ablation_gpu{gpu_id}.log'
        p = subprocess.Popen(
            ['bash', '-c', f'({shell_cmd}) > {log_file} 2>&1'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        procs.append((gpu_id, cfg_names, p, log_file))
        print(f"GPU {gpu_id}: {cfg_names}  -> {log_file}")

    print(f"\nWaiting for all {sum(len(v) for v in GPU_ASSIGN.values())} configs...")
    for gpu_id, cfg_names, p, log_file in procs:
        p.wait()
        print(f"GPU {gpu_id} finished: {cfg_names}")

    print("\nAll training done. Generating plots...")
    plot_ablation()


# ================================================================
# Entry point
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   choices=['launch', 'train', 'plot'], default='launch')
    parser.add_argument('--config', type=str, help='Config name for train mode')
    parser.add_argument('--gpu',    type=int, default=0)
    args = parser.parse_args()

    if args.mode == 'launch':
        launch_all()

    elif args.mode == 'train':
        cfg = next((c for c in CONFIGS if c['name'] == args.config), None)
        if cfg is None:
            print(f"Unknown config: {args.config}")
            sys.exit(1)
        # CUDA_VISIBLE_DEVICES remaps the GPU to index 0 within the process
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_config(cfg, device)

    elif args.mode == 'plot':
        plot_ablation()
