"""
Length-aware CNN + BiMamba (v2) for track association.

Key fixes over v1 (ablation_cnn_mamba.py):
  1. Backward Mamba only processes the REAL part of each sequence.
     (The naive torch.flip(x) in v1 puts 250+ padding frames BEFORE the
      real frames when running the backward scan, polluting the state.)
  2. Pooling (mean + max) is done ONLY over real frames using a mask.
  3. DataLoader now returns per-sample length.

Everything is fully vectorized — no per-sample Python loops in the forward pass.

Usage:
  python ablation_cnn_mamba_v2.py --mode launch
  python ablation_cnn_mamba_v2.py --mode train --config v2_best --gpu 0
  python ablation_cnn_mamba_v2.py --mode plot
"""

import os, sys, json, argparse, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# =============================================================
DATA_DIR    = '/home/yangcq/track_association/data/final_dataset'
RESULTS_DIR = '/home/yangcq/track_association/output/ablation_v2'
FIG_DIR     = '/home/yangcq/track_association/output/figures'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,     exist_ok=True)

# ---- Configs: verify the 4 best-region points with the v2 model ----
CONFIGS = [
    # Keep (n_cnn=4, drop=0.4) family: best cell from grid search
    {'name': 'v2_c4_m1_d04', 'n_cnn': 4, 'n_mamba': 1, 'dropout': 0.4},
    {'name': 'v2_c4_m2_d04', 'n_cnn': 4, 'n_mamba': 2, 'dropout': 0.4},
    # (n_cnn=3, drop=0.2) family: another strong cell
    {'name': 'v2_c3_m1_d02', 'n_cnn': 3, 'n_mamba': 1, 'dropout': 0.2},
    {'name': 'v2_c3_m2_d02', 'n_cnn': 3, 'n_mamba': 2, 'dropout': 0.2},
    # Baseline equivalent for sanity check
    {'name': 'v2_c1_m1_d02', 'n_cnn': 1, 'n_mamba': 1, 'dropout': 0.2},
    # Extra: try more Mamba layers with the strongest CNN backbone
    {'name': 'v2_c4_m3_d04', 'n_cnn': 4, 'n_mamba': 3, 'dropout': 0.4},
]

GPU_ASSIGN = {
    0: ['v2_c4_m1_d04', 'v2_c1_m1_d02'],
    1: ['v2_c4_m2_d04'],
    2: ['v2_c3_m1_d02', 'v2_c3_m2_d02'],
    3: ['v2_c4_m3_d04'],
}

# ---- Hyper-params ----
EPOCHS     = 25
BATCH_SIZE = 64
LR         = 2e-4
D_MODEL    = 64
D_STATE    = 32
D_CONV     = 4
EXPAND     = 2
PATIENCE   = 7


# ====================================================================
# Utilities: length-aware operations
# ====================================================================
def real_mask_from_lengths(lengths, T):
    """(B,) int -> (B, T) bool, True for real frames."""
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


def reverse_real_only(x, lengths):
    """
    Reverse the first L[i] frames of each sample, leave padding as zeros.

    x       : (B, T, D)
    lengths : (B,)
    returns : (B, T, D) with x_rev[i, :L[i]] = x[i, L[i]-1::-1], rest 0.
    """
    B, T, D = x.shape
    positions = torch.arange(T, device=x.device)[None, :]           # (1, T)
    L  = lengths[:, None]                                            # (B, 1)
    src_idx = (L - 1 - positions).clamp(min=0)                       # (B, T)
    src_idx_exp = src_idx.unsqueeze(-1).expand(-1, -1, D)             # (B, T, D)
    gathered = torch.gather(x, 1, src_idx_exp)                       # (B, T, D)
    mask = (positions < L).unsqueeze(-1).to(x.dtype)                 # (B, T, 1)
    return gathered * mask


def masked_mean_max_pool(h, lengths):
    """
    h       : (B, T, D)
    lengths : (B,)
    returns : (B, 2D)  -- concat[masked_mean, masked_max]
    """
    B, T, D = h.shape
    mask = real_mask_from_lengths(lengths, T)               # (B, T) bool
    mask_f = mask.unsqueeze(-1).to(h.dtype)                  # (B, T, 1)
    # mean over real frames only
    s = (h * mask_f).sum(dim=1)
    denom = lengths.clamp(min=1).unsqueeze(-1).to(h.dtype)   # (B, 1)
    mean = s / denom
    # max over real frames only (set padding to -inf)
    neg_inf = torch.finfo(h.dtype).min
    masked_h = h.masked_fill(~mask.unsqueeze(-1), neg_inf)
    maxv = masked_h.max(dim=1).values
    return torch.cat([mean, maxv], dim=-1)


# ====================================================================
# Model components
# ====================================================================
class LinearEmbedder(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths=None):
        return F.gelu(self.norm(self.proj(x)))


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 5, padding=2)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.proj  = nn.Conv1d(in_ch,  out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        x   = x.permute(0, 2, 1)
        res = self.proj(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = F.gelu(self.bn2(self.conv2(out)) + res)
        return out.permute(0, 2, 1)


class LengthAwareBiMambaLayer(nn.Module):
    """
    Bidirectional Mamba layer where the backward branch only processes the
    real (non-padded) portion of each sample.
    """
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        from mamba_ssm import Mamba
        self.norm_fwd  = nn.LayerNorm(d_model)
        self.norm_bwd  = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model, d_state, d_conv, expand)
        self.mamba_bwd = Mamba(d_model, d_state, d_conv, expand)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # Forward branch: zero out padding after scan (Mamba itself doesn't know)
        fwd = self.mamba_fwd(self.norm_fwd(x))

        # Backward branch: reverse only the real part, scan, then reverse back
        x_bwd_in  = reverse_real_only(self.norm_bwd(x), lengths)
        bwd_raw   = self.mamba_bwd(x_bwd_in)
        bwd       = reverse_real_only(bwd_raw, lengths)

        # Mask out padding positions in both outputs
        T    = x.size(1)
        mask = real_mask_from_lengths(lengths, T).unsqueeze(-1).to(x.dtype)
        fused = (fwd + bwd) * 0.5
        fused = fused * mask

        # Residual (x already has 0 in padding region from embedder/CNN)
        return self.drop(x + fused)


class AblationCNNMambaV2(nn.Module):
    """Length-aware version. forward(traj_a, traj_b, lengths_a, lengths_b)."""
    def __init__(self, input_dim=4, d_model=D_MODEL,
                 n_cnn=1, n_mamba=1,
                 d_state=D_STATE, d_conv=D_CONV, expand=EXPAND,
                 dropout=0.2):
        super().__init__()

        # --- Embedder ---
        if n_cnn == 0:
            self.embedder = nn.ModuleList([LinearEmbedder(input_dim, d_model)])
        else:
            layers = [CNNBlock(input_dim, d_model, dropout * 0.5)]
            for _ in range(n_cnn - 1):
                layers.append(CNNBlock(d_model, d_model, dropout * 0.5))
            self.embedder = nn.ModuleList(layers)

        # --- BiMamba stack ---
        self.mamba_layers = nn.ModuleList([
            LengthAwareBiMambaLayer(d_model, d_state, d_conv, expand, dropout * 0.5)
            for _ in range(n_mamba)
        ])

        # --- Classifier ---
        enc_dim    = d_model * 2        # masked mean + masked max
        fusion_dim = enc_dim * 4         # [a, b, a-b, |a-b|]
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

    def _encode(self, x, lengths):
        h = x
        for layer in self.embedder:
            h = layer(h, lengths)
        # Ensure padding is zero after embedder (BN can add a bias)
        T    = h.size(1)
        mask = real_mask_from_lengths(lengths, T).unsqueeze(-1).to(h.dtype)
        h = h * mask
        for layer in self.mamba_layers:
            h = layer(h, lengths)
            h = h * mask          # keep padding at zero between layers
        return masked_mean_max_pool(h, lengths)

    def forward(self, traj_a, traj_b, lengths_a, lengths_b):
        g_a = self._encode(traj_a, lengths_a)
        g_b = self._encode(traj_b, lengths_b)
        comb = torch.cat([g_a, g_b, g_a - g_b, torch.abs(g_a - g_b)], dim=-1)
        return self.classifier(comb).squeeze(-1)


# ====================================================================
# Training
# ====================================================================
def load_data():
    t1_tr = torch.FloatTensor(np.load(f'{DATA_DIR}/track1_train.npy'))
    t2_tr = torch.FloatTensor(np.load(f'{DATA_DIR}/track2_train.npy'))
    lb_tr = torch.FloatTensor(np.load(f'{DATA_DIR}/labels_train.npy'))
    ln_tr = torch.LongTensor(np.load(f'{DATA_DIR}/lengths_train.npy'))
    t1_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/track1_val.npy'))
    t2_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/track2_val.npy'))
    lb_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/labels_val.npy'))
    ln_vl = torch.LongTensor(np.load(f'{DATA_DIR}/lengths_val.npy'))

    # lengths may be 1-D (same for both tracks) or 2-D  (len_a, len_b).
    if ln_tr.ndim == 1:
        # Both tracks share the length
        ds_tr = TensorDataset(t1_tr, t2_tr, ln_tr, ln_tr, lb_tr)
        ds_vl = TensorDataset(t1_vl, t2_vl, ln_vl, ln_vl, lb_vl)
    else:
        ds_tr = TensorDataset(t1_tr, t2_tr, ln_tr[:, 0], ln_tr[:, 1], lb_tr)
        ds_vl = TensorDataset(t1_vl, t2_vl, ln_vl[:, 0], ln_vl[:, 1], lb_vl)

    pos_w = torch.tensor([(lb_tr == 0).sum() / (lb_tr == 1).sum()])
    train_ld = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_ld   = DataLoader(ds_vl, batch_size=128,        shuffle=False,
                          num_workers=4, pin_memory=True)
    return train_ld, val_ld, pos_w


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, correct, total = 0., 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for t1, t2, la, lb, y in loader:
            t1, t2 = t1.to(device, non_blocking=True), t2.to(device, non_blocking=True)
            la, lb = la.to(device, non_blocking=True), lb.to(device, non_blocking=True)
            y      = y.to(device, non_blocking=True)
            out = model(t1, t2, la, lb)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += ((torch.sigmoid(out) > 0.5) == y.bool()).sum().item()
            total      += len(y)
    return total_loss / total, correct / total


def train_config(cfg, device):
    print(f"\n{'='*55}")
    print(f"  Config: {cfg['name']}  |  n_cnn={cfg['n_cnn']}  "
          f"n_mamba={cfg['n_mamba']}  dropout={cfg['dropout']}")
    print(f"{'='*55}")

    train_ld, val_ld, pos_w = load_data()
    model = AblationCNNMambaV2(
        n_cnn=cfg['n_cnn'], n_mamba=cfg['n_mamba'], dropout=cfg['dropout']
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    best_val_acc  = 0.0
    best_epoch    = 0
    patience_cnt  = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        tr_loss, _      = run_epoch(model, train_ld, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_ld,   criterion, optimizer, device, train=False)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        marker = ''
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_val_acc  = vl_acc
            best_epoch    = epoch + 1
            torch.save(model.state_dict(), f'{RESULTS_DIR}/{cfg["name"]}_best.pth')
            patience_cnt = 0
            marker = '  ✓'
        else:
            patience_cnt += 1

        print(f"  Epoch {epoch+1:02d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"val_acc={vl_acc:.4f}{marker}", flush=True)

        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    result = {
        'name':          cfg['name'],
        'n_cnn':         cfg['n_cnn'],
        'n_mamba':       cfg['n_mamba'],
        'dropout':       cfg['dropout'],
        'n_params':      n_params,
        'best_val_acc':  best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch':    best_epoch,
        'history':       history,
    }
    with open(f'{RESULTS_DIR}/{cfg["name"]}_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  >> Best val_acc={best_val_acc:.4f}  (epoch {best_epoch})")
    return result


# ====================================================================
# Launcher & plotting
# ====================================================================
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
        log_file  = f'/tmp/v2_gpu{gpu_id}.log'
        p = subprocess.Popen(
            ['bash', '-c', f'({shell_cmd}) > {log_file} 2>&1'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        procs.append((gpu_id, cfg_names, p, log_file))
        print(f'GPU {gpu_id}: {cfg_names}  -> {log_file}')

    print(f'\nLaunched {sum(len(v) for v in GPU_ASSIGN.values())} configs.')
    for gpu_id, cfg_names, p, _ in procs:
        p.wait()
        print(f'GPU {gpu_id} done: {cfg_names}')
    plot_compare()


def plot_compare():
    """Compare v1 (standard BiMamba) vs v2 (length-aware BiMamba)."""
    V1_DIR = '/home/yangcq/track_association/output/ablation'
    GRID   = '/home/yangcq/track_association/output/grid_search'

    # Same 4 matched points
    pairs = [
        ('n_cnn=1, drop=0.2',            f'{V1_DIR}/cnn1_result.json',           f'{RESULTS_DIR}/v2_c1_m1_d02_result.json'),
        ('n_cnn=3, drop=0.2',            f'{V1_DIR}/cnn3_result.json',           f'{RESULTS_DIR}/v2_c3_m1_d02_result.json'),
        ('n_cnn=3, m=2, drop=0.2',       f'{V1_DIR}/mamba2_result.json',         f'{RESULTS_DIR}/v2_c3_m2_d02_result.json'),
        ('n_cnn=4, drop=0.4 (best v1)',  f'{GRID}/g_cnn4_drop04_result.json',    f'{RESULTS_DIR}/v2_c4_m1_d04_result.json'),
        ('n_cnn=4, m=2, drop=0.4',       None,                                   f'{RESULTS_DIR}/v2_c4_m2_d04_result.json'),
        ('n_cnn=4, m=3, drop=0.4',       None,                                   f'{RESULTS_DIR}/v2_c4_m3_d04_result.json'),
    ]

    def load(p):
        if p and os.path.exists(p):
            return json.load(open(p))['best_val_acc'] * 100
        return None

    labels, v1_accs, v2_accs = [], [], []
    for name, p1, p2 in pairs:
        labels.append(name)
        v1_accs.append(load(p1))
        v2_accs.append(load(p2))

    # ---- Bar chart ----
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    w = 0.36
    v1_plot = [v if v is not None else 0 for v in v1_accs]
    v2_plot = [v if v is not None else 0 for v in v2_accs]
    b1 = ax.bar(x - w/2, v1_plot, w, label='v1 (Standard BiMamba)',
                color='#4472C4', edgecolor='black', linewidth=0.7)
    b2 = ax.bar(x + w/2, v2_plot, w, label='v2 (Length-aware BiMamba)',
                color='#ED7D31', edgecolor='black', linewidth=0.7)

    for bars, accs in [(b1, v1_accs), (b2, v2_accs)]:
        for bar, v in zip(bars, accs):
            if v is None:
                continue
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                    f'{v:.2f}%', ha='center', va='bottom', fontsize=9)

    # Mark N/A
    for i, v in enumerate(v1_accs):
        if v is None:
            ax.text(i - w/2, 0.5, 'N/A', ha='center', color='gray', fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Length-Aware BiMamba (v2) vs Standard BiMamba (v1)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(85, 96)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = f'{FIG_DIR}/v1_vs_v2_bimamba.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # ---- Print table ----
    print(f'\n{"Config":<32} {"v1":>10} {"v2":>10} {"Δ":>8}')
    print('-'*64)
    for name, a1, a2 in zip(labels, v1_accs, v2_accs):
        a1s = f'{a1:.2f}%' if a1 is not None else '  N/A '
        a2s = f'{a2:.2f}%' if a2 is not None else '  N/A '
        if a1 is not None and a2 is not None:
            delta = a2 - a1
            dsign = f'+{delta:.2f}' if delta >= 0 else f'{delta:.2f}'
        else:
            dsign = ''
        print(f'{name:<32} {a1s:>10} {a2s:>10} {dsign:>8}')


# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['launch', 'train', 'plot'], default='launch')
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.mode == 'launch':
        launch_all()
    elif args.mode == 'train':
        cfg = next((c for c in CONFIGS if c['name'] == args.config), None)
        if cfg is None:
            print(f'Unknown config: {args.config}')
            sys.exit(1)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_config(cfg, device)
    elif args.mode == 'plot':
        plot_compare()
