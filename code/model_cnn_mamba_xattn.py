"""
CNN + BiMamba + Cross-Attention model for track association.

Architecture:
    Track A: 4×CNN -> 1×BiMamba(v1 original) -> features_A  (B, T, d_model)
    Track B: 4×CNN -> 1×BiMamba(v1 original) -> features_B  (B, T, d_model)
               |                                       |
               +---- Cross-Attention (A<->B) ----------+
                         (with padding mask)
                                |
                mean+max pool (v1 style, no mask)
                                |
            concat[a, b, a-b, |a-b|]  ->  MLP  ->  logit

Design choices:
  - BiMamba stays in *v1* form (standard flip-based, no length-aware).
    This preserves the "length-as-signal" information that helps on this
    dataset (shown to be relevant by the data-quality diagnostics).
  - Cross-Attention uses *key_padding_mask* to ignore padded frames on the
    key/value side.  (Attention supports this natively, unlike Mamba.)
  - Pooling remains the v1 unmasked mean+max to keep the length signal.

Fixed config (as requested by user):
    n_cnn = 4, n_mamba = 1, dropout = 0.4
"""

import os, json, argparse, time
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
OUT_DIR     = '/home/yangcq/track_association/output/xattn'
FIG_DIR     = '/home/yangcq/track_association/output/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Hyper-params ----
N_CNN      = 4
N_MAMBA    = 1
DROPOUT    = 0.4
N_HEADS    = 4
EPOCHS     = 25
BATCH_SIZE = 64
LR         = 2e-4
D_MODEL    = 64
D_STATE    = 32
D_CONV     = 4
EXPAND     = 2
PATIENCE   = 7
NAME       = 'xattn_c4_m1_d04'


# =============================================================
# CNN blocks (same as v1)
# =============================================================
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 5, padding=2)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.proj  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x   = x.permute(0, 2, 1)
        res = self.proj(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = F.gelu(self.bn2(self.conv2(out)) + res)
        return out.permute(0, 2, 1)


# =============================================================
# Standard v1 BiMamba (flip-based, no length awareness)
# =============================================================
class BiMambaV1(nn.Module):
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


# =============================================================
# Cross-Attention layer (A <-> B, with padding mask)
# =============================================================
class CrossAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention:
        A_out = A + attn(Q=A, K=B, V=B, mask=pad_B)
        B_out = B + attn(Q=B, K=A, V=A, mask=pad_A)
    Followed by a small FFN per branch.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_a1  = nn.LayerNorm(d_model)
        self.norm_b1  = nn.LayerNorm(d_model)
        self.attn_ab  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_ba  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm_a2  = nn.LayerNorm(d_model)
        self.norm_b2  = nn.LayerNorm(d_model)
        self.ffn_a    = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_model * 2, d_model))
        self.ffn_b    = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_model * 2, d_model))
        self.drop     = nn.Dropout(dropout)

    def forward(self, a, b, pad_mask_a, pad_mask_b):
        """
        a, b: (B, T, D)
        pad_mask_*: (B, T) bool, True for *padded* positions (PyTorch convention)
        """
        qa = self.norm_a1(a)
        qb = self.norm_b1(b)
        a2, _ = self.attn_ab(qa, qb, qb, key_padding_mask=pad_mask_b, need_weights=False)
        b2, _ = self.attn_ba(qb, qa, qa, key_padding_mask=pad_mask_a, need_weights=False)
        a = a + self.drop(a2)
        b = b + self.drop(b2)
        a = a + self.drop(self.ffn_a(self.norm_a2(a)))
        b = b + self.drop(self.ffn_b(self.norm_b2(b)))
        return a, b


# =============================================================
# Full model
# =============================================================
class CNNMambaXAttn(nn.Module):
    def __init__(self,
                 input_dim=4, d_model=D_MODEL,
                 n_cnn=N_CNN, n_mamba=N_MAMBA,
                 d_state=D_STATE, d_conv=D_CONV, expand=EXPAND,
                 dropout=DROPOUT, n_heads=N_HEADS):
        super().__init__()
        # CNN stack
        layers = [CNNBlock(input_dim, d_model, dropout * 0.5)]
        for _ in range(n_cnn - 1):
            layers.append(CNNBlock(d_model, d_model, dropout * 0.5))
        self.cnn = nn.Sequential(*layers)

        # BiMamba (v1 standard)
        self.mamba_layers = nn.ModuleList([
            BiMambaV1(d_model, d_state, d_conv, expand, dropout * 0.5)
            for _ in range(n_mamba)
        ])

        # Cross-attention
        self.xattn = CrossAttentionLayer(d_model, n_heads, dropout=dropout * 0.5)

        # Classifier
        enc_dim    = d_model * 2           # mean + max pool
        fusion_dim = enc_dim * 4           # [a, b, a-b, |a-b|]
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

    def _encode_stream(self, x):
        h = self.cnn(x)
        for m in self.mamba_layers:
            h = m(h)
        return h

    @staticmethod
    def _pad_mask(x):
        # Detect padding by: a frame is "pad" iff all 4 raw features are 0.
        # Since x here is AFTER the CNN, we can't do this — we need the raw input.
        # So we compute the mask in forward() from the raw input, not here.
        raise NotImplementedError

    def forward(self, ra, rb):
        """
        ra, rb: raw (B, T, input_dim). Padding frames are exactly zero.
        """
        # Build padding masks from raw input (True == pad)
        with torch.no_grad():
            pad_a = (ra.abs().sum(dim=-1) == 0)   # (B, T) bool
            pad_b = (rb.abs().sum(dim=-1) == 0)

        fa = self._encode_stream(ra)   # (B, T, D)
        fb = self._encode_stream(rb)

        fa, fb = self.xattn(fa, fb, pad_a, pad_b)

        # v1-style unmasked pool (keeps length signal)
        g_a = torch.cat([fa.mean(dim=1), fa.max(dim=1).values], dim=-1)
        g_b = torch.cat([fb.mean(dim=1), fb.max(dim=1).values], dim=-1)
        comb = torch.cat([g_a, g_b, g_a - g_b, torch.abs(g_a - g_b)], dim=-1)
        return self.classifier(comb).squeeze(-1)


# =============================================================
# Train / eval
# =============================================================
def load_data():
    t1_tr = torch.FloatTensor(np.load(f'{DATA_DIR}/track1_train.npy'))
    t2_tr = torch.FloatTensor(np.load(f'{DATA_DIR}/track2_train.npy'))
    lb_tr = torch.FloatTensor(np.load(f'{DATA_DIR}/labels_train.npy'))
    t1_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/track1_val.npy'))
    t2_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/track2_val.npy'))
    lb_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/labels_val.npy'))
    pos_w = torch.tensor([(lb_tr == 0).sum() / (lb_tr == 1).sum()])
    train_ld = DataLoader(TensorDataset(t1_tr, t2_tr, lb_tr),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_ld   = DataLoader(TensorDataset(t1_vl, t2_vl, lb_vl),
                          batch_size=128, shuffle=False,
                          num_workers=4, pin_memory=True)
    return train_ld, val_ld, pos_w


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, correct, total = 0., 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for t1, t2, y in loader:
            t1, t2, y = t1.to(device, non_blocking=True), t2.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(t1, t2)
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


def train_main(device):
    print(f"Training CNN+BiMamba+CrossAttention ({NAME})")
    print(f"  n_cnn={N_CNN}, n_mamba={N_MAMBA}, dropout={DROPOUT}, n_heads={N_HEADS}")

    train_ld, val_ld, pos_w = load_data()
    model = CNNMambaXAttn().to(device)
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

    t0 = time.time()
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
            torch.save(model.state_dict(), f'{OUT_DIR}/{NAME}_best.pth')
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

    elapsed = time.time() - t0
    result = {
        'name':          NAME,
        'n_cnn':         N_CNN,
        'n_mamba':       N_MAMBA,
        'dropout':       DROPOUT,
        'n_heads':       N_HEADS,
        'n_params':      n_params,
        'best_val_acc':  best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch':    best_epoch,
        'elapsed_sec':   elapsed,
        'history':       history,
    }
    with open(f'{OUT_DIR}/{NAME}_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nDone.  Best val_acc={best_val_acc*100:.2f}%  (epoch {best_epoch})  "
          f"elapsed={elapsed:.0f}s")


# =============================================================
# Comparison plot
# =============================================================
def plot_compare():
    # Baselines to compare against
    baselines = {
        'BiLSTM':            None,    # fill if available
        'BiGRU':             None,
        'CNN-BiLSTM':        None,
        'CNN-Mamba (v1 original)': '/home/yangcq/track_association/output/grid_search/g_cnn4_drop04_result.json',
        'CNN-Mamba (v1 cnn3)':    '/home/yangcq/track_association/output/ablation/cnn3_result.json',
        'CNN-Mamba + CrossAttn':  f'{OUT_DIR}/{NAME}_result.json',
    }

    entries = []
    for label, path in baselines.items():
        if path and os.path.exists(path):
            r = json.load(open(path))
            entries.append((label, r['best_val_acc'] * 100, r.get('n_params', 0)))
    # sort by acc
    entries.sort(key=lambda t: t[1])

    labels   = [e[0] for e in entries]
    accs     = [e[1] for e in entries]
    params_M = [e[2] / 1000 for e in entries]   # K-params

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4472C4'] * len(entries)
    # highlight ours
    for i, l in enumerate(labels):
        if 'CrossAttn' in l:
            colors[i] = '#ED7D31'
    bars = ax.barh(labels, accs, color=colors, edgecolor='black', linewidth=0.7)
    for bar, a, p in zip(bars, accs, params_M):
        ax.text(a + 0.05, bar.get_y() + bar.get_height()/2,
                f'{a:.2f}%  ({p:.0f}K)',
                va='center', fontsize=10)
    ax.set_xlabel('Validation Accuracy (%)')
    ax.set_title('CNN + BiMamba + Cross-Attention vs CNN+Mamba (v1)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(88, max(accs) + 1.0)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = f'{FIG_DIR}/xattn_vs_v1.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    print('\n  Comparison:')
    for l, a, p in entries:
        print(f'    {l:<35} {a:.2f}%  ({p:.0f}K params)')


# =============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'plot'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_main(device)
        plot_compare()
    else:
        plot_compare()
