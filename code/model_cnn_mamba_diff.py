"""
CNN-Mamba with Differential (relative) input features.

Motivation
----------
The vanilla CNN-Mamba model encodes the two tracks independently and only
interacts them in the final classifier via [g_a, g_b, g_a - g_b, |g_a - g_b|].
Track association is physically about RELATIVE quantities at every timestep
(closing rate, relative heading, inter-track distance).  This model feeds
explicit cross-track differences into the encoder itself, so that the CNN
can directly extract physics-motivated features.

Input  : traj_a, traj_b   each (B, T, 4)
Per-stream input after augmentation :
         x_a = concat([traj_a, traj_b - traj_a], dim=-1)   -> (B, T, 8)
         x_b = concat([traj_b, traj_a - traj_b], dim=-1)   -> (B, T, 8)
Encoder: AblationCNNMamba (input_dim = 8, n_cnn = 4, n_mamba = 1, dropout = 0.4)
Fusion : [g_a, g_b, g_a - g_b, |g_a - g_b|]  (same as before)

Config
------
n_cnn    = 4       (grid-search optimum)
n_mamba  = 1       (layer-ablation optimum)
dropout  = 0.4     (grid-search optimum)

Usage
-----
python model_cnn_mamba_diff.py            # auto pick idle GPU + train
python model_cnn_mamba_diff.py --plot     # plot comparison
"""
import os, sys, json, time, argparse, subprocess
# ----- pick idle GPU before importing torch ----------
def pick_idle_gpu():
    try:
        q = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        rows = []
        for l in q:
            i, u, m = [x.strip() for x in l.split(',')]
            rows.append((int(i), int(u), int(m)))
        rows.sort(key=lambda r: (r[1], r[2]))
        print('GPU status:'); [print(f'   GPU {r[0]}: util={r[1]}%  mem={r[2]} MiB') for r in rows]
        print(f'Picked GPU: {rows[0][0]}')
        return rows[0][0]
    except Exception as e:
        print(f'nvidia-smi failed ({e}); fallback to GPU 0'); return 0

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true')
parser.add_argument('--gpu', type=int, default=None)
args = parser.parse_args()

if not args.plot:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(
        args.gpu if args.gpu is not None else pick_idle_gpu())

# -----------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_cnn_mamba as ab
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# -----------------------------------------------------
DATA_DIR = '/home/yangcq/track_association/data/final_dataset'
OUT_DIR  = '/home/yangcq/track_association/output/diff_input'
FIG_DIR  = '/home/yangcq/track_association/output/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------
# Config (mirrors grid-search best)
# -----------------------------------------------------
N_CNN       = 4
N_MAMBA     = 1
DROPOUT     = 0.4
EPOCHS      = ab.EPOCHS
BATCH_SIZE  = ab.BATCH_SIZE
LR          = ab.LR
PATIENCE    = ab.PATIENCE


# -----------------------------------------------------
# Model wrapper: concat diff features at input, then call AblationCNNMamba
# -----------------------------------------------------
class CNNMambaDiff(nn.Module):
    def __init__(self, n_cnn=N_CNN, n_mamba=N_MAMBA, dropout=DROPOUT):
        super().__init__()
        self.core = ab.AblationCNNMamba(
            input_dim=8, n_cnn=n_cnn, n_mamba=n_mamba, dropout=dropout)

    def forward(self, traj_a, traj_b):
        diff_ab = traj_b - traj_a                 # (B, T, 4)
        diff_ba = traj_a - traj_b                 # (B, T, 4)
        x_a = torch.cat([traj_a, diff_ab], dim=-1)  # (B, T, 8)
        x_b = torch.cat([traj_b, diff_ba], dim=-1)  # (B, T, 8)
        return self.core(x_a, x_b)


# -----------------------------------------------------
# Training
# -----------------------------------------------------
def train_main(device):
    print('='*60)
    print(f'CNN-Mamba + Differential Input')
    print(f'  n_cnn={N_CNN}  n_mamba={N_MAMBA}  dropout={DROPOUT}')
    print('='*60)

    train_ld, val_ld, pos_w = ab.load_data()
    model = CNNMambaDiff().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {n_params:,}')

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf'); patience_cnt = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        tr_loss, _ = ab.run_epoch(model, train_ld, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = ab.run_epoch(model, val_ld, criterion, optimizer, device, train=False)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        marker = ''
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss; best_val_acc = vl_acc; best_epoch = epoch+1
            torch.save(model.state_dict(), f'{OUT_DIR}/diff_best.pth')
            patience_cnt = 0; marker = '  ✓'
        else:
            patience_cnt += 1
        print(f'  Epoch {epoch+1:02d}/{EPOCHS}  '
              f'train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  '
              f'val_acc={vl_acc:.4f}{marker}', flush=True)
        if patience_cnt >= PATIENCE:
            print(f'  Early stop at epoch {epoch+1}'); break

    result = {
        'name': 'cnn_mamba_diff',
        'n_cnn': N_CNN, 'n_mamba': N_MAMBA, 'dropout': DROPOUT,
        'n_params': n_params,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'history': history,
    }
    with open(f'{OUT_DIR}/diff_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {OUT_DIR}/diff_result.json')
    print(f'>> Best val_acc={best_val_acc:.4f}  (epoch {best_epoch})')


# -----------------------------------------------------
# Plot: compare with CNN-BiLSTM, CNN-LSTM, CNN-Mamba tuned
# -----------------------------------------------------
def plot_compare():
    with open(f'{OUT_DIR}/diff_result.json') as f:
        diff_r = json.load(f)
    diff_acc = diff_r['best_val_acc'] * 100

    entries = [
        ('CNN-Mamba orig',        88.67, '#aaaaaa'),
        ('CNN-LSTM',              91.42, '#1f77b4'),
        ('CNN-BiLSTM',            91.87, '#1f77b4'),
        ('CNN-Mamba tuned',       92.38, '#ff7b00'),
        ('CNN-Mamba + Diff',      diff_acc, '#d62728'),
    ]
    entries.sort(key=lambda x: x[1])
    names  = [e[0] for e in entries]
    accs   = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    plt.figure(figsize=(9.5, 5.5))
    bars = plt.barh(names, accs, color=colors, edgecolor='black', alpha=0.88)
    for b, a in zip(bars, accs):
        plt.text(a + 0.1, b.get_y() + b.get_height()/2,
                 f'{a:.2f}%', va='center', fontsize=11)
    plt.xlabel('Validation Accuracy (%)', fontsize=12)
    plt.title('CNN-Mamba with Differential Input  vs  Top Baselines',
              fontsize=13, fontweight='bold')
    plt.xlim(min(accs) - 1, max(accs) + 1.5)
    plt.grid(True, axis='x', alpha=0.35, linestyle='--')
    plt.tight_layout()
    out = f'{FIG_DIR}/diff_input_compare.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# -----------------------------------------------------
if __name__ == '__main__':
    if args.plot:
        plot_compare()
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'device: {device}')
        t0 = time.time()
        train_main(device)
        print(f'Total: {time.time()-t0:.1f}s')
        plot_compare()
