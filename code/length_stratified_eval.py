"""
Evaluate top models on SHORT / MEDIUM / LONG test sets.

Purpose: show how each model's accuracy scales with track length,
supporting the "Mamba scales better on long sequences" claim.

Models:
    CNN-Mamba tuned   (n_cnn=4, n_mamba=1, dropout=0.4)  - our best
    CNN-BiLSTM
    CNN-LSTM

Test sets:
    test_short   L in [21, 79]     N = 9836
    test_medium  L in [80, 180]    N = 7239
    test_long    L in [181, 350]   N = 2282

Outputs:
    output/length_eval/test_length_eval.json
    output/figures/test_length_eval.png   (two-panel: acc curve + gap bar)
"""
import os, sys, json, subprocess
import numpy as np


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
        print(f'Picked GPU: {rows[0][0]}')
        return rows[0][0]
    except Exception:
        return 0


if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(pick_idle_gpu())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

DATA_DIR  = '/home/yangcq/track_association/data/final_dataset'
MODEL_DIR = '/home/yangcq/track_association/output/model'
ABL_DIR   = '/home/yangcq/track_association/output/grid_search'
OUT_DIR   = '/home/yangcq/track_association/output/length_eval'
FIG_DIR   = '/home/yangcq/track_association/output/figures'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


# -----------------------------------------------------
def build_and_load(name):
    if name == 'cnn_mamba_tuned':
        from ablation_cnn_mamba import AblationCNNMamba
        m = AblationCNNMamba(input_dim=4, n_cnn=4, n_mamba=1, dropout=0.4)
        ckpt = f'{ABL_DIR}/g_cnn4_drop04_best.pth'
    elif name == 'cnn_bilstm':
        from utils.model_cnn_bilstm import CNNTrajectoryMatcher_BiLSTM
        m = CNNTrajectoryMatcher_BiLSTM()
        ckpt = f'{MODEL_DIR}/best_cnn_bilstm.pth'
    elif name == 'cnn_lstm':
        from utils.model_cnn_lstm import CNNTrajectoryMatcher_LSTM
        m = CNNTrajectoryMatcher_LSTM()
        ckpt = f'{MODEL_DIR}/best_cnn_lstm.pth'
    else:
        raise ValueError(name)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd)
    m.to(device).eval()
    return m, ckpt


def infer_all(model, t1, t2, batch_size=128):
    ds = TensorDataset(t1, t2)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for a, b in ld:
            a = a.to(device); b = b.to(device)
            out = model(a, b)
            if out.dim() > 1:
                out = out.squeeze(-1)
            if out.min() < 0 or out.max() > 1.0:
                p = torch.sigmoid(out)
            else:
                p = out
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


# -----------------------------------------------------
def load_split(kind):
    """kind in {short, medium, long}"""
    t1 = torch.FloatTensor(np.load(f'{DATA_DIR}/track1_test_{kind}.npy'))
    t2 = torch.FloatTensor(np.load(f'{DATA_DIR}/track2_test_{kind}.npy'))
    y  = np.load(f'{DATA_DIR}/labels_test_{kind}.npy').astype(int)
    L  = np.load(f'{DATA_DIR}/lengths_test_{kind}.npy')
    return t1, t2, y, L


# -----------------------------------------------------
def main():
    splits = ['short', 'medium', 'long']
    models = [
        ('CNN-Mamba tuned', 'cnn_mamba_tuned',  '#d62728'),
        ('CNN-BiLSTM',      'cnn_bilstm',       '#1f77b4'),
        ('CNN-LSTM',        'cnn_lstm',         '#2ca02c'),
    ]

    print('Loading all test splits ...')
    splits_data = {}
    for k in splits:
        t1, t2, y, L = load_split(k)
        splits_data[k] = (t1, t2, y, L)
        print(f'  {k:<7}  N={len(y):>5}  lengths mean={L.mean():.1f}  '
              f'range=[{L.min()}, {L.max()}]')

    results = {}   # results[model_name][split] = {acc, prec, rec, f1, n}
    for label, name, _ in models:
        try:
            m, ckpt = build_and_load(name)
        except Exception as e:
            print(f'[SKIP] {label}: {e}'); continue
        print(f'\nEvaluating {label}  ({ckpt})')
        results[name] = {}
        for k in splits:
            t1, t2, y, L = splits_data[k]
            probs = infer_all(m, t1, t2)
            preds = (probs >= 0.5).astype(int)
            acc  = accuracy_score(y, preds)
            prec = precision_score(y, preds, zero_division=0)
            rec  = recall_score(y, preds, zero_division=0)
            f1   = f1_score(y, preds, zero_division=0)
            results[name][k] = {
                'n':   int(len(y)),
                'acc': float(acc),
                'prec': float(prec),
                'rec': float(rec),
                'f1':  float(f1),
            }
            print(f'  {k:<7}  acc={acc*100:5.2f}%  prec={prec*100:5.2f}%  '
                  f'rec={rec*100:5.2f}%  f1={f1*100:5.2f}%')
        del m
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    with open(f'{OUT_DIR}/test_length_eval.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_DIR}/test_length_eval.json')

    # ---------- PLOT ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.5))

    # left: accuracy per split per model
    x_pos = np.arange(len(splits))
    for label, name, color in models:
        if name not in results:
            continue
        accs = [results[name][k]['acc'] * 100 for k in splits]
        ax1.plot(x_pos, accs, 'o-', color=color, linewidth=2.3,
                 markersize=10, label=label)
        for i, a in enumerate(accs):
            ax1.annotate(f'{a:.2f}%', (x_pos[i], a), xytext=(0, 8),
                         textcoords='offset points', ha='center',
                         fontsize=10, color=color, fontweight='bold')

    for i, k in enumerate(splits):
        t1, t2, y, L = splits_data[k]
        ax1.text(i, ax1.get_ylim()[0], f'N={len(y)}\nL={L.mean():.0f}',
                 ha='center', fontsize=8, color='#555',
                 transform=ax1.get_xaxis_transform())

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [f'Short\n[21,79]', f'Medium\n[80,180]', f'Long\n[181,350]'])
    ax1.set_xlabel('Test Subset by Track Length', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy vs Track Length',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.35, linestyle='--')
    ax1.legend(loc='best', fontsize=11)

    # right: CNN-Mamba − CNN-BiLSTM gap per split
    if 'cnn_mamba_tuned' in results and 'cnn_bilstm' in results:
        mamba_accs  = [results['cnn_mamba_tuned'][k]['acc'] for k in splits]
        bilstm_accs = [results['cnn_bilstm'][k]['acc']      for k in splits]
        gaps = [(m - b) * 100 for m, b in zip(mamba_accs, bilstm_accs)]
        colors = ['#d62728' if g > 0 else '#888' for g in gaps]
        bars = ax2.bar(x_pos, gaps, color=colors, edgecolor='black', alpha=0.85)
        for bar, g in zip(bars, gaps):
            h = bar.get_height()
            y_text = h + (0.05 if h >= 0 else -0.15)
            ax2.text(bar.get_x() + bar.get_width()/2, y_text,
                     f'{g:+.2f}', ha='center',
                     fontsize=11, fontweight='bold',
                     color='#d62728' if g > 0 else '#555')
        ax2.axhline(0, color='k', linewidth=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Short', 'Medium', 'Long'])
        ax2.set_xlabel('Test Subset', fontsize=12)
        ax2.set_ylabel('Accuracy Advantage (pp)', fontsize=12)
        ax2.set_title('CNN-Mamba tuned  −  CNN-BiLSTM',
                      fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.35, axis='y', linestyle='--')

    plt.tight_layout()
    out = f'{FIG_DIR}/test_length_eval.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # summary table
    print('\n============================================================')
    print(f'{"Model":<18}{"Short":>12}{"Medium":>12}{"Long":>12}')
    print('-' * 54)
    for label, name, _ in models:
        if name not in results:
            continue
        row = [f'{label:<18}']
        for k in splits:
            a = results[name][k]['acc'] * 100
            row.append(f'{a:>10.2f}%  ')
        print(''.join(row))
    print('============================================================')


if __name__ == '__main__':
    main()
