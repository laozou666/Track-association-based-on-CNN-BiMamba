"""
Evaluate every trained model on the held-out **test set** of final_dataset.

Models included
---------------
Baselines (output/model/best_<name>.pth):
    ANN, BiGRU, BiLSTM, LSTM, CNN, CNN-LSTM, CNN-BiLSTM, CNN-BiMamba (untuned)

Tuned CNN-BiMamba (n_cnn=4, n_mamba=1, dropout=0.4):
    output/grid_search/g_cnn4_drop04_best.pth

Direction ablation (same hyper-params as tuned, but uni-directional):
    Forward Mamba   -> output/direction_ablation/mamba_fwd_best.pth
    Backward Mamba  -> output/direction_ablation/mamba_bwd_best.pth

Pure BiMamba (no CNN frontend, n_cnn=0, n_mamba=1, dropout=0.4):
    output/pure_bimamba/pure_bimamba_d04_best.pth

Outputs
-------
output/test_eval/
    test_metrics.csv     (accuracy, precision, recall, f1, auc, n_params per model)
    test_metrics.json
    bar_test_acc.png     (single-metric accuracy bar chart, sorted desc)
    bar_test_metrics.png (4-metric grouped bar chart)
"""
import os, sys, json, subprocess, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

DATA_DIR  = '/home/yangcq/track_association/data/final_dataset'
MODEL_DIR = '/home/yangcq/track_association/output/model'
OUT_DIR   = '/home/yangcq/track_association/output/test_eval'
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------- pick least-busy GPU ---------------------
def pick_idle_gpu():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits']).decode().strip().splitlines()
        return int(np.argmin([int(x) for x in out]))
    except Exception:
        return 0

GPU = pick_idle_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
print(f'[gpu] physical GPU {GPU} -> cuda:0', flush=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ---------------------- load test set --------------------------
t1 = torch.FloatTensor(np.load(f'{DATA_DIR}/track1_test.npy'))
t2 = torch.FloatTensor(np.load(f'{DATA_DIR}/track2_test.npy'))
yt = torch.FloatTensor(np.load(f'{DATA_DIR}/labels_test.npy'))
print(f'[data] test set: {len(yt)} samples, '
      f'pos_ratio={float(yt.mean()):.4f}', flush=True)

loader = DataLoader(TensorDataset(t1, t2, yt),
                    batch_size=128, shuffle=False,
                    num_workers=2, pin_memory=True)


# ---------------------- evaluation helper ----------------------
@torch.no_grad()
def evaluate(model):
    model.eval()
    probs, labels = [], []
    for a, b, y in loader:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        out = model(a, b).view(-1)
        # Some heads return logits; treat anything outside [0,1] as logits.
        if out.min() < 0 or out.max() > 1:
            p = torch.sigmoid(out)
        else:
            p = out
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    p = np.concatenate(probs); y = np.concatenate(labels)
    pred = (p > 0.5).astype(np.float32)
    return {
        'accuracy':  accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall':    recall_score(y, pred, zero_division=0),
        'f1':        f1_score(y, pred, zero_division=0),
        'auc':       roc_auc_score(y, p),
    }


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


# ---------------------- builders -------------------------------
def build_baseline(name):
    if name == 'ann':
        from utils.model_ann  import TrajectoryMatcher_ANN; return TrajectoryMatcher_ANN()
    if name == 'bigru':
        from utils.model_bigru import TrajectoryMatcher_BiGRU; return TrajectoryMatcher_BiGRU()
    if name == 'bilstm':
        from utils.model_bilstm import TrajectoryMatcher_BiLSTM; return TrajectoryMatcher_BiLSTM()
    if name == 'lstm':
        from utils.model_lstm import TrajectoryMatcher_LSTM; return TrajectoryMatcher_LSTM()
    if name == 'cnn':
        from utils.model_cnn import TrajectoryMatcher_CNN; return TrajectoryMatcher_CNN()
    if name == 'cnn_lstm':
        from utils.model_cnn_lstm import CNNTrajectoryMatcher_LSTM; return CNNTrajectoryMatcher_LSTM()
    if name == 'cnn_bilstm':
        from utils.model_cnn_bilstm import CNNTrajectoryMatcher_BiLSTM; return CNNTrajectoryMatcher_BiLSTM()
    if name == 'cnn_mamba':
        from utils.model_cnn_mamba import CNNTrajectoryMatcher; return CNNTrajectoryMatcher()
    raise ValueError(name)


# tuned CNN-BiMamba & pure BiMamba: same class with n_cnn switched
def build_ablation(n_cnn, n_mamba=1, dropout=0.4):
    import ablation_cnn_mamba as ab
    return ab.AblationCNNMamba(n_cnn=n_cnn, n_mamba=n_mamba, dropout=dropout)


def build_directional(direction):
    import compare_direction as cd
    return cd.DirectionalCNNMamba(direction=direction)


# ---------------------- model registry -------------------------
JOBS = [
    # (display name, builder, ckpt path)
    ('ANN',         lambda: build_baseline('ann'),         f'{MODEL_DIR}/best_ann.pth'),
    ('LSTM',        lambda: build_baseline('lstm'),        f'{MODEL_DIR}/best_lstm.pth'),
    ('BiLSTM',      lambda: build_baseline('bilstm'),      f'{MODEL_DIR}/best_bilstm.pth'),
    ('BiGRU',       lambda: build_baseline('bigru'),       f'{MODEL_DIR}/best_bigru.pth'),
    ('CNN',         lambda: build_baseline('cnn'),         f'{MODEL_DIR}/best_cnn.pth'),
    ('CNN-LSTM',    lambda: build_baseline('cnn_lstm'),    f'{MODEL_DIR}/best_cnn_lstm.pth'),
    ('CNN-BiLSTM',  lambda: build_baseline('cnn_bilstm'),  f'{MODEL_DIR}/best_cnn_bilstm.pth'),
    ('CNN-BiMamba (untuned)',
                    lambda: build_baseline('cnn_mamba'),   f'{MODEL_DIR}/best_cnn_mamba.pth'),
    ('CNN-BiMamba (tuned)',
                    lambda: build_ablation(n_cnn=4),
                    '/home/yangcq/track_association/output/grid_search/g_cnn4_drop04_best.pth'),
    ('CNN-Mamba (forward)',
                    lambda: build_directional('fwd'),
                    '/home/yangcq/track_association/output/direction_ablation/mamba_fwd_best.pth'),
    ('CNN-Mamba (backward)',
                    lambda: build_directional('bwd'),
                    '/home/yangcq/track_association/output/direction_ablation/mamba_bwd_best.pth'),
    ('Pure BiMamba',
                    lambda: build_ablation(n_cnn=0),
                    '/home/yangcq/track_association/output/pure_bimamba/pure_bimamba_d04_best.pth'),
]


# ---------------------- run ------------------------------------
results = []
for label, builder, ckpt in JOBS:
    if not os.path.exists(ckpt):
        print(f'[skip] {label}: ckpt missing -> {ckpt}', flush=True)
        continue
    print(f'\n[eval] {label}', flush=True)
    try:
        m = builder().to(device)
        sd = torch.load(ckpt, map_location=device)
        m.load_state_dict(sd)
        t0 = time.time()
        metrics = evaluate(m)
        elapsed = time.time() - t0
        params = n_params(m)
        row = {'model': label, 'n_params': params,
               'eval_seconds': round(elapsed, 2), **metrics}
        results.append(row)
        print(f'  acc={metrics["accuracy"]*100:.2f}%  '
              f'f1={metrics["f1"]*100:.2f}%  '
              f'auc={metrics["auc"]*100:.2f}%  '
              f'params={params/1e3:.1f}K  '
              f'time={elapsed:.1f}s', flush=True)
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  [FAIL] {label}: {e}', flush=True)


# ---------------------- save tables ----------------------------
csv_path = f'{OUT_DIR}/test_metrics.csv'
with open(csv_path, 'w') as f:
    f.write('model,n_params,eval_seconds,accuracy,precision,recall,f1,auc\n')
    for r in results:
        f.write(f"{r['model']},{r['n_params']},{r['eval_seconds']},"
                f"{r['accuracy']:.6f},{r['precision']:.6f},"
                f"{r['recall']:.6f},{r['f1']:.6f},{r['auc']:.6f}\n")

with open(f'{OUT_DIR}/test_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print('\n[summary] (sorted by accuracy)')
for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
    print(f'  {r["model"]:<28}  '
          f'acc={r["accuracy"]*100:6.2f}%  '
          f'f1={r["f1"]*100:6.2f}%  '
          f'auc={r["auc"]*100:6.2f}%  '
          f'params={r["n_params"]/1e3:6.1f}K')


# ---------------------- bar chart ------------------------------
def color_for(name):
    n = name.lower()
    if 'untuned' in n:                            return '#9E480E'   # dark orange-brown for untuned baseline
    if 'tuned'   in n:                            return '#C00000'   # highlight: best CNN-BiMamba tuned
    if 'forward' in n or 'backward' in n:         return '#ED7D31'   # orange family for direction ablation
    if 'pure'    in n:                            return '#70AD47'   # green for pure BiMamba
    if 'cnn'     in n:                            return '#4472C4'   # blue for CNN-* baselines
    return '#7F7F7F'                                                 # grey for plain RNN/ANN

# sort by accuracy descending
ranked = sorted(results, key=lambda x: x['accuracy'], reverse=True)
names = [r['model'] for r in ranked]
accs  = [r['accuracy'] * 100 for r in ranked]
colors = [color_for(n) for n in names]

# 1) accuracy-only bar
fig, ax = plt.subplots(figsize=(11.5, 6))
bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=0.6, width=0.65)
ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Test-set accuracy across all trained models', fontsize=13)
y_min = max(0.0, min(accs) - 5)
ax.set_ylim(y_min, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(rotation=25, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/bar_test_acc.png', dpi=160)
plt.close()

# 2) grouped bar: accuracy / precision / recall / f1
metric_keys = ['accuracy', 'precision', 'recall', 'f1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(ranked))
width = 0.2

fig, ax = plt.subplots(figsize=(13.5, 6))
for i, (k, mn) in enumerate(zip(metric_keys, metric_names)):
    vals = [r[k] * 100 for r in ranked]
    ax.bar(x + (i - 1.5) * width, vals, width, label=mn, edgecolor='black', linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Test-set metrics across all trained models', fontsize=13)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='lower left', ncol=4, fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/bar_test_metrics.png', dpi=160)
plt.close()

print(f'\n[saved] {csv_path}')
print(f'[saved] {OUT_DIR}/test_metrics.json')
print(f'[saved] {OUT_DIR}/bar_test_acc.png')
print(f'[saved] {OUT_DIR}/bar_test_metrics.png')
