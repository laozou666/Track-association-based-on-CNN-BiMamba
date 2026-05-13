"""
Evaluate all 8 baseline models + CNN+Mamba+xattn on the validation set.
Runs on whichever GPU is idle (< 30% utilization), else on CPU.
"""

import os, sys, json, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- Pick the most idle GPU, else CPU ----
def pick_gpu():
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'], text=True).strip()
        best = None
        for line in out.splitlines():
            idx, util, mem = [int(x.strip()) for x in line.split(',')]
            if best is None or (util, mem) < (best[1], best[2]):
                best = (idx, util, mem)
        # Only use GPU if it's really idle
        if best and best[1] < 30:
            return f'cuda:{best[0]}', best
    except Exception:
        pass
    return 'cpu', None

device_str, gpu_info = pick_gpu()
if gpu_info:
    print(f'Using GPU {gpu_info[0]}  (util={gpu_info[1]}%  mem={gpu_info[2]}MB)')
else:
    print('Using CPU')
device = torch.device(device_str)

# ---- Data ----
DATA_DIR = '/home/yangcq/track_association/data/final_dataset'
t1_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/track1_val.npy'))
t2_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/track2_val.npy'))
lb_vl = torch.FloatTensor(np.load(f'{DATA_DIR}/labels_val.npy'))
print(f'Val set: {len(lb_vl)} samples')

loader = DataLoader(TensorDataset(t1_vl, t2_vl, lb_vl),
                    batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# ---- Model registry ----
MODEL_DIR = '/home/yangcq/track_association/output/model'

def build(name):
    if name == 'ann':
        from utils.model_ann  import TrajectoryMatcher_ANN
        return TrajectoryMatcher_ANN()
    if name == 'bigru':
        from utils.model_bigru import TrajectoryMatcher_BiGRU
        return TrajectoryMatcher_BiGRU()
    if name == 'bilstm':
        from utils.model_bilstm import TrajectoryMatcher_BiLSTM
        return TrajectoryMatcher_BiLSTM()
    if name == 'lstm':
        from utils.model_lstm import TrajectoryMatcher_LSTM
        return TrajectoryMatcher_LSTM()
    if name == 'cnn':
        from utils.model_cnn import TrajectoryMatcher_CNN
        return TrajectoryMatcher_CNN()
    if name == 'cnn_lstm':
        from utils.model_cnn_lstm import CNNTrajectoryMatcher_LSTM
        return CNNTrajectoryMatcher_LSTM()
    if name == 'cnn_bilstm':
        from utils.model_cnn_bilstm import CNNTrajectoryMatcher_BiLSTM
        return CNNTrajectoryMatcher_BiLSTM()
    if name == 'cnn_mamba':
        from utils.model_cnn_mamba import CNNTrajectoryMatcher
        return CNNTrajectoryMatcher()
    raise ValueError(name)

BASELINES = ['ann', 'bigru', 'bilstm', 'lstm',
             'cnn', 'cnn_lstm', 'cnn_bilstm', 'cnn_mamba']


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    for t1, t2, y in loader:
        t1, t2, y = t1.to(device), t2.to(device), y.to(device)
        out = model(t1, t2)
        # Some models return (B,), some (B,1); some output logits, some probabilities
        out = out.view(-1)
        prob = torch.sigmoid(out) if out.abs().max() > 1.5 else out
        # Clamp for safety
        prob = prob.clamp(0, 1)
        pred = (prob > 0.5).float()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


results = {}
for name in BASELINES:
    ckpt = f'{MODEL_DIR}/best_{name}.pth'
    if not os.path.exists(ckpt):
        print(f'  SKIP {name}: no checkpoint')
        continue
    try:
        model = build(name).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        acc = evaluate(model, loader, device)
        results[name] = acc
        print(f'  {name:<12}  val_acc = {acc*100:.2f}%')
    except Exception as e:
        print(f'  FAIL {name}: {e}')
        results[name] = None

# ---- Also load our best tuned models ----
tuned = {
    'CNN-Mamba (cnn4,m1,d0.4 grid)':
        '/home/yangcq/track_association/output/grid_search/g_cnn4_drop04_result.json',
    'CNN-Mamba (cnn3,m1,d0.2 abl)':
        '/home/yangcq/track_association/output/ablation/cnn3_result.json',
}
for label, p in tuned.items():
    if os.path.exists(p):
        r = json.load(open(p))
        results[label] = r['best_val_acc']

# xattn might not be done yet
xattn_p = '/home/yangcq/track_association/output/xattn/xattn_c4_m1_d04_result.json'
if os.path.exists(xattn_p):
    r = json.load(open(xattn_p))
    results['CNN-Mamba + Cross-Attn (cnn4,m1,d0.4)'] = r['best_val_acc']
else:
    print('\n  (CNN-Mamba + CrossAttn is still training)')

# ---- Print ranked leaderboard ----
print('\n' + '='*65)
print(f'  {"Rank":<5}{"Model":<45}{"Val Acc":>12}')
print('-'*65)
ranked = sorted([(k, v) for k, v in results.items() if v is not None],
                key=lambda t: t[1], reverse=True)
for i, (name, acc) in enumerate(ranked, 1):
    print(f'  {i:<5}{name:<45}{acc*100:>10.2f}%')
print('='*65)

# Save
with open('/home/yangcq/track_association/output/baseline_leaderboard.json', 'w') as f:
    json.dump({k: (v if v is None else float(v)) for k, v in results.items()},
              f, indent=2, ensure_ascii=False)
print('\nSaved: /home/yangcq/track_association/output/baseline_leaderboard.json')
