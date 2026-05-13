"""
Benchmark parameter count and inference speed for all models.

Reports:
  - trainable parameter count
  - per-batch forward time on GPU (batch=64)  averaged over 50 runs, warmup 10
  - per-sample inference time in ms (implicitly)
  - approximate FLOPs via thop (if available, else skipped)

Covers:
  ANN, CNN, LSTM, BiGRU, BiLSTM, CNN-LSTM, CNN-BiLSTM,
  CNN-Mamba (orig), CNN-Mamba tuned (c4_m1_d04), CNN-Mamba + Diff (if trained)

Saves to output/benchmark/benchmark.json + figure bench_comparison.png
"""
import os, sys, json, time, subprocess
import numpy as np
import torch

# pick idle GPU (but only briefly)
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
        return rows[0][0]
    except Exception:
        return 0


if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(pick_idle_gpu())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

OUT_DIR = '/home/yangcq/track_association/output/benchmark'
FIG_DIR = '/home/yangcq/track_association/output/figures'
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 350
FEAT    = 4
BATCH   = 64
WARMUP  = 10
ITERS   = 50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


# -----------------------------------------------------
# Model builders
# -----------------------------------------------------
def build_ann():
    from utils.model_ann import TrajectoryMatcher_ANN; return TrajectoryMatcher_ANN()
def build_cnn():
    from utils.model_cnn import TrajectoryMatcher_CNN; return TrajectoryMatcher_CNN()
def build_lstm():
    from utils.model_lstm import TrajectoryMatcher_LSTM; return TrajectoryMatcher_LSTM()
def build_bigru():
    from utils.model_bigru import TrajectoryMatcher_BiGRU; return TrajectoryMatcher_BiGRU()
def build_bilstm():
    from utils.model_bilstm import TrajectoryMatcher_BiLSTM; return TrajectoryMatcher_BiLSTM()
def build_cnn_lstm():
    from utils.model_cnn_lstm import CNNTrajectoryMatcher_LSTM; return CNNTrajectoryMatcher_LSTM()
def build_cnn_bilstm():
    from utils.model_cnn_bilstm import CNNTrajectoryMatcher_BiLSTM; return CNNTrajectoryMatcher_BiLSTM()
def build_cnn_mamba():
    from utils.model_cnn_mamba import CNNTrajectoryMatcher; return CNNTrajectoryMatcher()
def build_cnn_mamba_tuned():
    from ablation_cnn_mamba import AblationCNNMamba
    return AblationCNNMamba(input_dim=4, n_cnn=4, n_mamba=1, dropout=0.4)
def build_cnn_mamba_diff():
    from model_cnn_mamba_diff import CNNMambaDiff
    return CNNMambaDiff()


MODELS = [
    ('ANN',              build_ann,              'deep'),
    ('CNN',              build_cnn,              'deep'),
    ('LSTM',             build_lstm,             'deep'),
    ('BiGRU',            build_bigru,            'deep'),
    ('BiLSTM',           build_bilstm,           'deep'),
    ('CNN-LSTM',         build_cnn_lstm,         'deep'),
    ('CNN-BiLSTM',       build_cnn_bilstm,       'deep'),
    ('CNN-Mamba orig',   build_cnn_mamba,        'mamba'),
    ('CNN-Mamba tuned',  build_cnn_mamba_tuned,  'mamba'),
    ('CNN-Mamba + Diff', build_cnn_mamba_diff,   'mamba'),
]


def benchmark(model):
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    a = torch.randn(BATCH, SEQ_LEN, FEAT, device=device)
    b = torch.randn(BATCH, SEQ_LEN, FEAT, device=device)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(a, b)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(ITERS):
            _ = model(a, b)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - t0

    batch_ms  = elapsed / ITERS * 1000
    sample_ms = batch_ms / BATCH
    return n_params, batch_ms, sample_ms


# -----------------------------------------------------
def run_all():
    results = []
    for name, build_fn, tag in MODELS:
        try:
            model = build_fn()
        except Exception as e:
            print(f'[SKIP] {name}: build failed -> {e}')
            continue
        try:
            n, bms, sms = benchmark(model)
        except Exception as e:
            print(f'[SKIP] {name}: benchmark failed -> {e}')
            continue
        results.append({
            'name':         name,
            'type':         tag,
            'n_params':     int(n),
            'batch_ms':     float(bms),
            'sample_ms':    float(sms),
        })
        print(f'{name:<22} params={n:>10,}  batch_ms={bms:6.2f}  per-sample_ms={sms:6.3f}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    with open(f'{OUT_DIR}/benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_DIR}/benchmark.json')
    return results


def plot_benchmark(results):
    """Two-panel: parameter count and inference time per sample."""
    # attach known val acc (from previous runs)
    acc_map = {
        'ANN': 85.50, 'CNN': 83.00, 'LSTM': 40.97,
        'BiGRU': 86.00, 'BiLSTM': 87.00,
        'CNN-LSTM': 91.42, 'CNN-BiLSTM': 91.87,
        'CNN-Mamba orig': 88.67, 'CNN-Mamba tuned': 92.38,
    }
    # try to include diff result
    try:
        with open('/home/yangcq/track_association/output/diff_input/diff_result.json') as f:
            diff_r = json.load(f)
        acc_map['CNN-Mamba + Diff'] = diff_r['best_val_acc'] * 100
    except FileNotFoundError:
        pass

    results = [r for r in results if r['name'] in acc_map]

    names    = [r['name'] for r in results]
    params_k = [r['n_params'] / 1000 for r in results]
    sample_us = [r['sample_ms'] * 1000 for r in results]  # microseconds
    accs      = [acc_map.get(r['name'], 0) for r in results]

    # single bubble plot: x=params, y=accuracy, bubble size = speed
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for r, p, a in zip(results, params_k, accs):
        c = '#d62728' if 'Diff' in r['name'] else (
            '#ff7b00' if 'tuned' in r['name'] else
            ('#2a7de1' if 'Mamba' in r['name'] else '#888888'))
        ax.scatter(p, a, s=max(80, 500 - r['sample_ms']*200),
                   color=c, alpha=0.75, edgecolors='black', linewidth=1.2)
        ax.annotate(r['name'], (p, a), xytext=(6, 6),
                    textcoords='offset points', fontsize=10)

    ax.set_xlabel('Parameters (K)', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Parameter Count  (bubble size ~ inference speed)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.35, linestyle='--')
    plt.tight_layout()
    path = f'{FIG_DIR}/bench_accuracy_vs_params.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

    # bar chart: per-sample inference time
    order = np.argsort(sample_us)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ys = np.array(names)[order]
    xs = np.array(sample_us)[order]
    ax.barh(ys, xs, color='#2a7de1', edgecolor='black', alpha=0.85)
    for i, v in enumerate(xs):
        ax.text(v + 2, i, f'{v:.1f} μs', va='center', fontsize=10)
    ax.set_xlabel('Per-sample Inference Time (μs, batch=64 on GPU)',
                  fontsize=11)
    ax.set_title('Inference Speed Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.35, axis='x', linestyle='--')
    plt.tight_layout()
    path = f'{FIG_DIR}/bench_inference_speed.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    results = run_all()
    plot_benchmark(results)
    print('\nDone.')
