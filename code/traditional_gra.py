"""
Traditional (non-deep-learning) track association baselines.

Implemented methods:
  1. Grey Relational Analysis  (GRA)  <- main method
  2. Pointwise Euclidean distance
  3. Pearson correlation coefficient (per feature, averaged)

Procedure:
  - Uses the SAME train / val split as the deep learning models
    (/home/yangcq/track_association/data/final_dataset/*.npy).
  - For each track pair (A, B), truncate to each track's real length,
    then align to the common length  L = min(len_A, len_B).
  - Compute a similarity score.
  - Tune the decision threshold on a subset of the training set
    to maximise accuracy, then evaluate on the full validation set.

Output:
  output/traditional/gra_metrics.json
  output/figures/traditional_roc.png
  output/figures/traditional_confusion.png

Runs on CPU (does not interfere with GPU jobs).
"""
import os, json, time
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------
DATA_DIR = '/home/yangcq/track_association/data/final_dataset'
OUT_DIR  = '/home/yangcq/track_association/output/traditional'
FIG_DIR  = '/home/yangcq/track_association/output/figures'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# --------------------------------------------------------------
# Similarity functions  (all work on 1 pair at a time)
# --------------------------------------------------------------
def grey_relational_grade(a, b, rho=0.5):
    """Classical grey relational grade, range ~(0, 1].

    a, b : ndarray (L, D) with REAL length already truncated.
    rho  : distinguishing coefficient, 0.5 is the textbook default.
    """
    L = min(len(a), len(b))
    if L < 2:
        return 0.0
    a = a[:L].astype(np.float32)
    b = b[:L].astype(np.float32)

    diff = np.abs(a - b)              # (L, D)
    min_ref = diff.min()
    max_ref = diff.max()
    if max_ref < 1e-12:
        return 1.0                    # identical sequences
    coeff = (min_ref + rho * max_ref) / (diff + rho * max_ref)
    return float(coeff.mean())


def euclidean_similarity(a, b):
    """Negative mean L2 distance -> map to similarity in (0,1]."""
    L = min(len(a), len(b))
    if L < 2:
        return 0.0
    d = np.linalg.norm(a[:L] - b[:L], axis=-1).mean()
    return float(1.0 / (1.0 + d))     # monotone transform


def pearson_similarity(a, b):
    """Mean Pearson correlation across feature dimensions."""
    L = min(len(a), len(b))
    if L < 3:
        return 0.0
    a = a[:L]
    b = b[:L]
    corrs = []
    for d in range(a.shape[1]):
        ad = a[:, d] - a[:, d].mean()
        bd = b[:, d] - b[:, d].mean()
        denom = np.sqrt((ad ** 2).sum() * (bd ** 2).sum())
        if denom < 1e-12:
            continue
        corrs.append((ad * bd).sum() / denom)
    if not corrs:
        return 0.0
    return float(np.mean(corrs))      # range [-1, 1]


METHODS = {
    'GRA':       grey_relational_grade,
    'Euclidean': euclidean_similarity,
    'Pearson':   pearson_similarity,
}


# --------------------------------------------------------------
# Compute scores for a full set
# --------------------------------------------------------------
def compute_scores(t1, t2, lengths, func, tag=''):
    N = len(t1)
    scores = np.zeros(N, dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        L = int(lengths[i])
        scores[i] = func(t1[i, :L], t2[i, :L])
        if (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f'  [{tag}] {i+1:>6}/{N}   {rate:.0f} pairs/sec')
    print(f'  [{tag}] done in {time.time() - t0:.1f}s')
    return scores


# --------------------------------------------------------------
# Threshold tuning on (a subset of) the training set
# --------------------------------------------------------------
def tune_threshold(scores, labels, n_grid=401):
    """Grid search the decision threshold that maximises accuracy."""
    lo, hi = float(scores.min()), float(scores.max())
    thresholds = np.linspace(lo, hi, n_grid)
    best_th, best_acc = 0.0, 0.0
    for th in thresholds:
        acc = ((scores > th).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc, best_th = acc, float(th)
    return best_th, best_acc


# --------------------------------------------------------------
# Full evaluation for one method
# --------------------------------------------------------------
def evaluate_method(method_name, func,
                    t1_tr, t2_tr, len_tr, y_tr,
                    t1_vl, t2_vl, len_vl, y_vl,
                    train_subset=8000, rng=None):
    print(f'\n==== {method_name} ====')

    # 1. tune threshold on a TRAIN subset (class-balanced)
    rng = rng or np.random.default_rng(0)
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    k = train_subset // 2
    sel = np.concatenate([
        rng.choice(pos_idx, min(k, len(pos_idx)), replace=False),
        rng.choice(neg_idx, min(k, len(neg_idx)), replace=False),
    ])
    print(f'  tuning threshold on {len(sel)} balanced train pairs '
          f'({sum(y_tr[sel]==1)} pos / {sum(y_tr[sel]==0)} neg)')
    s_tr = compute_scores(t1_tr[sel], t2_tr[sel], len_tr[sel], func,
                          tag=f'{method_name} train')
    th, tr_acc = tune_threshold(s_tr, y_tr[sel])
    print(f'  best threshold = {th:.4f}   train_acc (subset) = {tr_acc:.4f}')

    # 2. full validation set
    print(f'  scoring full validation set (N = {len(y_vl)})')
    s_vl = compute_scores(t1_vl, t2_vl, len_vl, func,
                          tag=f'{method_name} val')
    y_pred = (s_vl > th).astype(int)

    acc  = accuracy_score(y_vl, y_pred)
    prec = precision_score(y_vl, y_pred, zero_division=0)
    rec  = recall_score(y_vl, y_pred, zero_division=0)
    f1   = f1_score(y_vl, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_vl, s_vl)
    except ValueError:
        auc = float('nan')

    print(f'  VAL   acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  '
          f'f1={f1:.4f}  auc={auc:.4f}')

    fpr, tpr, _ = roc_curve(y_vl, s_vl)
    cm = confusion_matrix(y_vl, y_pred)

    return {
        'method':      method_name,
        'threshold':   th,
        'train_acc':   tr_acc,
        'val_acc':     acc,
        'val_precision': prec,
        'val_recall':  rec,
        'val_f1':      f1,
        'val_auc':     auc,
        'confusion':   cm.tolist(),
        'roc_fpr':     fpr.tolist(),
        'roc_tpr':     tpr.tolist(),
        'scores_val':  s_vl.tolist(),   # for later plotting if needed
    }


# --------------------------------------------------------------
# Plots
# --------------------------------------------------------------
def plot_roc(results):
    plt.figure(figsize=(7.5, 6.5))
    colors = {'GRA': '#d62728', 'Euclidean': '#1f77b4', 'Pearson': '#2ca02c'}
    for r in results:
        plt.plot(r['roc_fpr'], r['roc_tpr'],
                 color=colors.get(r['method'], 'k'), linewidth=2.1,
                 label=f"{r['method']}  (AUC = {r['val_auc']:.3f}, "
                       f"Acc = {r['val_acc']*100:.2f}%)")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate',  fontsize=12)
    plt.title('Traditional Track Association Methods  -  ROC on Validation',
              fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    path = f'{FIG_DIR}/traditional_roc.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_confusion(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r['confusion'])
        cm_pct = cm / cm.sum() * 100
        im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=cm_pct.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                        ha='center', va='center', fontsize=11,
                        color='white' if cm_pct[i, j] > cm_pct.max() / 2
                        else 'black')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        ax.set_title(f"{r['method']}\nAcc = {r['val_acc']*100:.2f}%",
                     fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = f'{FIG_DIR}/traditional_confusion.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_vs_deep(results):
    """Traditional vs deep learning bar chart."""
    deep = [
        ('LSTM',         40.97),
        ('CNN',          83.00),
        ('ANN',          85.50),
        ('BiGRU',        86.00),
        ('BiLSTM',       87.00),
        ('CNN-Mamba orig', 88.67),
        ('CNN-LSTM',     91.42),
        ('CNN-BiLSTM',   91.87),
        ('CNN-Mamba tuned', 92.38),
    ]
    trad = [(r['method'], r['val_acc'] * 100) for r in results]
    combined = [(n, a, 'Traditional') for n, a in trad] + \
               [(n, a, 'Deep Learning') for n, a in deep]
    combined.sort(key=lambda x: x[1])

    names  = [c[0] for c in combined]
    accs   = [c[1] for c in combined]
    colors = ['#d62728' if c[2] == 'Traditional' else '#2a7de1'
              for c in combined]

    plt.figure(figsize=(11, 6.5))
    bars = plt.barh(names, accs, color=colors, edgecolor='black', alpha=0.88)
    for b, a in zip(bars, accs):
        plt.text(a + 0.25, b.get_y() + b.get_height() / 2,
                 f'{a:.2f}%', va='center', fontsize=10)
    plt.xlabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Traditional vs Deep Learning Track Association',
              fontsize=13, fontweight='bold')
    plt.xlim(0, max(accs) + 6)

    from matplotlib.patches import Patch
    legend = [Patch(color='#d62728', label='Traditional'),
              Patch(color='#2a7de1', label='Deep Learning')]
    plt.legend(handles=legend, loc='lower right')
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')
    plt.tight_layout()
    path = f'{FIG_DIR}/traditional_vs_deep.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
if __name__ == '__main__':
    print('Loading data ...')
    t1_tr = np.load(f'{DATA_DIR}/track1_train.npy')
    t2_tr = np.load(f'{DATA_DIR}/track2_train.npy')
    len_tr = np.load(f'{DATA_DIR}/lengths_train.npy')
    y_tr = np.load(f'{DATA_DIR}/labels_train.npy').astype(int)

    t1_vl = np.load(f'{DATA_DIR}/track1_val.npy')
    t2_vl = np.load(f'{DATA_DIR}/track2_val.npy')
    len_vl = np.load(f'{DATA_DIR}/lengths_val.npy')
    y_vl = np.load(f'{DATA_DIR}/labels_val.npy').astype(int)
    print(f'  train: {len(y_tr)}  val: {len(y_vl)}')

    results = []
    for name, func in METHODS.items():
        r = evaluate_method(name, func,
                            t1_tr, t2_tr, len_tr, y_tr,
                            t1_vl, t2_vl, len_vl, y_vl,
                            train_subset=8000)
        results.append(r)

    # Save
    light = [{k: v for k, v in r.items()
              if k not in ('scores_val', 'roc_fpr', 'roc_tpr')}
             for r in results]
    with open(f'{OUT_DIR}/gra_metrics.json', 'w') as f:
        json.dump(light, f, indent=2)
    print(f'\nSaved: {OUT_DIR}/gra_metrics.json')

    # Plots
    plot_roc(results)
    plot_confusion(results)
    plot_vs_deep(results)

    # Summary
    print('\n' + '=' * 60)
    print(f'{"Method":<12}{"Acc":<9}{"Prec":<9}{"Rec":<9}{"F1":<9}{"AUC":<9}')
    print('-' * 60)
    for r in results:
        print(f'{r["method"]:<12}'
              f'{r["val_acc"]*100:<9.2f}'
              f'{r["val_precision"]*100:<9.2f}'
              f'{r["val_recall"]*100:<9.2f}'
              f'{r["val_f1"]*100:<9.2f}'
              f'{r["val_auc"]*100:<9.2f}')
