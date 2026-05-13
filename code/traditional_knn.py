"""
Traditional KNN baseline for track association.

Feature engineering per track pair  (hand-crafted, 11-dim):
  1. common length L           (= min(len_a, len_b))
  2. length diff   |L_a - L_b|
  3. mean |diff|  for each of the 4 features  (4 dims)
  4. std  of diff for each of the 4 features  (4 dims)
  5. GRA score (reuses grey_relational_grade)

Workflow:
  - Extract features from train + val (CPU, vectorised where possible).
  - StandardScaler on train, apply to val.
  - KNN with K in {1, 3, 5, 7, 9, 15, 21, 31, 51}.
  - Pick best K on a held-out subset of train (stratified 80/20),
    then retrain on full train and evaluate on val.
  - Save metrics, ROC, confusion.
"""
import os, json, time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traditional_gra import grey_relational_grade

# --------------------------------------------------------------
DATA_DIR = '/home/yangcq/track_association/data/final_dataset'
OUT_DIR  = '/home/yangcq/track_association/output/traditional'
FIG_DIR  = '/home/yangcq/track_association/output/figures'
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------
FEATURE_NAMES = (
    ['common_len', 'len_diff'] +
    [f'mean_diff_f{d}' for d in range(4)] +
    [f'std_diff_f{d}'  for d in range(4)] +
    ['gra_score']
)


def extract_features(t1, t2, lengths, tag=''):
    """Return (N, 11) feature matrix."""
    N = len(t1)
    X = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        L = int(lengths[i])
        if L < 2:
            continue
        a = t1[i, :L]
        b = t2[i, :L]

        X[i, 0] = L
        X[i, 1] = 0                           # same length by construction
        diff = np.abs(a - b)                  # (L, 4)
        X[i, 2:6] = diff.mean(axis=0)
        X[i, 6:10] = diff.std(axis=0)
        X[i, 10] = grey_relational_grade(a, b)

        if (i + 1) % 10000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f'  [{tag}] {i+1:>6}/{N}   {rate:.0f} pairs/sec')
    print(f'  [{tag}] extracted {N} pairs in {time.time() - t0:.1f}s')
    return X


# --------------------------------------------------------------
# K sweep on held-out train split
# --------------------------------------------------------------
def sweep_k(Xt, yt, Xv, yv, k_list):
    print('\nK sweep (on train held-out split):')
    print(f'  {"K":<6}{"acc":<10}{"f1":<10}')
    print('  ' + '-' * 26)
    results = []
    for k in k_list:
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
        clf.fit(Xt, yt)
        pred = clf.predict(Xv)
        acc = accuracy_score(yv, pred)
        f1  = f1_score(yv, pred, zero_division=0)
        results.append({'K': k, 'acc': acc, 'f1': f1})
        print(f'  {k:<6}{acc:<10.4f}{f1:<10.4f}')
    return results


# --------------------------------------------------------------
# Plots
# --------------------------------------------------------------
def plot_k_sweep(sweep, best_k, out_path):
    ks    = [r['K']   for r in sweep]
    accs  = [r['acc'] * 100 for r in sweep]
    f1s   = [r['f1']  * 100 for r in sweep]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, accs, 'o-', color='#2a7de1', linewidth=2, markersize=8,
            label='Accuracy')
    ax.plot(ks, f1s,  's--', color='#d62728', linewidth=2, markersize=7,
            label='F1 score')
    for x, a in zip(ks, accs):
        ax.annotate(f'{a:.2f}', (x, a), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9, color='#2a7de1')
    ax.axvline(best_k, linestyle=':', color='orange', alpha=0.6,
               label=f'Best K = {best_k}')
    ax.set_xlabel('K (number of neighbours)', fontsize=12)
    ax.set_ylabel('Metric (%)', fontsize=12)
    ax.set_title('KNN – K sweep on train held-out split',
                 fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def plot_roc_cm(method, y_true, scores, y_pred, out_roc, out_cm):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    plt.figure(figsize=(6.5, 6))
    plt.plot(fpr, tpr, linewidth=2.2, color='#d62728',
             label=f'{method}  (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{method} – ROC on Validation', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_roc, dpi=200, bbox_inches='tight'); plt.close()
    print(f'Saved: {out_roc}')

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum() * 100
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=cm_pct.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=11,
                    color='white' if cm_pct[i, j] > cm_pct.max() / 2 else 'black')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['True 0', 'True 1'])
    ax.set_title(f"{method}\nAcc = {accuracy_score(y_true, y_pred)*100:.2f}%",
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_cm, dpi=200, bbox_inches='tight'); plt.close()
    print(f'Saved: {out_cm}')
    return auc


def plot_combined_vs_deep(knn_acc):
    """Update the combined traditional-vs-deep comparison with KNN included."""
    # load previous traditional results
    try:
        with open(f'{OUT_DIR}/gra_metrics.json') as f:
            trad = json.load(f)
    except FileNotFoundError:
        trad = []
    trad_entries = [(r['method'], r['val_acc'] * 100) for r in trad]
    trad_entries.append(('KNN', knn_acc * 100))

    deep = [
        ('LSTM',            40.97),
        ('CNN',             83.00),
        ('ANN',             85.50),
        ('BiGRU',           86.00),
        ('BiLSTM',          87.00),
        ('CNN-Mamba orig',  88.67),
        ('CNN-LSTM',        91.42),
        ('CNN-BiLSTM',      91.87),
        ('CNN-Mamba tuned', 92.38),
    ]
    combined = [(n, a, 'Traditional') for n, a in trad_entries] + \
               [(n, a, 'Deep Learning') for n, a in deep]
    combined.sort(key=lambda x: x[1])

    names  = [c[0] for c in combined]
    accs   = [c[1] for c in combined]
    colors = ['#d62728' if c[2] == 'Traditional' else '#2a7de1'
              for c in combined]

    plt.figure(figsize=(11, 7))
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
    print(f'Saved (updated): {path}')


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

    # --- feature extraction ---
    print('\nExtracting features ...')
    X_tr = extract_features(t1_tr, t2_tr, len_tr, tag='train')
    X_vl = extract_features(t1_vl, t2_vl, len_vl, tag='val')

    # save features to avoid recomputation
    np.savez(f'{OUT_DIR}/pair_features.npz',
             X_tr=X_tr, X_vl=X_vl, y_tr=y_tr, y_vl=y_vl,
             feature_names=np.array(FEATURE_NAMES))
    print(f'Saved: {OUT_DIR}/pair_features.npz')

    # --- standardise ---
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    # --- K sweep on held-out train split ---
    Xa, Xb, ya, yb = train_test_split(X_tr_s, y_tr,
                                      test_size=0.2, random_state=0,
                                      stratify=y_tr)
    k_list = [1, 3, 5, 7, 9, 15, 21, 31, 51]
    sweep = sweep_k(Xa, ya, Xb, yb, k_list)
    best = max(sweep, key=lambda r: r['acc'])
    best_k = best['K']
    print(f"\nBest K = {best_k}  (held-out acc = {best['acc']:.4f})")

    plot_k_sweep(sweep, best_k, f'{FIG_DIR}/knn_k_sweep.png')

    # --- retrain on full train, evaluate on val ---
    print(f'\nRetraining KNN(K={best_k}) on FULL train ...')
    clf = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1,
                               weights='distance')
    t0 = time.time()
    clf.fit(X_tr_s, y_tr)
    print(f'  fit time: {time.time() - t0:.1f}s')

    print('Predicting on validation ...')
    t0 = time.time()
    proba = clf.predict_proba(X_vl_s)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    print(f'  predict time: {time.time() - t0:.1f}s')

    acc  = accuracy_score(y_vl, pred)
    prec = precision_score(y_vl, pred, zero_division=0)
    rec  = recall_score(y_vl, pred, zero_division=0)
    f1   = f1_score(y_vl, pred, zero_division=0)
    auc  = roc_auc_score(y_vl, proba)

    print('\n' + '=' * 50)
    print(f'KNN (K={best_k})  VAL:')
    print(f'  Accuracy   = {acc * 100:.2f}%')
    print(f'  Precision  = {prec * 100:.2f}%')
    print(f'  Recall     = {rec * 100:.2f}%')
    print(f'  F1         = {f1 * 100:.2f}%')
    print(f'  AUC        = {auc * 100:.2f}%')

    metrics = {
        'method':        'KNN',
        'best_K':        int(best_k),
        'n_features':    X_tr.shape[1],
        'feature_names': FEATURE_NAMES,
        'k_sweep':       sweep,
        'val_acc':       acc,
        'val_precision': prec,
        'val_recall':    rec,
        'val_f1':        f1,
        'val_auc':       auc,
        'confusion':     confusion_matrix(y_vl, pred).tolist(),
    }
    with open(f'{OUT_DIR}/knn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nSaved: {OUT_DIR}/knn_metrics.json')

    plot_roc_cm('KNN', y_vl, proba, pred,
                f'{FIG_DIR}/knn_roc.png',
                f'{FIG_DIR}/knn_confusion.png')

    plot_combined_vs_deep(acc)
