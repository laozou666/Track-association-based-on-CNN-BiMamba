"""
Evaluate the 4 traditional methods (GRA / Euclidean / Pearson / KNN) on the
TEST set (final_dataset/test) and merge with the deep-learning test-set
results to produce a unified leaderboard + bar charts.

Procedure
---------
GRA / Euclidean / Pearson :
    Re-use threshold tuning on a balanced subset (8000) of the training set,
    then score the full test set with that threshold.

KNN :
    Re-use the 11-dim hand-crafted features. Refit StandardScaler + KNN(K=21)
    on FULL train, predict on full test.

Outputs
-------
output/test_eval/traditional_test_metrics.json
output/test_eval/all_methods_test_metrics.csv
output/test_eval/all_methods_test_metrics.json
output/test_eval/bar_test_acc_all.png        (sorted; deep vs traditional colored)
output/test_eval/bar_test_metrics_all.png    (4-metric grouped)
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from traditional_gra import (grey_relational_grade, euclidean_similarity,
                             pearson_similarity, compute_scores, tune_threshold)
from traditional_knn import extract_features

DATA_DIR = '/home/yangcq/track_association/data/final_dataset'
OUT_DIR  = '/home/yangcq/track_association/output/test_eval'
os.makedirs(OUT_DIR, exist_ok=True)

DEEP_JSON = f'{OUT_DIR}/test_metrics.json'           # produced earlier
TRAD_JSON = f'{OUT_DIR}/traditional_test_metrics.json'

METHODS = {
    'GRA':       grey_relational_grade,
    'Euclidean': euclidean_similarity,
    'Pearson':   pearson_similarity,
}


# --------------------------------------------------------------
print('Loading data ...', flush=True)
t1_tr = np.load(f'{DATA_DIR}/track1_train.npy')
t2_tr = np.load(f'{DATA_DIR}/track2_train.npy')
len_tr = np.load(f'{DATA_DIR}/lengths_train.npy')
y_tr  = np.load(f'{DATA_DIR}/labels_train.npy').astype(int)

t1_te = np.load(f'{DATA_DIR}/track1_test.npy')
t2_te = np.load(f'{DATA_DIR}/track2_test.npy')
len_te = np.load(f'{DATA_DIR}/lengths_test.npy')
y_te  = np.load(f'{DATA_DIR}/labels_test.npy').astype(int)
print(f'  train: {len(y_tr)}  test: {len(y_te)}', flush=True)


# --------------------------------------------------------------
# Threshold-based methods
# --------------------------------------------------------------
trad_results = []
rng = np.random.default_rng(0)
pos_idx = np.where(y_tr == 1)[0]
neg_idx = np.where(y_tr == 0)[0]
k = 8000 // 2
sel = np.concatenate([
    rng.choice(pos_idx, min(k, len(pos_idx)), replace=False),
    rng.choice(neg_idx, min(k, len(neg_idx)), replace=False),
])
print(f'Tuning thresholds on {len(sel)} balanced train pairs '
      f'({sum(y_tr[sel]==1)} pos / {sum(y_tr[sel]==0)} neg)', flush=True)

for name, func in METHODS.items():
    print(f'\n==== {name} ====', flush=True)
    t0 = time.time()

    s_tr = compute_scores(t1_tr[sel], t2_tr[sel], len_tr[sel], func,
                          tag=f'{name} train')
    th, tr_acc = tune_threshold(s_tr, y_tr[sel])
    print(f'  best threshold = {th:.4f}   train_acc = {tr_acc:.4f}', flush=True)

    s_te = compute_scores(t1_te, t2_te, len_te, func, tag=f'{name} test')
    pred = (s_te > th).astype(int)
    metrics = {
        'method':    name,
        'threshold': th,
        'n_params':  0,
        'eval_seconds': round(time.time() - t0, 2),
        'accuracy':  accuracy_score(y_te, pred),
        'precision': precision_score(y_te, pred, zero_division=0),
        'recall':    recall_score(y_te, pred, zero_division=0),
        'f1':        f1_score(y_te, pred, zero_division=0),
        'auc':       roc_auc_score(y_te, s_te),
    }
    print(f'  TEST acc={metrics["accuracy"]*100:.2f}%  '
          f'f1={metrics["f1"]*100:.2f}%  auc={metrics["auc"]*100:.2f}%',
          flush=True)
    trad_results.append(metrics)


# --------------------------------------------------------------
# KNN
# --------------------------------------------------------------
print('\n==== KNN ====', flush=True)
t0 = time.time()
print('  feature extraction (train) ...', flush=True)
X_tr = extract_features(t1_tr, t2_tr, len_tr, tag='train')
print('  feature extraction (test)  ...', flush=True)
X_te = extract_features(t1_te, t2_te, len_te, tag='test')

scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)

K = 21
clf = KNeighborsClassifier(n_neighbors=K, n_jobs=-1, weights='distance')
clf.fit(X_tr_s, y_tr)
proba = clf.predict_proba(X_te_s)[:, 1]
pred  = (proba >= 0.5).astype(int)

knn_metrics = {
    'method':    'KNN',
    'best_K':    K,
    'n_params':  0,
    'eval_seconds': round(time.time() - t0, 2),
    'accuracy':  accuracy_score(y_te, pred),
    'precision': precision_score(y_te, pred, zero_division=0),
    'recall':    recall_score(y_te, pred, zero_division=0),
    'f1':        f1_score(y_te, pred, zero_division=0),
    'auc':       roc_auc_score(y_te, proba),
}
print(f'  TEST acc={knn_metrics["accuracy"]*100:.2f}%  '
      f'f1={knn_metrics["f1"]*100:.2f}%  '
      f'auc={knn_metrics["auc"]*100:.2f}%', flush=True)
trad_results.append(knn_metrics)

with open(TRAD_JSON, 'w') as f:
    json.dump(trad_results, f, indent=2, ensure_ascii=False)
print(f'\nSaved: {TRAD_JSON}', flush=True)


# --------------------------------------------------------------
# Merge with deep results, sort, plot
# --------------------------------------------------------------
with open(DEEP_JSON) as f:
    deep_results = json.load(f)

DEEP_NAMES = {r['model'] for r in deep_results}

def label(r):
    return r.get('model') or r.get('method')

merged = []
for r in deep_results:
    merged.append({'name': r['model'],
                   'kind': 'deep',
                   'n_params': r['n_params'],
                   'accuracy':  r['accuracy'],
                   'precision': r['precision'],
                   'recall':    r['recall'],
                   'f1':        r['f1'],
                   'auc':       r['auc']})
for r in trad_results:
    merged.append({'name': r['method'],
                   'kind': 'traditional',
                   'n_params': 0,
                   'accuracy':  r['accuracy'],
                   'precision': r['precision'],
                   'recall':    r['recall'],
                   'f1':        r['f1'],
                   'auc':       r['auc']})

merged.sort(key=lambda x: x['accuracy'], reverse=True)

csv_path = f'{OUT_DIR}/all_methods_test_metrics.csv'
with open(csv_path, 'w') as f:
    f.write('rank,name,kind,n_params,accuracy,precision,recall,f1,auc\n')
    for i, r in enumerate(merged, 1):
        f.write(f'{i},{r["name"]},{r["kind"]},{r["n_params"]},'
                f'{r["accuracy"]:.6f},{r["precision"]:.6f},'
                f'{r["recall"]:.6f},{r["f1"]:.6f},{r["auc"]:.6f}\n')
with open(f'{OUT_DIR}/all_methods_test_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)


def color_for(r):
    if r['kind'] == 'traditional':
        return '#7C3AED'                                    # purple
    n = r['name'].lower()
    if 'untuned' in n:                            return '#9E480E'
    if 'tuned'   in n:                            return '#C00000'
    if 'forward' in n or 'backward' in n:         return '#ED7D31'
    if 'pure'    in n:                            return '#70AD47'
    if 'cnn'     in n:                            return '#4472C4'
    return '#7F7F7F'


# 1) accuracy-only
names = [r['name'] for r in merged]
accs  = [r['accuracy'] * 100 for r in merged]
colors = [color_for(r) for r in merged]

fig, ax = plt.subplots(figsize=(13.5, 6.2))
bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=0.6, width=0.65)
ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=8.5)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Test-set accuracy: deep learning vs traditional methods', fontsize=13)
ax.set_ylim(max(0, min(accs) - 5), 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')

from matplotlib.patches import Patch
legend = [
    Patch(color='#C00000', label='CNN-BiMamba (tuned)'),
    Patch(color='#9E480E', label='CNN-BiMamba (untuned)'),
    Patch(color='#ED7D31', label='Mamba (uni-directional)'),
    Patch(color='#70AD47', label='Pure BiMamba'),
    Patch(color='#4472C4', label='Other CNN baselines'),
    Patch(color='#7F7F7F', label='RNN / ANN baselines'),
    Patch(color='#7C3AED', label='Traditional methods'),
]
ax.legend(handles=legend, loc='lower left', ncol=2, fontsize=8, framealpha=0.9)
plt.xticks(rotation=25, ha='right', fontsize=8.5)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/bar_test_acc_all.png', dpi=160)
plt.close()

# 2) grouped 4-metric
keys  = ['accuracy', 'precision', 'recall', 'f1']
labs  = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(merged))
width = 0.2

fig, ax = plt.subplots(figsize=(15, 6.2))
for i, (k, mn) in enumerate(zip(keys, labs)):
    vals = [r[k] * 100 for r in merged]
    ax.bar(x + (i - 1.5) * width, vals, width, label=mn, edgecolor='black', linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8.5)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Test-set metrics: deep learning vs traditional methods', fontsize=13)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='lower left', ncol=4, fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/bar_test_metrics_all.png', dpi=160)
plt.close()

print(f'\n[saved] {csv_path}')
print(f'[saved] {OUT_DIR}/all_methods_test_metrics.json')
print(f'[saved] {OUT_DIR}/bar_test_acc_all.png')
print(f'[saved] {OUT_DIR}/bar_test_metrics_all.png')

# Print final ranking
print('\n[final ranking on TEST]')
for i, r in enumerate(merged, 1):
    print(f'  {i:>2}. [{r["kind"][:4]}] {r["name"]:<28}  '
          f'acc={r["accuracy"]*100:6.2f}%  '
          f'f1={r["f1"]*100:6.2f}%  '
          f'auc={r["auc"]*100:6.2f}%')
