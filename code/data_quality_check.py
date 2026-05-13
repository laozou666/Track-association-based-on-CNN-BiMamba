"""
Comprehensive quality diagnostics for the regenerated track-association dataset.

Checks:
  1. Class balance & sample counts
  2. Sequence-length distribution (positive vs negative) -> verifies Bug 2 fix
  3. Kinematic-similarity test (velocity / course gaps) -> verifies Bug 1 fix
  4. Padding-region sanity      (only real frames non-zero?)  -> verifies Bug 3 fix
  5. Per-feature statistics & normalization scale
  6. Trivial-separability stress test:
        train a logistic-regression on hand-crafted scalar features
        (mean/std of velocity, course, position) — if it gets >>50%,
        the dataset still has shortcut leakage.
  7. Train / val distribution drift (KS test)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

DATA_DIR = '/home/yangcq/track_association/data/final_dataset'
OUT_DIR  = '/home/yangcq/track_association/output/data_quality'
os.makedirs(OUT_DIR, exist_ok=True)

REPORT = []
def log(msg=''):
    print(msg)
    REPORT.append(msg)

def section(title):
    log('\n' + '='*70)
    log(f'  {title}')
    log('='*70)

# -----------------------------------------------------------------
# Load
# -----------------------------------------------------------------
section('0. Files & Shapes')
files = {
    't1_tr':   'track1_train.npy',
    't2_tr':   'track2_train.npy',
    'lb_tr':   'labels_train.npy',
    'len_tr':  'lengths_train.npy',
    't1_vl':   'track1_val.npy',
    't2_vl':   'track2_val.npy',
    'lb_vl':   'labels_val.npy',
    'len_vl':  'lengths_val.npy',
    't1_te':   'track1_test.npy',
    't2_te':   'track2_test.npy',
    'lb_te':   'labels_test.npy',
    'len_te':  'lengths_test.npy',
    'mean':    'scaler_mean.npy',
    'scale':   'scaler_scale.npy',
}
arr = {}
for k, f in files.items():
    p = f'{DATA_DIR}/{f}'
    arr[k] = np.load(p)
    log(f'  {f:25s}  shape={str(arr[k].shape):20s}  dtype={arr[k].dtype}')

# -----------------------------------------------------------------
# 1. Class balance
# -----------------------------------------------------------------
section('1. Class balance')
for split, lb in [('train', arr['lb_tr']), ('val', arr['lb_vl']), ('test', arr['lb_te'])]:
    pos = int((lb == 1).sum()); neg = int((lb == 0).sum())
    log(f'  {split:5s}: total={len(lb):>6}  pos={pos:>6} ({pos/len(lb)*100:5.1f}%)  '
        f'neg={neg:>6} ({neg/len(lb)*100:5.1f}%)  ratio={pos/neg:.2f}')

# -----------------------------------------------------------------
# 2. Sequence length distribution (Bug 2 fix verification)
# -----------------------------------------------------------------
section('2. Sequence-length distribution  (Bug 2 fix)')
len_tr = arr['len_tr']         # shape (N, 2)  -> [len_track1, len_track2]
lb_tr  = arr['lb_tr']

if len_tr.ndim == 2:
    L1_pos = len_tr[lb_tr == 1, 0]; L2_pos = len_tr[lb_tr == 1, 1]
    L1_neg = len_tr[lb_tr == 0, 0]; L2_neg = len_tr[lb_tr == 0, 1]
else:
    L1_pos = L2_pos = len_tr[lb_tr == 1]
    L1_neg = L2_neg = len_tr[lb_tr == 0]

def stats(name, x):
    log(f'  {name:18s} n={len(x):>5}  '
        f'mean={x.mean():6.1f}  std={x.std():5.1f}  '
        f'min={x.min():>3}  med={np.median(x):>5.1f}  max={x.max():>3}')
stats('pos track1 length', L1_pos)
stats('pos track2 length', L2_pos)
stats('neg track1 length', L1_neg)
stats('neg track2 length', L2_neg)

# KS test: are pos/neg lengths statistically different?
ks1 = ks_2samp(L1_pos, L1_neg)
ks2 = ks_2samp(L2_pos, L2_neg)
log(f'\n  KS test (pos vs neg) on track1 length: D={ks1.statistic:.4f}  p={ks1.pvalue:.4g}')
log(f'  KS test (pos vs neg) on track2 length: D={ks2.statistic:.4f}  p={ks2.pvalue:.4g}')
if ks1.statistic < 0.05 and ks2.statistic < 0.05:
    log('  ✓ Length distributions are nearly identical (Bug 2 fixed)')
elif ks1.statistic < 0.15:
    log('  ~ Length distributions differ slightly but acceptable')
else:
    log('  ✗ Length distributions differ a lot — possible length leakage')

# Plot length histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, p, n, title in [(axes[0], L1_pos, L1_neg, 'Track 1 length'),
                         (axes[1], L2_pos, L2_neg, 'Track 2 length')]:
    ax.hist(p, bins=40, alpha=0.6, label=f'pos (n={len(p)})', color='#1f77b4')
    ax.hist(n, bins=40, alpha=0.6, label=f'neg (n={len(n)})', color='#d62728')
    ax.set_title(title); ax.set_xlabel('length'); ax.set_ylabel('count')
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_length_distribution.png', dpi=140, bbox_inches='tight')
plt.close()
log(f'  Saved: {OUT_DIR}/01_length_distribution.png')

# -----------------------------------------------------------------
# 3. Kinematic similarity (Bug 1 fix verification)
# -----------------------------------------------------------------
section('3. Kinematic similarity of paired tracks  (Bug 1 fix)')

# Feature layout: x = [lat, lon, vel, cou]
LAT, LON, VEL, COU = 0, 1, 2, 3

def real_mean(x, lengths, ch):
    """Mean over the *real* (non-padded) frames per sample."""
    out = np.empty(len(x))
    for i in range(len(x)):
        L = lengths[i] if lengths.ndim == 1 else lengths[i]
        out[i] = x[i, :L, ch].mean()
    return out

# Use validation set (smaller, faster)
t1, t2, lb, lens = arr['t1_vl'], arr['t2_vl'], arr['lb_vl'], arr['len_vl']
if lens.ndim == 1:
    L1, L2 = lens, lens
else:
    L1, L2 = lens[:, 0], lens[:, 1]

# Mean velocity / course per track (real frames only)
v1 = np.array([t1[i, :L1[i], VEL].mean() for i in range(len(t1))])
v2 = np.array([t2[i, :L2[i], VEL].mean() for i in range(len(t2))])
c1 = np.array([t1[i, :L1[i], COU].mean() for i in range(len(t1))])
c2 = np.array([t2[i, :L2[i], COU].mean() for i in range(len(t2))])

dv = np.abs(v1 - v2)
dc_raw = np.abs(c1 - c2)
# Course is in the normalized space, but circular structure broken — use raw diff
# (we just want to compare the *gap* between pos and neg).

def summary(name, vals_pos, vals_neg):
    log(f'  {name:25s}  pos: mean={vals_pos.mean():.4f} std={vals_pos.std():.4f}    '
        f'neg: mean={vals_neg.mean():.4f} std={vals_neg.std():.4f}    '
        f'ratio_neg/pos={vals_neg.mean()/(vals_pos.mean()+1e-9):.2f}x')

summary('|Δ mean velocity|',  dv[lb == 1], dv[lb == 0])
summary('|Δ mean course|  ',  dc_raw[lb == 1], dc_raw[lb == 0])

ks_v = ks_2samp(dv[lb == 1], dv[lb == 0])
ks_c = ks_2samp(dc_raw[lb == 1], dc_raw[lb == 0])
log(f'\n  KS  on |Δvel|: D={ks_v.statistic:.4f}  p={ks_v.pvalue:.4g}')
log(f'  KS  on |Δcou|: D={ks_c.statistic:.4f}  p={ks_c.pvalue:.4g}')

if ks_v.statistic < 0.30 and ks_c.statistic < 0.30:
    log('  ✓ Pos and neg are kinematically similar (Bug 1 fixed)')
elif ks_v.statistic < 0.50 and ks_c.statistic < 0.50:
    log('  ~ Some kinematic gap remains — moderate negatives')
else:
    log('  ✗ Negative samples are kinematically much more spread — leakage risk')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, p, n, title in [(axes[0], dv[lb == 1], dv[lb == 0], '|Δ mean velocity|'),
                         (axes[1], dc_raw[lb == 1], dc_raw[lb == 0], '|Δ mean course|')]:
    bins = np.linspace(0, np.percentile(np.concatenate([p, n]), 99), 50)
    ax.hist(p, bins=bins, alpha=0.6, label='pos', color='#1f77b4', density=True)
    ax.hist(n, bins=bins, alpha=0.6, label='neg', color='#d62728', density=True)
    ax.set_title(title); ax.set_xlabel('value'); ax.set_ylabel('density')
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_kinematic_similarity.png', dpi=140, bbox_inches='tight')
plt.close()
log(f'  Saved: {OUT_DIR}/02_kinematic_similarity.png')

# -----------------------------------------------------------------
# 4. Padding-region sanity check (Bug 3 fix)
# -----------------------------------------------------------------
section('4. Padding-region sanity  (Bug 3 fix)')
# After Bug 3 fix, frames beyond `length` should be exactly 0.
def check_pad(name, x, lengths, n_check=2000):
    n_check = min(n_check, len(x))
    bad = 0; nonzero_max = 0.
    for i in range(n_check):
        L = lengths[i] if lengths.ndim == 1 else lengths[i]
        if L < x.shape[1]:
            tail = x[i, L:]
            mx = np.abs(tail).max()
            nonzero_max = max(nonzero_max, mx)
            if mx > 1e-6:
                bad += 1
    log(f'  {name}: checked {n_check} samples, '
        f'frames-after-len with non-zero entries: {bad}  '
        f'(max abs in padding = {nonzero_max:.6g})')
    if bad == 0:
        log('    ✓ Padding is clean (Bug 3 fixed)')
    elif bad < n_check * 0.01:
        log('    ~ A few outliers but mostly clean')
    else:
        log('    ✗ Padding contains real signal — leakage risk')

check_pad('train track1', arr['t1_tr'], L1 if False else (arr['len_tr'][:, 0] if arr['len_tr'].ndim == 2 else arr['len_tr']))
check_pad('train track2', arr['t2_tr'], (arr['len_tr'][:, 1] if arr['len_tr'].ndim == 2 else arr['len_tr']))

# -----------------------------------------------------------------
# 5. Per-feature statistics & normalization
# -----------------------------------------------------------------
section('5. Per-feature normalization stats')
log(f'  scaler_mean : {arr["mean"]}')
log(f'  scaler_scale: {arr["scale"]}')

# Compute mean/std over real frames only
def real_feat_stats(x, lengths, name):
    feats = []
    L_arr = lengths[:, 0] if lengths.ndim == 2 else lengths
    L_arr = L_arr[:5000]    # sample
    for i in range(len(L_arr)):
        feats.append(x[i, :L_arr[i]])
    feats = np.concatenate(feats, axis=0)
    log(f'  {name} (real frames): mean={feats.mean(axis=0)}  std={feats.std(axis=0)}')

real_feat_stats(arr['t1_tr'], arr['len_tr'], 'train track1')
real_feat_stats(arr['t2_tr'], arr['len_tr'], 'train track2')

# -----------------------------------------------------------------
# 6. Trivial-separability stress test  (THE big one)
# -----------------------------------------------------------------
section('6. Trivial separability  (logistic regression on scalar features)')
log('  If logistic regression on simple (mean/std/diff) features gets')
log('  >> 50% accuracy, the dataset is too easy / has shortcut leakage.')
log('  A healthy dataset should show LR ≈ 60-70%, not 95%+.')

def hand_crafted(t1, t2, L1_, L2_):
    feats = []
    for i in range(len(t1)):
        a = t1[i, :L1_[i]]
        b = t2[i, :L2_[i]]
        ma, sa = a.mean(0), a.std(0)
        mb, sb = b.mean(0), b.std(0)
        feats.append(np.concatenate([ma, sa, mb, sb,
                                      ma - mb, np.abs(ma - mb),
                                      sa - sb, np.abs(sa - sb)]))
    return np.array(feats)

# subsample for speed
N = min(10000, len(arr['t1_tr']))
idx = np.random.RandomState(0).choice(len(arr['t1_tr']), N, replace=False)
len_tr2 = arr['len_tr']
L1_tr = len_tr2[idx, 0] if len_tr2.ndim == 2 else len_tr2[idx]
L2_tr = len_tr2[idx, 1] if len_tr2.ndim == 2 else len_tr2[idx]
X = hand_crafted(arr['t1_tr'][idx], arr['t2_tr'][idx], L1_tr, L2_tr)
y = arr['lb_tr'][idx]
X = StandardScaler().fit_transform(X)

log(f'  feature dim={X.shape[1]}  n={len(X)}')
clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0)
scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy', n_jobs=-1)
log(f'  3-fold LR accuracy: {scores.mean()*100:.2f}%  (per fold: {[f"{s*100:.2f}" for s in scores]})')

# Also: just length features alone
Xlen = np.stack([L1_tr, L2_tr,
                 np.abs(L1_tr - L2_tr),
                 np.maximum(L1_tr, L2_tr) / np.maximum(np.minimum(L1_tr, L2_tr), 1)], axis=1)
Xlen = StandardScaler().fit_transform(Xlen)
scores_l = cross_val_score(clf, Xlen, y, cv=3, scoring='accuracy', n_jobs=-1)
log(f'  3-fold LR on LENGTH-ONLY features: {scores_l.mean()*100:.2f}%  '
    f'(should be ≈50% if no length leakage)')

# Verdict
lr_acc = scores.mean()
if lr_acc < 0.72:
    log('  ✓ Dataset is appropriately hard (LR can\'t shortcut it)')
elif lr_acc < 0.85:
    log('  ~ Some easy structure exists — acceptable')
else:
    log('  ✗ Dataset is too easy — there is still shortcut leakage!')

# -----------------------------------------------------------------
# 7. Train / val distribution drift
# -----------------------------------------------------------------
section('7. Train vs Val distribution drift')
for ch, name in zip(range(4), ['lat','lon','vel','cou']):
    a = arr['t1_tr'][:5000, :, ch].flatten()
    b = arr['t1_vl'][:, :, ch].flatten()
    a = a[a != 0]; b = b[b != 0]
    ks = ks_2samp(a[:50000], b[:50000])
    log(f'  feature {name:>4}: KS D={ks.statistic:.4f}  p={ks.pvalue:.4g}')

# -----------------------------------------------------------------
# Save report
# -----------------------------------------------------------------
with open(f'{OUT_DIR}/quality_report.txt', 'w') as f:
    f.write('\n'.join(REPORT))
log(f'\nFull report saved to {OUT_DIR}/quality_report.txt')
