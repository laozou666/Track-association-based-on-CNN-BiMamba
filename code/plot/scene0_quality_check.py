"""
Scene-0 simulation data quality check
Style follows track_fig reference images.
Run: conda run -n track_association python3 code/plot/scene0_quality_check.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random, os, sys

# --- Force DejaVu Sans to prevent CJK font fallback rendering boxes ---
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

random.seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.simulation_track6 import (
    simulate_track_with_observer, shift_track_perfect,
    get_kinematically_similar_pairs, MIN_TRACK_POINTS
)

OUT_DIR = '/home/yangcq/track_association/output/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ===== Load scene-0 =====
df = pd.read_csv('/home/yangcq/track_association/data/raw/真实场景/场景-0.csv')
track_pool_raw = [
    g.sort_values('time').reset_index(drop=True)
    for _, g in df.groupby('MMSI')
    if len(g) >= MIN_TRACK_POINTS
]
random.shuffle(track_pool_raw)

# ===== Positive samples (filter static ships) =====
n_pos = len(track_pool_raw) // 2
pos_samples = []
for tr in track_pool_raw[:n_pos]:
    if tr['vel'].mean() < 1.0:
        continue
    if (tr['lat'].max() - tr['lat'].min()) < 0.003:
        continue
    t1, l1 = simulate_track_with_observer(tr, 9001)
    t2, l2 = simulate_track_with_observer(tr, 9002)
    pos_samples.append({'truth': tr, 't1': t1, 't2': t2,
                        'l1': l1, 'l2': l2, 'mmsi': str(tr['MMSI'].iloc[0])})

# ===== Negative samples (kinematic-constrained) =====
neg_pool = [
    tr for tr in track_pool_raw[n_pos:]
    if tr['vel'].mean() >= 1.2 and (tr['lat'].max() - tr['lat'].min()) >= 0.003
]
neg_pairs_raw = get_kinematically_similar_pairs(neg_pool)
neg_samples = []
for trA, trB in neg_pairs_raw:
    t1, l1 = simulate_track_with_observer(trA, 9001)
    t2, l2 = simulate_track_with_observer(trB, 9002)
    t2 = shift_track_perfect(t2, t1)
    neg_samples.append({
        'trA': trA, 'trB': trB, 't1': t1, 't2': t2,
        'l1': l1, 'l2': l2,
        'mmsiA': str(trA['MMSI'].iloc[0]), 'mmsiB': str(trB['MMSI'].iloc[0])
    })

print(f"Positive: {len(pos_samples)}   Negative: {len(neg_samples)}")


def draw_track_pair(ax, t1, t2, title, label='POSITIVE'):
    """Draw one track pair in the reference style."""
    ax.scatter(t1['lon'], t1['lat'], s=14, color='#1f77b4',
               alpha=0.85, zorder=3, label='Track 1 (9001)')
    ax.scatter(t2['lon'], t2['lat'], s=14, color='#d62728',
               alpha=0.85, zorder=3, label='Track 2 (9002)')

    # Start markers (green circle)
    ax.scatter(t1['lon'].iloc[0], t1['lat'].iloc[0],
               s=120, color='green', zorder=5, label='Start')
    ax.scatter(t2['lon'].iloc[0], t2['lat'].iloc[0],
               s=120, color='green', zorder=5)

    # End markers (red X)
    ax.scatter(t1['lon'].iloc[-1], t1['lat'].iloc[-1],
               s=180, color='red', marker='x', linewidths=2.5, zorder=5, label='End')
    ax.scatter(t2['lon'].iloc[-1], t2['lat'].iloc[-1],
               s=180, color='red', marker='x', linewidths=2.5, zorder=5)

    # Label badge in upper-left corner
    badge_color = '#2ca02c' if label == 'POSITIVE' else '#d62728'
    ax.text(0.03, 0.97, f'[{label}]', transform=ax.transAxes,
            fontsize=10, fontweight='bold', color='white',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, alpha=0.85))

    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8, loc='upper right', markerscale=1.4)


# ===== Figure: 3 positive + 3 negative, 2-column layout =====
N = 3   # pairs per class to show
fig, axes = plt.subplots(N, 2, figsize=(14, 5 * N))
fig.suptitle('Scene-0  |  Simulated Track Pairs Quality Check\n'
             'Left: POSITIVE (same ship)      Right: NEGATIVE (different ships, kinematically similar)',
             fontsize=12, fontweight='bold', y=1.01)

for row in range(N):
    # --- left column: positive ---
    ax = axes[row, 0]
    if row < len(pos_samples):
        s = pos_samples[row]
        t1, t2 = s['t1'], s['t2']
        dv = abs(t1['vel'].mean() - t2['vel'].mean())
        dc = abs(t1['cou'].mean() - t2['cou'].mean())
        dc = min(dc, 360 - dc)
        title = (f'Positive #{row+1}  |  MMSI: {s["mmsi"]}\n'
                 f'Track1 n={s["l1"]}   Track2 n={s["l2"]}   '
                 f'|Dvel|={dv:.2f}kn   |Dcou|={dc:.1f}deg')
        draw_track_pair(ax, t1, t2, title, label='POSITIVE')
    else:
        ax.axis('off')

    # --- right column: negative ---
    ax = axes[row, 1]
    if row < len(neg_samples):
        s = neg_samples[row]
        t1, t2 = s['t1'], s['t2']
        dv = abs(s['trA']['vel'].mean() - s['trB']['vel'].mean())
        dc = abs(s['trA']['cou'].mean() - s['trB']['cou'].mean())
        dc = min(dc, 360 - dc)
        title = (f'Negative #{row+1}  |  ShipA: {s["mmsiA"][:12]}  /  ShipB: {s["mmsiB"][:12]}\n'
                 f'Track1 n={s["l1"]}   Track2 n={s["l2"]}   '
                 f'|Dvel|={dv:.2f}kn   |Dcou|={dc:.1f}deg')
        draw_track_pair(ax, t1, t2, title, label='NEGATIVE')
    else:
        ax.axis('off')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'scene0_track_pairs.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
