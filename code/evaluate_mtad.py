"""
MTAD 数据集评估脚本
====================
数据来源: 崔亚奇等, 基于全球AIS的多源航迹关联数据集(MTAD) V2, Science Data Bank, 2025
路径:     /home/yangcq/track_association/data/raw

目录结构
--------
    关联表/关联结果-<scene>.csv    每行一条观测航迹: (t_s, t_e, mmsi, source, batch)
    量测场景/场景-<scene>.csv       9001/9002 两源的带噪 AIS 观测, 按 (batch, source) 区分航迹
    真实场景/场景-<scene>.csv       同一 MMSI 的真值 AIS 轨迹

本脚本完成的评估
----------------
1. 扫描全部场景的关联表, 统计:
   - 每场景 9001/9002 各有多少批次
   - 每场景能构造的正样本对数 (跨源同 mmsi) 与负样本对数 (跨源不同 mmsi)
   - mmsi 拆分后缀含义统计
2. 统计量测航迹长度/采样率/特征范围
3. 抽样若干场景做噪声分析: 对齐 mmsi 的量测轨迹与真实轨迹, 计算位置/速度/航向误差
4. 可视化正负样本对 (英文标签, 蓝红统一配色)
5. 与 final_dataset (训练集) 做横向对比
6. 输出 markdown + csv 报告到 output/mtad_eval/
"""

import os
import re
import glob
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = '/home/yangcq/track_association/data/raw'
ASSOC_DIR = os.path.join(DATA_ROOT, '关联表')
MEAS_DIR  = os.path.join(DATA_ROOT, '量测场景')
TRUE_DIR  = os.path.join(DATA_ROOT, '真实场景')

OUT_DIR   = '/home/yangcq/track_association/output/mtad_eval'
FIG_DIR   = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

FINAL_DS  = '/home/yangcq/track_association/data/final_dataset'

BLUE = '#1f77b4'
RED  = '#d62728'
GREY = '#888888'


# ------------------------------------------------------------------
# util
# ------------------------------------------------------------------
def scene_id(path):
    m = re.search(r'-(\d+)\.csv$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_scenes():
    files = glob.glob(os.path.join(ASSOC_DIR, '关联结果-*.csv'))
    ids = sorted({scene_id(f) for f in files})
    return ids


def load_assoc(sid):
    return pd.read_csv(os.path.join(ASSOC_DIR, f'关联结果-{sid}.csv'))


def load_meas(sid):
    return pd.read_csv(os.path.join(MEAS_DIR, f'场景-{sid}.csv'))


def load_true(sid):
    return pd.read_csv(os.path.join(TRUE_DIR, f'场景-{sid}.csv'))


# MMSI 格式: <真实mmsi>-<子段号>-<未知>   关联表保留完整字符串, 真实场景只保留前两段
def mmsi_core(mmsi_str):
    """去掉关联表尾部的 '-0' 之类后缀, 返回与真实场景 MMSI 可比对的 key."""
    parts = str(mmsi_str).split('-')
    if len(parts) >= 2:
        return '-'.join(parts[:2])
    return str(mmsi_str)


# ------------------------------------------------------------------
# 1. 扫描关联表
# ------------------------------------------------------------------
def scan_assoc_tables(scene_ids):
    print(f'[1/5] Scanning {len(scene_ids)} 关联表 ...', flush=True)
    rows = []
    total_pos_pairs = 0
    total_neg_pairs_max = 0
    mmsi_suffix_counter = Counter()
    source_counter = Counter()

    for i, sid in enumerate(scene_ids):
        df = load_assoc(sid)
        src_counts = df['source'].value_counts().to_dict()
        n_9001 = int(src_counts.get(9001, 0))
        n_9002 = int(src_counts.get(9002, 0))

        df['mmsi_core'] = df['mmsi'].apply(mmsi_core)
        by_src = df.groupby('source')['mmsi_core'].apply(set)
        m_9001 = by_src.get(9001, set())
        m_9002 = by_src.get(9002, set())
        shared = m_9001 & m_9002

        pos_pairs = 0
        for m in shared:
            c1 = int(((df['source'] == 9001) & (df['mmsi_core'] == m)).sum())
            c2 = int(((df['source'] == 9002) & (df['mmsi_core'] == m)).sum())
            pos_pairs += c1 * c2
        neg_pairs_max = n_9001 * n_9002 - pos_pairs

        total_pos_pairs += pos_pairs
        total_neg_pairs_max += neg_pairs_max

        for m in df['mmsi']:
            parts = str(m).split('-')
            if len(parts) >= 3:
                mmsi_suffix_counter[parts[-1]] += 1
        for s in df['source']:
            source_counter[int(s)] += 1

        rows.append({
            'scene': sid,
            'n_tracks': len(df),
            'n_9001': n_9001,
            'n_9002': n_9002,
            'n_shared_mmsi': len(shared),
            'n_mmsi_9001_only': len(m_9001 - m_9002),
            'n_mmsi_9002_only': len(m_9002 - m_9001),
            'pos_pairs': pos_pairs,
            'neg_pairs_max': neg_pairs_max,
        })
        if (i + 1) % 500 == 0:
            print(f'    scanned {i + 1}/{len(scene_ids)}', flush=True)

    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(OUT_DIR, 'per_scene_stats.csv'), index=False)

    summary = {
        'n_scenes': len(scene_ids),
        'tracks_total': int(stats['n_tracks'].sum()),
        'tracks_per_scene_mean': float(stats['n_tracks'].mean()),
        'tracks_per_scene_std':  float(stats['n_tracks'].std()),
        'tracks_per_scene_min':  int(stats['n_tracks'].min()),
        'tracks_per_scene_max':  int(stats['n_tracks'].max()),
        'n_9001_total': int(stats['n_9001'].sum()),
        'n_9002_total': int(stats['n_9002'].sum()),
        'shared_mmsi_per_scene_mean': float(stats['n_shared_mmsi'].mean()),
        'pos_pairs_total': int(total_pos_pairs),
        'neg_pairs_max_total': int(total_neg_pairs_max),
        'pos_over_total': total_pos_pairs / max(1, total_pos_pairs + total_neg_pairs_max),
        'mmsi_suffix_top': dict(mmsi_suffix_counter.most_common(10)),
        'source_counter': dict(source_counter),
    }
    with open(os.path.join(OUT_DIR, 'summary_assoc.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 分布图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(stats['n_tracks'], bins=40, color=BLUE, alpha=0.85)
    axes[0].set_title('Tracks per scene')
    axes[0].set_xlabel('# tracks'); axes[0].set_ylabel('# scenes')

    axes[1].hist(stats['n_shared_mmsi'], bins=40, color=RED, alpha=0.85)
    axes[1].set_title('Shared MMSI (9001 & 9002) per scene')
    axes[1].set_xlabel('# shared MMSI')

    axes[2].scatter(stats['pos_pairs'], stats['neg_pairs_max'], s=6, alpha=0.5, color=GREY)
    axes[2].set_title('Pos vs Max-Neg pairs per scene')
    axes[2].set_xlabel('# positive pairs'); axes[2].set_ylabel('# max negative pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'assoc_stats.png'), dpi=140)
    plt.close()

    return stats, summary


# ------------------------------------------------------------------
# 2. 量测航迹长度 / 采样率 / 特征范围
# ------------------------------------------------------------------
def analyze_measurements(scene_ids, sample_k=200, seed=0):
    print(f'[2/5] Analyzing measurements on {sample_k} sampled scenes ...', flush=True)
    rng = np.random.RandomState(seed)
    picks = rng.choice(scene_ids, size=min(sample_k, len(scene_ids)), replace=False)

    track_lengths = []
    dt_all = []
    lat_all, lon_all, vel_all, cou_all = [], [], [], []
    duration_sec = []

    for sid in picks:
        df = load_meas(int(sid))
        grouped = df.groupby(['batch', 'source'])
        for (_, _), g in grouped:
            g = g.sort_values('time')
            track_lengths.append(len(g))
            duration_sec.append(float(g['time'].max() - g['time'].min()))
            if len(g) > 1:
                dt_all.extend(np.diff(g['time'].values).tolist())
        lat_all.extend(df['lat'].values.tolist())
        lon_all.extend(df['lon'].values.tolist())
        vel_all.extend(df['vel'].values.tolist())
        cou_all.extend(df['cou'].values.tolist())

    track_lengths = np.array(track_lengths)
    dt_all = np.array(dt_all)

    def describe(a, name):
        a = np.asarray(a)
        return {
            'name': name, 'n': int(len(a)),
            'mean': float(np.mean(a)), 'std': float(np.std(a)),
            'min': float(np.min(a)), 'p25': float(np.percentile(a, 25)),
            'p50': float(np.percentile(a, 50)), 'p75': float(np.percentile(a, 75)),
            'p95': float(np.percentile(a, 95)), 'max': float(np.max(a)),
        }

    feats = {
        'track_length_steps': describe(track_lengths, 'track_length_steps'),
        'track_duration_sec': describe(duration_sec, 'track_duration_sec'),
        'inter_sample_dt_sec': describe(dt_all, 'inter_sample_dt_sec'),
        'lat':  describe(lat_all, 'lat'),
        'lon':  describe(lon_all, 'lon'),
        'vel':  describe(vel_all, 'vel'),
        'cou':  describe(cou_all, 'cou'),
    }

    pd.DataFrame(feats).T.to_csv(os.path.join(OUT_DIR, 'feature_stats.csv'))

    # 长度分布图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(track_lengths, bins=50, color=BLUE, alpha=0.85)
    axes[0].set_title(f'Track length (steps)  median={int(np.median(track_lengths))}')
    axes[0].set_xlabel('# time steps'); axes[0].set_ylabel('# tracks')

    axes[1].hist(np.array(duration_sec) / 60.0, bins=50, color=RED, alpha=0.85)
    axes[1].set_title('Track duration (minutes)')
    axes[1].set_xlabel('minutes')

    dt_clip = dt_all[dt_all < np.percentile(dt_all, 99)]
    axes[2].hist(dt_clip, bins=60, color=GREY, alpha=0.85)
    axes[2].set_title(f'Inter-sample dt (s)  median={np.median(dt_all):.2f}')
    axes[2].set_xlabel('seconds (clipped 99%)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'measurement_len_dt.png'), dpi=140)
    plt.close()

    # 特征分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, data, name in zip(axes.ravel(),
                              [lat_all, lon_all, vel_all, cou_all],
                              ['lat (deg)', 'lon (deg)', 'vel (knots?)', 'cou (deg)']):
        arr = np.asarray(data)
        clipped = arr[(arr >= np.percentile(arr, 0.5)) & (arr <= np.percentile(arr, 99.5))]
        ax.hist(clipped, bins=80, color=BLUE, alpha=0.8)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'feature_hist.png'), dpi=140)
    plt.close()

    return feats


# ------------------------------------------------------------------
# 3. 噪声分析: 量测 vs 真实
# ------------------------------------------------------------------
def noise_analysis(scene_ids, sample_k=50, seed=1):
    print(f'[3/5] Noise analysis on {sample_k} scenes ...', flush=True)
    rng = np.random.RandomState(seed)
    picks = rng.choice(scene_ids, size=min(sample_k, len(scene_ids)), replace=False)

    pos_errs = []   # great-circle distance (m) between meas & true
    vel_errs = []   # knot diff
    cou_errs = []   # deg diff (mod 180)

    def gc_dist(lat1, lon1, lat2, lon2):
        R = 6371000.0
        la1 = np.radians(lat1); la2 = np.radians(lat2)
        dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    for sid in picks:
        assoc = load_assoc(int(sid))
        meas  = load_meas(int(sid))
        true  = load_true(int(sid))
        # 关联表 mmsi 带 3 段后缀, 真实场景只带 2 段
        assoc['mmsi_core'] = assoc['mmsi'].apply(mmsi_core)

        true_by_mmsi = {m: g.sort_values('time') for m, g in true.groupby('MMSI')}

        for _, row in assoc.iterrows():
            key = row['mmsi_core']
            if key not in true_by_mmsi:
                continue
            t_df = true_by_mmsi[key]
            m_df = meas[(meas['batch'] == row['batch']) & (meas['source'] == row['source'])]
            if len(m_df) == 0 or len(t_df) == 0:
                continue
            m_df = m_df.sort_values('time')
            # 对每个量测时间做线性插值求真值
            t_time = t_df['time'].values
            for col_m, col_t, bucket in [
                ('lat', 'lat', pos_errs),       # 占位, 稍后 gc_dist 重算
                ('vel', 'vel', vel_errs),
                ('cou', 'cou', cou_errs),
            ]:
                pass
            tm = m_df['time'].values
            lat_true = np.interp(tm, t_time, t_df['lat'].values)
            lon_true = np.interp(tm, t_time, t_df['lon'].values)
            vel_true = np.interp(tm, t_time, t_df['vel'].values)
            cou_true = np.interp(tm, t_time, t_df['cou'].values)

            valid = (tm >= t_time.min()) & (tm <= t_time.max())
            if not np.any(valid):
                continue
            tm    = tm[valid]
            lat_m = m_df['lat'].values[valid]
            lon_m = m_df['lon'].values[valid]
            vel_m = m_df['vel'].values[valid]
            cou_m = m_df['cou'].values[valid]
            lat_t = lat_true[valid]; lon_t = lon_true[valid]
            vel_t = vel_true[valid]; cou_t = cou_true[valid]

            d = gc_dist(lat_m, lon_m, lat_t, lon_t)
            pos_errs.extend(d.tolist())
            vel_errs.extend((vel_m - vel_t).tolist())
            diff = np.abs(cou_m - cou_t) % 360
            diff = np.minimum(diff, 360 - diff)
            cou_errs.extend(diff.tolist())

    res = {}
    for name, arr in [('position_error_m', pos_errs),
                      ('velocity_diff_knot', vel_errs),
                      ('course_diff_deg', cou_errs)]:
        a = np.array(arr)
        res[name] = {
            'n': int(len(a)),
            'mean': float(np.mean(a)), 'std': float(np.std(a)),
            'p50': float(np.percentile(a, 50)),
            'p95': float(np.percentile(a, 95)),
            'max': float(np.max(a)),
        }
    with open(os.path.join(OUT_DIR, 'noise_stats.json'), 'w') as f:
        json.dump(res, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, arr, title, color in zip(
        axes,
        [pos_errs, vel_errs, cou_errs],
        ['Position error (m, great-circle)',
         'Velocity diff (knots)',
         'Course diff (deg, wrapped)'],
        [BLUE, RED, GREY]):
        a = np.array(arr)
        clip = a[(a >= np.percentile(a, 0.5)) & (a <= np.percentile(a, 99.5))]
        ax.hist(clip, bins=80, color=color, alpha=0.85)
        ax.set_title(f'{title}\nmedian={np.percentile(a, 50):.2f}, p95={np.percentile(a, 95):.2f}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'noise_hist.png'), dpi=140)
    plt.close()

    return res


# ------------------------------------------------------------------
# 4. 可视化: 一个正样本对 + 一个负样本对
# ------------------------------------------------------------------
def _pick_pair(assoc):
    """在关联表内选一个正样本 (同 mmsi_core 跨源) 和一个负样本 (不同 mmsi_core 跨源)"""
    assoc = assoc.copy()
    assoc['mmsi_core'] = assoc['mmsi'].apply(mmsi_core)
    a1 = assoc[assoc['source'] == 9001]
    a2 = assoc[assoc['source'] == 9002]
    pos = None
    shared = set(a1['mmsi_core']) & set(a2['mmsi_core'])
    for m in shared:
        r1 = a1[a1['mmsi_core'] == m].iloc[0]
        r2 = a2[a2['mmsi_core'] == m].iloc[0]
        pos = (r1, r2, m)
        break
    neg = None
    if len(a1) and len(a2):
        for _, r1 in a1.iterrows():
            for _, r2 in a2.iterrows():
                if r1['mmsi_core'] != r2['mmsi_core']:
                    neg = (r1, r2, None)
                    break
            if neg is not None:
                break
    return pos, neg


def _plot_pair(ax, meas, r1, r2, title, label_prefix=''):
    t1 = meas[(meas['batch'] == r1['batch']) & (meas['source'] == r1['source'])].sort_values('time')
    t2 = meas[(meas['batch'] == r2['batch']) & (meas['source'] == r2['source'])].sort_values('time')
    ax.plot(t1['lon'], t1['lat'], '-o', color=BLUE, markersize=3, linewidth=1.2,
            label=f"{label_prefix}9001 batch {int(r1['batch'])}")
    ax.plot(t2['lon'], t2['lat'], '-s', color=RED,  markersize=3, linewidth=1.2,
            label=f"{label_prefix}9002 batch {int(r2['batch'])}")
    ax.set_xlabel('Longitude (deg)'); ax.set_ylabel('Latitude (deg)')
    ax.set_title(title); ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)


def visualize_pairs(scene_ids, n_scenes=3, seed=2):
    print(f'[4/5] Visualizing pairs in {n_scenes} scenes ...', flush=True)
    rng = np.random.RandomState(seed)
    picks = rng.choice(scene_ids, size=min(n_scenes, len(scene_ids)), replace=False)

    for sid in picks:
        assoc = load_assoc(int(sid))
        meas  = load_meas(int(sid))
        pos, neg = _pick_pair(assoc)
        if pos is None or neg is None:
            continue
        r1p, r2p, mmsi = pos
        r1n, r2n, _ = neg

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        _plot_pair(axes[0], meas, r1p, r2p,
                   f'Scene {sid} | Positive pair (same MMSI {mmsi})')
        _plot_pair(axes[1], meas, r1n, r2n,
                   f'Scene {sid} | Negative pair (different MMSI)')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'pair_scene_{sid}.png'), dpi=140)
        plt.close()

        # 叠加真实轨迹
        true = load_true(int(sid))
        fig, ax = plt.subplots(figsize=(8, 6))
        t1 = meas[(meas['batch'] == r1p['batch']) & (meas['source'] == r1p['source'])].sort_values('time')
        t2 = meas[(meas['batch'] == r2p['batch']) & (meas['source'] == r2p['source'])].sort_values('time')
        tgt = true[true['MMSI'] == mmsi].sort_values('time')
        ax.plot(tgt['lon'], tgt['lat'], '-', color='black', linewidth=2.5, label='Ground truth')
        ax.plot(t1['lon'], t1['lat'], '-o', color=BLUE, markersize=3, linewidth=1, label='9001 meas')
        ax.plot(t2['lon'], t2['lat'], '-s', color=RED,  markersize=3, linewidth=1, label='9002 meas')
        ax.set_xlabel('Longitude (deg)'); ax.set_ylabel('Latitude (deg)')
        ax.set_title(f'Scene {sid} | MMSI {mmsi} : measurements vs ground truth')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'pair_vs_truth_scene_{sid}.png'), dpi=140)
        plt.close()


# ------------------------------------------------------------------
# 5. 与 final_dataset 对比
# ------------------------------------------------------------------
def compare_with_final_dataset():
    print('[5/5] Compare with final_dataset ...', flush=True)
    info = {'final_dataset_present': os.path.exists(FINAL_DS)}
    if not info['final_dataset_present']:
        return info
    for split in ['train', 'val', 'test']:
        t1_path = os.path.join(FINAL_DS, f'track1_{split}.npy')
        lab_path = os.path.join(FINAL_DS, f'labels_{split}.npy')
        len_path = os.path.join(FINAL_DS, f'lengths_{split}.npy')
        if not os.path.exists(t1_path):
            continue
        t1 = np.load(t1_path)
        lab = np.load(lab_path) if os.path.exists(lab_path) else None
        lens = np.load(len_path) if os.path.exists(len_path) else None
        info[split] = {
            'shape': list(t1.shape),
            'max_len_configured': int(t1.shape[1]),
            'pos_ratio': float(np.mean(lab == 1)) if lab is not None else None,
            'length_median': float(np.median(lens)) if lens is not None else None,
            'length_max': int(np.max(lens)) if lens is not None else None,
            'length_min': int(np.min(lens)) if lens is not None else None,
        }
    with open(os.path.join(OUT_DIR, 'compare_final_dataset.json'), 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


# ------------------------------------------------------------------
# report
# ------------------------------------------------------------------
def build_report(assoc_summary, feat_stats, noise_stats, final_info):
    lines = []
    lines.append('# MTAD Dataset Evaluation Report')
    lines.append('')
    lines.append('Source: 崔亚奇等, 基于全球 AIS 的多源航迹关联数据集 (MTAD) V2, Science Data Bank, 2025')
    lines.append('Root:   `/home/yangcq/track_association/data/raw`')
    lines.append('')
    lines.append('## 1. Association table overview')
    for k, v in assoc_summary.items():
        if isinstance(v, float):
            lines.append(f'- **{k}**: {v:.6f}')
        else:
            lines.append(f'- **{k}**: {v}')
    lines.append('')
    lines.append('## 2. Measurement statistics (sampled scenes)')
    for k, d in feat_stats.items():
        lines.append(f'- **{k}**: n={d["n"]}, mean={d["mean"]:.3f}, std={d["std"]:.3f}, '
                     f'min={d["min"]:.3f}, p50={d["p50"]:.3f}, p95={d["p95"]:.3f}, max={d["max"]:.3f}')
    lines.append('')
    lines.append('## 3. Noise analysis (measurement vs ground truth)')
    for k, d in noise_stats.items():
        lines.append(f'- **{k}**: n={d["n"]}, mean={d["mean"]:.3f}, '
                     f'std={d["std"]:.3f}, p50={d["p50"]:.3f}, p95={d["p95"]:.3f}, max={d["max"]:.3f}')
    lines.append('')
    lines.append('## 4. Comparison with `final_dataset` used for training')
    lines.append('```')
    lines.append(json.dumps(final_info, indent=2, ensure_ascii=False))
    lines.append('```')
    lines.append('')
    lines.append('## 5. Usage suggestion')
    lines.append('- Positive pairs for training: all (batch, 9001) x (batch, 9002) tuples that share the same MMSI core inside a scene.')
    lines.append('- Negative pairs: sample cross-source tuples with different MMSI core from the same scene '
                 '(recommended to roughly balance or to use class-weighting like the current pipeline).')
    lines.append('- Features per step: `lat, lon, vel, cou` — same 4-D layout as the current model input, so weights transfer in principle.')
    lines.append('- Sequences are variable length with much larger max length than the current 350-step cap: '
                 'either re-pad to a new max, downsample, or window the measurements before feeding CNN-BiMamba.')
    lines.append('- Ground-truth interpolation from `真实场景` allows noise-error visualization and physical sanity checks.')
    lines.append('')

    with open(os.path.join(OUT_DIR, 'REPORT.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Report written -> {os.path.join(OUT_DIR, "REPORT.md")}')


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit_scenes', type=int, default=0,
                    help='only process first N scenes (0 = all 5000)')
    ap.add_argument('--sample_meas', type=int, default=200)
    ap.add_argument('--sample_noise', type=int, default=50)
    ap.add_argument('--viz_scenes', type=int, default=3)
    args = ap.parse_args()

    ids = list_scenes()
    if args.limit_scenes > 0:
        ids = ids[:args.limit_scenes]
    print(f'Total scenes found: {len(ids)}', flush=True)

    _, assoc_summary = scan_assoc_tables(ids)
    feat_stats = analyze_measurements(ids, sample_k=args.sample_meas)
    noise_stats = noise_analysis(ids, sample_k=args.sample_noise)
    visualize_pairs(ids, n_scenes=args.viz_scenes)
    final_info = compare_with_final_dataset()
    build_report(assoc_summary, feat_stats, noise_stats, final_info)

    print('Done. Outputs under:', OUT_DIR)


if __name__ == '__main__':
    main()
