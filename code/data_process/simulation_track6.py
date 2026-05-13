"""
simulation_track6.py  —— 修复版数据仿真脚本

Bug修复说明：
  Bug1（最致命）：负样本过于简单
    旧版：仅按轨迹长度配对，导致负样本中两艘船的航速/航向差异极大
         （实测：负样本平均航向差是正样本的13.9倍，ANN用均值就能达到99%准确率）
    修复：引入运动学约束配对，要求|Δvel| < VEL_THRESH 且 |Δcou| < COU_THRESH，
         生成运动学相似的困难负样本

  Bug2：正负样本有效长度分布严重不对称
    旧版：负样本将两条轨迹截断至相同长度，正样本不截断
         → 正样本平均长度≈97, 负样本≈52，padding比例成为额外判别信号
    修复：负样本不再强制等长截断，与正样本一致，保留各自的自然长度

  Bug3：归一化时短轨迹的零padding区域被错误变换
    旧版：apply_scaler用max(l1,l2)对两条轨迹同时变换，导致较短轨迹的
         零padding被scaler变换为非零异常值（如lat=0→normalized≈-2.9）
    修复：分别记录l1、l2，各自只对真实数据段做scaler变换
"""

import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore")

# ===================== 配置区 =====================
REAL_SCENE_ROOT = "/home/yangcq/track_association/data/raw/真实场景/"
OUTPUT_ROOT = "/home/yangcq/track_association/data/final_dataset"
CSV_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "simulated_scenes_csv")
TEST_CSV_ROOT = os.path.join(OUTPUT_ROOT, "test_set_csvs")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(CSV_OUTPUT_ROOT, exist_ok=True)

# 航迹参数
MAX_SEQ_LEN = 350
MIN_TRACK_POINTS = 25

# 场景范围
START_SCENE = 0
END_SCENE = 4999

# 随机种子
random.seed(42)
np.random.seed(42)

# 传感器参数
OBSERVER_PARAMS = {
    9001: {"name": "Radar_A", "bias_lon": 0.0015, "bias_lat": -0.0012,
           "noise_pos": 0.00035, "noise_vel": 1.0, "noise_cou": 6.0,
           "freq_ratio": 1.0, "dropout_rate": 0.18},
    9002: {"name": "Radar_B", "bias_lon": -0.002, "bias_lat": 0.0018,
           "noise_pos": 0.0005, "noise_vel": 1.5, "noise_cou": 9.0,
           "freq_ratio": 0.75, "dropout_rate": 0.25}
}

# ===================== 运动学相似度约束（Bug1修复）=====================
# 负样本配对时，要求两条真实航迹的平均速度差和平均航向差均在阈值内
VEL_THRESH = 3.0    # 节，速度差阈值
COU_THRESH = 45.0   # 度，航向差阈值（考虑360°循环）


# ===================== 核心仿真函数（不变）=====================
def simulate_track_with_observer(truth_track, source_id):
    params = OBSERVER_PARAMS[source_id]
    track = truth_track.copy().sort_values("time").reset_index(drop=True)

    original_len = len(track)
    if np.random.rand() < 0.6:
        start_offset = np.random.randint(0, int(original_len * 0.25))
        end_offset = np.random.randint(0, int(original_len * 0.15))
        track = track.iloc[start_offset: original_len - end_offset].reset_index(drop=True)

    target_len = int(len(track) * params["freq_ratio"])
    target_len = max(MIN_TRACK_POINTS, target_len)
    if len(track) != target_len:
        idx_old = np.linspace(0, len(track) - 1, len(track))
        idx_new = np.linspace(0, len(track) - 1, target_len)
        track_new = pd.DataFrame()
        track_new["lat"] = np.interp(idx_new, idx_old, track["lat"].values)
        track_new["lon"] = np.interp(idx_new, idx_old, track["lon"].values)
        track_new["vel"] = np.interp(idx_new, idx_old, track["vel"].values)
        track_new["cou"] = np.interp(idx_new, idx_old, track["cou"].values)
        track_new["time"] = np.linspace(track["time"].iloc[0], track["time"].iloc[-1], target_len)
        track_new["MMSI"] = track["MMSI"].iloc[0]
        track = track_new

    if np.random.rand() < 0.7:
        n_drop = int(len(track) * params["dropout_rate"])
        if n_drop > 0:
            drop_idx = np.random.choice(len(track), n_drop, replace=False)
            track = track.drop(drop_idx).reset_index(drop=True)

    track["lon"] += params["bias_lon"]
    track["lat"] += params["bias_lat"]
    t = np.linspace(0, 2 * np.pi, len(track))
    lon_noise = (np.cumsum(np.random.normal(0, params["noise_pos"] * 0.3, len(track))) * 0.3
                 + np.sin(t * np.random.uniform(1, 3)) * params["noise_pos"] * 1.5)
    lat_noise = (np.cumsum(np.random.normal(0, params["noise_pos"] * 0.3, len(track))) * 0.3
                 + np.cos(t * np.random.uniform(1, 3)) * params["noise_pos"] * 1.5)
    track["lon"] += lon_noise
    track["lat"] += lat_noise
    track["vel"] += np.random.normal(0, params["noise_vel"], len(track))
    track["cou"] = (track["cou"] + np.random.normal(0, params["noise_cou"], len(track))) % 360

    return track, len(track)


def shift_track_perfect(track, target_track):
    off_lon = target_track["lon"].mean() - track["lon"].mean()
    off_lat = target_track["lat"].mean() - track["lat"].mean()
    track = track.copy()
    track["lon"] += off_lon
    track["lat"] += off_lat
    return track


def pad_or_truncate_track(track):
    """截断或用零补齐到 MAX_SEQ_LEN，返回(array, 真实有效长度)"""
    features = track[["lat", "lon", "vel", "cou"]].values.astype(np.float32)
    seq_len = len(features)
    if seq_len >= MAX_SEQ_LEN:
        return features[:MAX_SEQ_LEN], MAX_SEQ_LEN
    pad_len = MAX_SEQ_LEN - seq_len
    padded = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
    return padded, seq_len


def _cou_diff(a, b):
    """计算航向差，考虑360°循环"""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


# ===================== Bug1修复：运动学相似负样本配对 =====================
def get_kinematically_similar_pairs(track_pool):
    """
    在运动学约束下配对不同船只，生成困难负样本。
    要求：|Δvel| < VEL_THRESH  AND  |Δcou| < COU_THRESH
    若约束内无合适配对，放宽到2倍阈值，确保尽量生成足够数量的负样本。
    """
    pairs, used = [], set()
    vel_means = [tr["vel"].mean() for tr in track_pool]
    cou_means = [tr["cou"].mean() for tr in track_pool]

    for threshold_mult in [1.0, 2.0]:
        for i in range(len(track_pool)):
            if i in used:
                continue
            best_j, best_score = -1, float("inf")
            for j in range(i + 1, len(track_pool)):
                if j in used:
                    continue
                dv = abs(vel_means[i] - vel_means[j])
                dc = _cou_diff(cou_means[i], cou_means[j])
                if dv <= VEL_THRESH * threshold_mult and dc <= COU_THRESH * threshold_mult:
                    score = dv + dc * 0.1
                    if score < best_score:
                        best_score, best_j = score, j
            if best_j != -1:
                pairs.append((track_pool[i], track_pool[best_j]))
                used.add(i)
                used.add(best_j)

    return pairs


def process_single_scene(scene_id):
    input_path = os.path.join(REAL_SCENE_ROOT, f"场景-{scene_id}.csv")
    if not os.path.exists(input_path):
        return []

    try:
        df = pd.read_csv(input_path)
        track_pool = [
            g.sort_values("time").reset_index(drop=True)
            for _, g in df.groupby("MMSI")
            if len(g) >= MIN_TRACK_POINTS
        ]
        random.shuffle(track_pool)

        if len(track_pool) < 2:
            return []

        samples = []
        batch_id = 0

        # ---- 正样本 ----
        n_pos = len(track_pool) // 2
        for tr in track_pool[:n_pos]:
            if tr['vel'].mean() < 1.0:
                continue
            if (tr['lat'].max() - tr['lat'].min()) < 0.003:
                continue
            t1, l1 = simulate_track_with_observer(tr, 9001)
            t2, l2 = simulate_track_with_observer(tr, 9002)
            t1["source"] = 9001
            t2["source"] = 9002
            f1, real_l1 = pad_or_truncate_track(t1)
            f2, real_l2 = pad_or_truncate_track(t2)
            # 每个样本保存 (npy数组, npy数组, label, l1, l2, scene_id, batch_id, t1_df, t2_df)
            samples.append((f1, f2, 1, real_l1, real_l2, scene_id, batch_id,
                            t1[["lat", "lon", "vel", "cou", "time", "MMSI", "source"]].copy(),
                            t2[["lat", "lon", "vel", "cou", "time", "MMSI", "source"]].copy()))
            batch_id += 1

        # ---- 负样本（运动学约束配对）----
        neg_pool = [
            tr for tr in track_pool[n_pos:]
            if tr["vel"].mean() >= 1.2
            and (tr["lat"].max() - tr["lat"].min()) >= 0.003
        ]
        neg_pairs = get_kinematically_similar_pairs(neg_pool)

        for trA, trB in neg_pairs:
            t1, l1 = simulate_track_with_observer(trA, 9001)
            t2, l2 = simulate_track_with_observer(trB, 9002)
            t2 = shift_track_perfect(t2, t1)
            t1["source"] = 9001
            t2["source"] = 9002
            f1, real_l1 = pad_or_truncate_track(t1)
            f2, real_l2 = pad_or_truncate_track(t2)
            samples.append((f1, f2, 0, real_l1, real_l2, scene_id, batch_id,
                            t1[["lat", "lon", "vel", "cou", "time", "MMSI", "source"]].copy(),
                            t2[["lat", "lon", "vel", "cou", "time", "MMSI", "source"]].copy()))
            batch_id += 1

        return samples

    except Exception:
        return []


def main():
    N_WORKERS = min(64, cpu_count())   # 用最多64核，避免IO争抢
    scene_ids = list(range(START_SCENE, END_SCENE + 1))

    print("=" * 60)
    print(f" Processing scenes: {START_SCENE} ~ {END_SCENE}")
    print(f" Workers: {N_WORKERS}  (available CPUs: {cpu_count()})")
    print("=" * 60)

    all_samples = []
    chunk = 500  # 每批打印一次进度

    with Pool(processes=N_WORKERS) as pool:
        for i in range(0, len(scene_ids), chunk):
            batch_ids = scene_ids[i: i + chunk]
            results   = pool.map(process_single_scene, batch_ids)
            for samples in results:
                if samples:
                    all_samples.extend(samples)
            print(f"  Scenes {batch_ids[0]}~{batch_ids[-1]} done | "
                  f"total samples: {len(all_samples)}")

    if not all_samples:
        print("No samples generated.")
        return

    print(f"\nTotal samples: {len(all_samples)}")

    # ---- Extract arrays ----
    print("\n========== Normalization ==========")
    t1_data   = np.array([s[0] for s in all_samples], dtype=np.float32)
    t2_data   = np.array([s[1] for s in all_samples], dtype=np.float32)
    labels    = np.array([s[2] for s in all_samples])
    l1_arr    = np.array([s[3] for s in all_samples])   # track1 real length
    l2_arr    = np.array([s[4] for s in all_samples])   # track2 real length
    meta_info = np.array([(s[5], s[6]) for s in all_samples])  # (scene_id, batch_id)
    t1_dfs    = [s[7] for s in all_samples]             # track1 DataFrames (for test CSV)
    t2_dfs    = [s[8] for s in all_samples]             # track2 DataFrames (for test CSV)

    # ---- Fit scaler on real data only (no padding) ----
    all_data_for_scaler = []
    for i in range(len(t1_data)):
        all_data_for_scaler.append(t1_data[i, :l1_arr[i]])
        all_data_for_scaler.append(t2_data[i, :l2_arr[i]])

    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_data_for_scaler, axis=0))
    np.save(os.path.join(OUTPUT_ROOT, "scaler_mean.npy"),  scaler.mean_)
    np.save(os.path.join(OUTPUT_ROOT, "scaler_scale.npy"), scaler.scale_)
    print("Scaler saved.")

    # ---- Apply scaler per-track (Bug3 fix: padding region stays 0) ----
    t1_norm = t1_data.copy()
    t2_norm = t2_data.copy()
    for i in range(len(t1_data)):
        t1_norm[i, :l1_arr[i]] = scaler.transform(t1_data[i, :l1_arr[i]])
        t2_norm[i, :l2_arr[i]] = scaler.transform(t2_data[i, :l2_arr[i]])
    t1_norm = t1_norm.astype(np.float32)
    t2_norm = t2_norm.astype(np.float32)

    # ---- Quality check ----
    pos = labels == 1
    neg = labels == 0
    print(f"Positive: {pos.sum()}   Negative: {neg.sum()}")
    print(f"Pos l1 mean={l1_arr[pos].mean():.1f}   Neg l1 mean={l1_arr[neg].mean():.1f}")
    cou_diff_pos = np.abs(t1_norm[pos, :, 3].mean(1) - t2_norm[pos, :, 3].mean(1))
    cou_diff_neg = np.abs(t1_norm[neg, :, 3].mean(1) - t2_norm[neg, :, 3].mean(1))
    print(f"Pos mean course diff (normalized): {cou_diff_pos.mean():.4f}")
    print(f"Neg mean course diff (normalized): {cou_diff_neg.mean():.4f}  (target: <=3x pos)")

    # ---- Train/Val/Test split ----
    print("\n========== Dataset Split ==========")
    idx = np.arange(len(labels))
    idx_train, idx_temp, _, _ = train_test_split(
        idx, labels, test_size=0.3, random_state=42, stratify=labels)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, labels[idx_temp], test_size=2/3, random_state=42,
        stratify=labels[idx_temp])
    print(f"Train: {len(idx_train)}   Val: {len(idx_val)}   Test: {len(idx_test)}")

    # ---- Save test set CSVs: sample_{batch_id}_track1.csv / track2.csv ----
    print("\n========== Saving Test Set CSVs ==========")
    SHORT_THRESH = 80
    LONG_THRESH  = 180

    for label_str in ["label_1", "label_0"]:
        for len_str in ["short", "medium", "long"]:
            os.makedirs(os.path.join(TEST_CSV_ROOT, label_str, len_str), exist_ok=True)

    for sample_num, global_idx in enumerate(idx_test):
        y  = labels[global_idx]
        L  = max(l1_arr[global_idx], l2_arr[global_idx])
        _, batch_id = meta_info[global_idx]

        label_folder = "label_1" if y == 1 else "label_0"
        len_folder   = "short" if L < SHORT_THRESH else ("long" if L > LONG_THRESH else "medium")

        save_dir = os.path.join(TEST_CSV_ROOT, label_folder, len_folder)
        t1_dfs[global_idx].to_csv(
            os.path.join(save_dir, f"sample_{sample_num}_track1.csv"), index=False)
        t2_dfs[global_idx].to_csv(
            os.path.join(save_dir, f"sample_{sample_num}_track2.csv"), index=False)

    print(f"Test CSV saved: {len(idx_test)} pairs  ->  {TEST_CSV_ROOT}")

    # ---- Save NPY files (same layout as final_dataset) ----
    print("\n========== Saving NPY Files ==========")

    def save_npy(name, idxs):
        np.save(os.path.join(OUTPUT_ROOT, f"track1_{name}.npy"),  t1_norm[idxs])
        np.save(os.path.join(OUTPUT_ROOT, f"track2_{name}.npy"),  t2_norm[idxs])
        np.save(os.path.join(OUTPUT_ROOT, f"labels_{name}.npy"),  labels[idxs])
        np.save(os.path.join(OUTPUT_ROOT, f"lengths_{name}.npy"),
                np.maximum(l1_arr[idxs], l2_arr[idxs]))

    save_npy("train", idx_train)
    save_npy("val",   idx_val)
    save_npy("test",  idx_test)

    test_lens = np.maximum(l1_arr[idx_test], l2_arr[idx_test])
    save_npy("test_short",  idx_test[test_lens <  SHORT_THRESH])
    save_npy("test_medium", idx_test[(test_lens >= SHORT_THRESH) & (test_lens <= LONG_THRESH)])
    save_npy("test_long",   idx_test[test_lens >  LONG_THRESH])

    print("NPY files saved.")
    print(f"\n{'='*60}")
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
