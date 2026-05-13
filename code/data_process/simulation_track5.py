import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ===================== 配置区 =====================
REAL_SCENE_ROOT = "/home/yangcq/track_association/data/raw/真实场景/"
OUTPUT_ROOT = "/home/yangcq/track_association/data/final_dataset"
CSV_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "simulated_scenes_csv")
# 【新增】测试集CSV分层保存根目录
TEST_CSV_ROOT = os.path.join(OUTPUT_ROOT, "test_set_csvs")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(CSV_OUTPUT_ROOT, exist_ok=True)

# 航迹参数
MAX_SEQ_LEN = 350
MIN_TRACK_POINTS = 25
LEN_DIFF_THRESH = 10
TRUNCATE_START_THRESH = 20
TRUNCATE_TARGET_THRESH = 5

# 场景范围
START_SCENE = 0
END_SCENE = 4999

# 随机种子
random.seed(42)
np.random.seed(42)

# 传感器参数
OBSERVER_PARAMS = {
    9001: {"name": "Radar_A", "bias_lon": 0.0015, "bias_lat": -0.0012, "noise_pos": 0.00035, "noise_vel": 1.0, "noise_cou": 6.0, "freq_ratio": 1.0, "dropout_rate": 0.18},
    9002: {"name": "Radar_B", "bias_lon": -0.002, "bias_lat": 0.0018, "noise_pos": 0.0005, "noise_vel": 1.5, "noise_cou": 9.0, "freq_ratio": 0.75, "dropout_rate": 0.25}
}

# ===================== 核心仿真函数 =====================
def simulate_track_with_observer(truth_track, source_id):
    params = OBSERVER_PARAMS[source_id]
    track = truth_track.copy().sort_values("time").reset_index(drop=True)
    
    original_len = len(track)
    if np.random.rand() < 0.6:
        start_offset = np.random.randint(0, int(original_len * 0.25))
        end_offset = np.random.randint(0, int(original_len * 0.15))
        track = track.iloc[start_offset : original_len - end_offset].reset_index(drop=True)
    
    target_len = int(len(track) * params["freq_ratio"])
    target_len = max(MIN_TRACK_POINTS, target_len)
    if len(track) != target_len:
        idx_old = np.linspace(0, len(track)-1, len(track))
        idx_new = np.linspace(0, len(track)-1, target_len)
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
    t = np.linspace(0, 2*np.pi, len(track))
    lon_noise = np.cumsum(np.random.normal(0, params["noise_pos"] * 0.3, len(track))) * 0.3 + np.sin(t * np.random.uniform(1, 3)) * params["noise_pos"] * 1.5
    lat_noise = np.cumsum(np.random.normal(0, params["noise_pos"] * 0.3, len(track))) * 0.3 + np.cos(t * np.random.uniform(1, 3)) * params["noise_pos"] * 1.5
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
    features = track[["lat", "lon", "vel", "cou"]].values.astype(np.float32)
    seq_len = len(features)
    if seq_len > MAX_SEQ_LEN:
        return features[:MAX_SEQ_LEN], seq_len
    elif seq_len < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - seq_len
        return np.pad(features, ((0, pad_len), (0, 0)), mode='constant'), seq_len
    return features, seq_len

def get_length_matched_pairs(track_pool):
    pairs, used = [], set()
    track_len = [len(tr) for tr in track_pool]
    for i in range(len(track_pool)):
        if i in used: continue
        best_j, min_diff = -1, 999
        for j in range(i+1, len(track_pool)):
            if j in used: continue
            diff = abs(track_len[i] - track_len[j])
            if diff <= LEN_DIFF_THRESH * 2 and diff < min_diff:
                min_diff, best_j = diff, j
        if best_j != -1:
            pairs.append((track_pool[i], track_pool[j]))
            used.add(i), used.add(j)
    return pairs

def process_single_scene(scene_id):
    input_path = os.path.join(REAL_SCENE_ROOT, f"场景-{scene_id}.csv")
    if not os.path.exists(input_path):
        return [], None, None
    
    try:
        df = pd.read_csv(input_path)
        track_pool = [g.sort_values("time").reset_index(drop=True) for _, g in df.groupby("MMSI") if len(g) >= MIN_TRACK_POINTS]
        random.shuffle(track_pool)
        
        if len(track_pool) < 2:
            return [], None, None

        samples = []
        scene_rows = []
        corr_rows = []
        batch_id = 0

        # 正样本
        n_pos = len(track_pool) // 2
        for tr in track_pool[:n_pos]:
            t1, l1 = simulate_track_with_observer(tr, 9001)
            t2, l2 = simulate_track_with_observer(tr, 9002)
            t1["source"], t2["source"] = 9001, 9002
            t1["batch"], t2["batch"] = batch_id, batch_id
            scene_rows += [t1, t2]
            corr_rows += [{"MMSI":tr["MMSI"].iloc[0],"source":9001,"batch":batch_id,"label":1}, {"MMSI":tr["MMSI"].iloc[0],"source":9002,"batch":batch_id,"label":1}]
            f1, _ = pad_or_truncate_track(t1)
            f2, _ = pad_or_truncate_track(t2)
            samples.append((f1, f2, 1, max(l1, l2), scene_id, batch_id)) # 【修改】记录scene_id和batch_id用于保存CSV
            batch_id += 1

        # 负样本
        neg_pairs = get_length_matched_pairs(track_pool[n_pos:])
        for trA, trB in neg_pairs:
            if trA["vel"].mean() < 1.2 or trB["vel"].mean() < 1.2: continue
            if (trA["lat"].max()-trA["lat"].min()) < 0.003: continue
            if (trB["lat"].max()-trB["lat"].min()) < 0.003: continue

            t1, l1 = simulate_track_with_observer(trA, 9001)
            t2, l2 = simulate_track_with_observer(trB, 9002)
            
            len1, len2 = len(t1), len(t2)
            len_diff = abs(len1 - len2)
            target_len = min(len1, len2) + (TRUNCATE_TARGET_THRESH if len_diff > TRUNCATE_START_THRESH else 0)
            t1 = t1.iloc[:target_len].reset_index(drop=True)
            t2 = t2.iloc[:target_len].reset_index(drop=True)
            t2 = shift_track_perfect(t2, t1)

            t1["source"], t2["source"] = 9001, 9002
            t1["batch"], t2["batch"] = batch_id, batch_id
            scene_rows += [t1, t2]
            corr_rows += [{"MMSI":trA["MMSI"].iloc[0],"source":9001,"batch":batch_id,"label":0}, {"MMSI":trB["MMSI"].iloc[0],"source":9002,"batch":batch_id,"label":0}]
            f1, _ = pad_or_truncate_track(t1)
            f2, _ = pad_or_truncate_track(t2)
            samples.append((f1, f2, 0, min(len(t1), len(t2)), scene_id, batch_id))
            batch_id += 1

        scene_df = pd.concat(scene_rows, ignore_index=True) if scene_rows else None
        corr_df = pd.DataFrame(corr_rows) if corr_rows else None
        return samples, scene_df, corr_df

    except Exception as e:
        return [], None, None

def main():
    print("="*60)
    print(f" 处理场景：{START_SCENE} ~ {END_SCENE}")
    print("="*60)

    all_samples = []
    # 【新增】存储所有场景的DataFrame，方便后续查找
    scene_df_cache = {} 
    corr_df_cache = {}

    # 1. 仿真所有场景
    for scene_id in range(START_SCENE, END_SCENE + 1):
        samples, scene_df, corr_df = process_single_scene(scene_id)
        
        if len(samples) > 0:
            all_samples.extend(samples)
            scene_df_cache[scene_id] = scene_df
            corr_df_cache[scene_id] = corr_df
            
            # 保存完整CSV（可选，保留你之前的习惯）
            scene_df.to_csv(os.path.join(CSV_OUTPUT_ROOT, f"场景-{scene_id}.csv"), index=False)
            corr_df.to_csv(os.path.join(CSV_OUTPUT_ROOT, f"关联结果-{scene_id}.csv"), index=False)

        if (scene_id - START_SCENE) % 100 == 0:
            print(f"⏳ 已处理 {scene_id - START_SCENE + 1} 个场景")

    if len(all_samples) == 0:
        print("❌ 无样本")
        return

    # 2. 提取数据
    print(f"\n========== 数据处理与归一化 ==========")
    t1_data = np.array([s[0] for s in all_samples])
    t2_data = np.array([s[1] for s in all_samples])
    labels = np.array([s[2] for s in all_samples])
    lengths = np.array([s[3] for s in all_samples])
    meta_info = np.array([(s[4], s[5]) for s in all_samples]) # (scene_id, batch_id)

    # 3. 拟合并保存归一化
    all_data_for_scaler = []
    for i in range(len(t1_data)):
        valid_len = lengths[i]
        all_data_for_scaler.append(t1_data[i][:valid_len])
        all_data_for_scaler.append(t2_data[i][:valid_len])
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_data_for_scaler, axis=0))
    np.save(os.path.join(OUTPUT_ROOT, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(OUTPUT_ROOT, "scaler_std.npy"), scaler.scale_)
    print(f"✅ 归一化参数已保存")

    # 4. 应用归一化
    def apply_scaler(data):
        data_norm = data.copy()
        for i in range(len(data)):
            valid_len = min(lengths[i], MAX_SEQ_LEN)
            data_norm[i, :valid_len] = scaler.transform(data[i, :valid_len])
        return data_norm.astype(np.float32)

    t1_data_norm = apply_scaler(t1_data)
    t2_data_norm = apply_scaler(t2_data)

    # 5. 划分数据集
    print(f"\n========== 划分数据集 ==========")
    idx = np.arange(len(labels))
    idx_train, idx_temp, _, _ = train_test_split(idx, labels, test_size=0.3, random_state=42, stratify=labels)
    idx_val, idx_test, _, _ = train_test_split(idx_temp, labels[idx_temp], test_size=2/3, random_state=42, stratify=labels[idx_temp])

    # 6. 【核心需求1】分层保存测试集CSV
    print(f"\n========== 分层保存测试集CSV ==========")
    
    # 长度阈值
    SHORT_THRESH = 80
    LONG_THRESH = 180

    # 创建目录结构
    for label_str in ["label_1", "label_0"]:
        for len_str in ["short", "medium", "long"]:
            dir_path = os.path.join(TEST_CSV_ROOT, label_str, len_str)
            os.makedirs(dir_path, exist_ok=True)

    # 遍历测试集每一个样本
    test_count = 0
    for global_idx in idx_test:
        y = labels[global_idx]
        L = lengths[global_idx]
        scene_id, batch_id = meta_info[global_idx]
        
        # 确定分类
        label_folder = "label_1" if y == 1 else "label_0"
        if L < SHORT_THRESH:
            len_folder = "short"
        elif L > LONG_THRESH:
            len_folder = "long"
        else:
            len_folder = "medium"
            
        # 从缓存中取出完整的场景DF，提取当前batch
        if scene_id in scene_df_cache:
            full_scene_df = scene_df_cache[scene_id]
            full_corr_df = corr_df_cache[scene_id]
            
            # 只提取当前batch的数据
            batch_scene_df = full_scene_df[full_scene_df["batch"] == batch_id]
            batch_corr_df = full_corr_df[full_corr_df["batch"] == batch_id]
            
            # 保存路径
            save_dir = os.path.join(TEST_CSV_ROOT, label_folder, len_folder)
            batch_scene_df.to_csv(os.path.join(save_dir, f"sample_{test_count}_scene.csv"), index=False)
            batch_corr_df.to_csv(os.path.join(save_dir, f"sample_{test_count}_corr.csv"), index=False)
            
            test_count += 1

    print(f"✅ 测试集CSV分层保存完成，共 {test_count} 个样本")

    # 7. 【核心需求2】保存归一化后的NPY
    print(f"\n========== 保存NPY文件 ==========")
    
    def save_npy(name, idxs):
        np.save(os.path.join(OUTPUT_ROOT, f"track1_{name}.npy"), t1_data_norm[idxs])
        np.save(os.path.join(OUTPUT_ROOT, f"track2_{name}.npy"), t2_data_norm[idxs])
        np.save(os.path.join(OUTPUT_ROOT, f"labels_{name}.npy"), labels[idxs])
        np.save(os.path.join(OUTPUT_ROOT, f"lengths_{name}.npy"), lengths[idxs])

    save_npy("train", idx_train)
    save_npy("val", idx_val)
    save_npy("test", idx_test)
    
    # 测试集也按长度保存NPY（可选，方便你直接用）
    test_lens = lengths[idx_test]
    save_npy("test_short", idx_test[test_lens < SHORT_THRESH])
    save_npy("test_medium", idx_test[(test_lens >= SHORT_THRESH) & (test_lens <= LONG_THRESH)])
    save_npy("test_long", idx_test[test_lens > LONG_THRESH])

    print(f"✅ NPY文件保存完成")
    print(f"\n" + "="*60)
    print("全部完成！")
    print("="*60)

if __name__ == "__main__":
    main()