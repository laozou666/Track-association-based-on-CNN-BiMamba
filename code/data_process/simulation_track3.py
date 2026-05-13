import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===================== 配置区（量产）=====================
REAL_SCENE_ROOT = "/home/yangcq/track_association/data/raw/真实场景/"
OUTPUT_ROOT = "/home/yangcq/track_association/data/final_dataset_5000_v3"
CSV_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "simulated_scenes_csv")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(CSV_OUTPUT_ROOT, exist_ok=True)

# 航迹参数
MAX_SEQ_LEN = 500
INPUT_DIM = 4
MIN_TRACK_POINTS = 25
LEN_DIFF_THRESH = 10  # 负样本长度差阈值

# 数据集划分
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 测试集长度划分
SHORT_THRESH = 100
LONG_THRESH = 300

# 场景范围
START_SCENE = 0
END_SCENE = 4999

# 随机种子
random.seed(42)
np.random.seed(42)
# =====================================================================

def simulate_track(truth_track, source_id):
    track = truth_track.copy().sort_values("time").reset_index(drop=True)
    track["lon"] += np.random.uniform(-0.001, 0.001)
    track["lat"] += np.random.uniform(-0.001, 0.001)
    track["lon"] += np.random.normal(0, 0.00008, len(track))
    track["lat"] += np.random.normal(0, 0.00008, len(track))
    track["vel"] += np.random.normal(0, 0.2, len(track))
    track["cou"] = (track["cou"] + np.random.normal(0, 1, len(track))) % 360
    return track

def align_track_length(t1, t2):
    min_len = min(len(t1), len(t2))
    return t1.iloc[:min_len].reset_index(drop=True), t2.iloc[:min_len].reset_index(drop=True)

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
    pairs = []
    used = set()
    track_len = [len(tr) for tr in track_pool]
    for i in range(len(track_pool)):
        if i in used: continue
        best_j, min_diff = -1, 9999
        for j in range(i+1, len(track_pool)):
            if j in used: continue
            diff = abs(track_len[i] - track_len[j])
            if diff <= LEN_DIFF_THRESH and diff < min_diff:
                min_diff, best_j = diff, j
        if best_j != -1:
            pairs.append((track_pool[i], track_pool[best_j]))
            used.add(i), used.add(best_j)
    return pairs

def process_single_scene(scene_id):
    input_path = os.path.join(REAL_SCENE_ROOT, f"场景-{scene_id}.csv")
    if not os.path.exists(input_path):
        return [], None, None
    try:
        df = pd.read_csv(input_path)
        track_pool = [g.sort_values("time").reset_index(drop=True)
                      for _, g in df.groupby("MMSI") if len(g) >= MIN_TRACK_POINTS]
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
            t1, t2 = simulate_track(tr,9001), simulate_track(tr,9002)
            t1, t2 = align_track_length(t1, t2)
            t1["source"], t2["source"] = 9001, 9002
            t1["batch"], t2["batch"] = batch_id, batch_id
            scene_rows += [t1, t2]
            corr_rows += [
                {"MMSI":tr["MMSI"].iloc[0],"source":9001,"batch":batch_id,"label":1},
                {"MMSI":tr["MMSI"].iloc[0],"source":9002,"batch":batch_id,"label":1}
            ]
            f1, l1 = pad_or_truncate_track(t1)
            f2, l2 = pad_or_truncate_track(t2)
            samples.append((f1, f2, 1, max(l1,l2)))
            batch_id += 1

        # 负样本（长度匹配+完美重叠）
        neg_pairs = get_length_matched_pairs(track_pool[n_pos:])
        for trA, trB in neg_pairs:
            t1 = simulate_track(trA, 9001)
            t2 = simulate_track(trB, 9002)
            t1, t2 = align_track_length(t1, t2)
            t2 = shift_track_perfect(t2, t1)
            t1["source"], t2["source"] = 9001, 9002
            t1["batch"], t2["batch"] = batch_id, batch_id
            scene_rows += [t1, t2]
            corr_rows += [
                {"MMSI":trA["MMSI"].iloc[0],"source":9001,"batch":batch_id,"label":0},
                {"MMSI":trB["MMSI"].iloc[0],"source":9002,"batch":batch_id,"label":0}
            ]
            f1, l1 = pad_or_truncate_track(t1)
            f2, l2 = pad_or_truncate_track(t2)
            samples.append((f1, f2, 0, max(l1,l2)))
            batch_id += 1

        scene_df = pd.concat(scene_rows, ignore_index=True) if scene_rows else None
        corr_df = pd.DataFrame(corr_rows) if corr_rows else None
        return samples, scene_df, corr_df
    except:
        return [], None, None

def main():
    print("="*80)
    print(" 5000场景量产版 | 长度匹配负样本 | 完美重叠 | 配对正确")
    print("="*80)

    # 1. 批量处理
    all_samples = []
    for sid in range(START_SCENE, END_SCENE+1):
        if (sid+1) % 500 == 0:
            print(f"进度：{sid+1}/{END_SCENE+1}")
        samples, scene_df, corr_df = process_single_scene(sid)
        all_samples += samples
        if scene_df is not None:
            scene_df.to_csv(os.path.join(CSV_OUTPUT_ROOT, f"场景-{sid}.csv"), index=False)
            corr_df.to_csv(os.path.join(CSV_OUTPUT_ROOT, f"关联结果-{sid}.csv"), index=False)

    print(f"\n总样本对：{len(all_samples)}")
    if len(all_samples) == 0:
        print("❌ 无样本")
        return

    # 2. 打包数据
    t1 = np.array([s[0] for s in all_samples])
    t2 = np.array([s[1] for s in all_samples])
    lab = np.array([s[2] for s in all_samples])
    lens = np.array([s[3] for s in all_samples])

    # 3. 划分
    idx_train, idx_rest = train_test_split(np.arange(len(lab)), test_size=0.3, stratify=lab, random_state=42)
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.5, stratify=lab[idx_rest], random_state=42)

    # 4. 测试集分长短
    len_test = lens[idx_test]
    idx_short = idx_test[len_test < SHORT_THRESH]
    idx_medium = idx_test[(len_test>=SHORT_THRESH)&(len_test<=LONG_THRESH)]
    idx_long = idx_test[len_test > LONG_THRESH]

    # 5. 标准化
    scaler = StandardScaler()
    scaler.fit(np.concatenate([t1[idx_train], t2[idx_train]]).reshape(-1,4))
    def norm(x): return scaler.transform(x.reshape(-1,4)).reshape(x.shape).astype(np.float32)
    t1, t2 = norm(t1), norm(t2)

    np.save(os.path.join(OUTPUT_ROOT, "scaler_mean.npy"), scaler.mean_.astype(np.float32))
    np.save(os.path.join(OUTPUT_ROOT, "scaler_scale.npy"), scaler.scale_.astype(np.float32))

    # 6. 保存全套NPY
    def save(name, a,b,c):
        np.save(f"{OUTPUT_ROOT}/track1_{name}.npy", a)
        np.save(f"{OUTPUT_ROOT}/track2_{name}.npy", b)
        np.save(f"{OUTPUT_ROOT}/labels_{name}.npy", c)

    save("train", t1[idx_train], t2[idx_train], lab[idx_train])
    save("val",   t1[idx_val],   t2[idx_val],   lab[idx_val])
    save("test",  t1[idx_test],  t2[idx_test],  lab[idx_test])
    save("test_short",  t1[idx_short],  t2[idx_short],  lab[idx_short])
    save("test_medium", t1[idx_medium], t2[idx_medium], lab[idx_medium])
    save("test_long",   t1[idx_long],   t2[idx_long],   lab[idx_long])

    print("\n" + "="*80)
    print("✅ 5000场景生成完成！")
    print(f"数据集路径：{OUTPUT_ROOT}")
    print(f"测试集分布：短{len(idx_short)} / 中{len(idx_medium)} / 长{len(idx_long)}")
    print("特性：负样本长度匹配 + 完全重叠 + 配对正确")
    print("="*80)

if __name__ == "__main__":
    main()