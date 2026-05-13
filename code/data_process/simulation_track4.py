import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===================== 配置区（只处理场景0）=====================
REAL_SCENE_ROOT = "/home/yangcq/track_association/data/raw/真实场景/"
OUTPUT_ROOT = "/home/yangcq/track_association/data/final_dataset_scene0_debug"
CSV_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "simulated_scenes_csv")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(CSV_OUTPUT_ROOT, exist_ok=True)

# 航迹参数
MAX_SEQ_LEN = 500
INPUT_DIM = 4
MIN_TRACK_POINTS = 25
LEN_DIFF_THRESH = 10

# 场景范围（只处理场景0）
START_SCENE = 0
END_SCENE = 0

# 随机种子
random.seed(42)
np.random.seed(42)
# =====================================================================

def simulate_track(truth_track, source_id):
    track = truth_track.copy().sort_values("time").reset_index(drop=True)
    
    # ========== 正样本核心改造：神似形不似 ==========
    # 1. 极小全局偏移（保证整体趋势一致，不明显分开）
    track["lon"] += np.random.uniform(-0.0008, 0.0008)
    track["lat"] += np.random.uniform(-0.0008, 0.0008)
    
    # 2. 局部平滑扰动（制造局部交叉、贴合、错开，不是整体抖动）
    # 生成平滑的随机偏移序列，让航迹局部缠绕
    t = np.linspace(0, 2*np.pi, len(track))
    lon_noise = np.cumsum(np.random.normal(0, 0.00003, len(track))) * 0.3
    lat_noise = np.cumsum(np.random.normal(0, 0.00003, len(track))) * 0.3
    # 加微小正弦波动，制造局部交叉
    lon_noise += np.sin(t * np.random.uniform(1, 3)) * 0.00012
    lat_noise += np.cos(t * np.random.uniform(1, 3)) * 0.00012
    
    track["lon"] += lon_noise
    track["lat"] += lat_noise
    
    # 3. 速度航向加合理噪声
    track["vel"] += np.random.normal(0, 0.35, len(track))
    track["cou"] = (track["cou"] + np.random.normal(0, 2.0, len(track))) % 360
    
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
        best_j, min_diff = -1, 999
        for j in range(i+1, len(track_pool)):
            if j in used: continue
            diff = abs(track_len[i] - track_len[j])
            if diff <= LEN_DIFF_THRESH and diff < min_diff:
                min_diff, best_j = diff, j
        if best_j != -1:
            pairs.append((track_pool[i], track_pool[best_j]))
            used.add(i), used.add(best_j)
    return pairs

def visualize_scene_samples(scene_df, corr_df, save_path="scene0_visualization.png"):
    """
    可视化场景0的正/负样本航迹对
    """
    batches = scene_df["batch"].unique()
    n_batches = len(batches)
    
    if n_batches == 0:
        print("❌ 无样本可可视化")
        return

    # 计算子图数量
    n_cols = 2
    n_rows = (n_batches + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_batches > 1 else [axes]

    for idx, batch_id in enumerate(batches):
        ax = axes[idx]
        batch_data = scene_df[scene_df["batch"] == batch_id]
        t1 = batch_data[batch_data["source"] == 9001]
        t2 = batch_data[batch_data["source"] == 9002]
        
        # 获取标签
        label = corr_df[corr_df["batch"] == batch_id]["label"].iloc[0]
        color = "green" if label == 1 else "red"
        title = f"Batch {batch_id} | Label={label} ({'Positive' if label == 1 else 'Negative'})"

        # 画航迹
        ax.scatter(t1["lon"], t1["lat"], c="#1f77b4", s=30, alpha=0.7, label="Source 9001")
        ax.scatter(t2["lon"], t2["lat"], c="#ff7f0e", s=30, alpha=0.7, label="Source 9002")
        
        # 起点终点
        ax.scatter(t1["lon"].iloc[0], t1["lat"].iloc[0], c="green", s=150, edgecolor="black", label="Start")
        ax.scatter(t1["lon"].iloc[-1], t1["lat"].iloc[-1], c="red", s=150, marker="x", linewidth=2, label="End")
        ax.scatter(t2["lon"].iloc[0], t2["lat"].iloc[0], c="green", s=150, edgecolor="black")
        ax.scatter(t2["lon"].iloc[-1], t2["lat"].iloc[-1], c="red", s=150, marker="x", linewidth=2)

        ax.set_title(title, fontsize=12, color=color)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 隐藏多余子图
    for idx in range(n_batches, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ 场景0可视化完成：{save_path}")

def process_single_scene(scene_id):
    input_path = os.path.join(REAL_SCENE_ROOT, f"场景-{scene_id}.csv")
    if not os.path.exists(input_path):
        print(f"❌ 场景{scene_id}不存在：{input_path}")
        return [], None, None
    
    try:
        df = pd.read_csv(input_path)
        track_pool = [g.sort_values("time").reset_index(drop=True)
                      for _, g in df.groupby("MMSI") if len(g) >= MIN_TRACK_POINTS]
        random.shuffle(track_pool)
        print(f"✅ 场景{scene_id}加载完成：原始航迹数 {len(track_pool)}")
        
        if len(track_pool) < 2:
            print("❌ 航迹数不足2，无法生成样本")
            return [], None, None

        samples = []
        scene_rows = []
        corr_rows = []
        batch_id = 0

        # 正样本
        n_pos = len(track_pool) // 2
        print(f"  生成正样本：{n_pos} 对")
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

        # ========== 安全版负样本生成 ==========
        # ========== 最终负样本：全程纠缠+无分离+长度自然+双运动 ==========
        neg_pairs = get_length_matched_pairs(track_pool[n_pos:])
        print(f"  生成负样本：{len(neg_pairs)} 对（全程交叉纠缠 | 无分离）")

        for trA, trB in neg_pairs:
            # 过滤静止/微动
            if trA["vel"].mean() < 1.2 or trB["vel"].mean() < 1.2: continue
            if (trA["lat"].max()-trA["lat"].min()) < 0.003: continue
            if (trB["lat"].max()-trB["lat"].min()) < 0.003: continue

            t1 = simulate_track(trA, 9001)
            t2 = simulate_track(trB, 9002)
            common_len = min(len(t1), len(t2))
            t1 = t1.iloc[:common_len].reset_index(drop=True)
            t2 = t2.iloc[:common_len].reset_index(drop=True)

            # ========== 核心：强制聚拢+小旋转 → 全程缠绕，绝不分开 ==========
            # 1. 把两条航迹强行拉到同一空间中心（杜绝远距离）
            mean_lon, mean_lat = t1["lon"].mean(), t1["lat"].mean()
            t2["lon"] = mean_lon + (t2["lon"] - t2["lon"].mean()) * 0.6
            t2["lat"] = mean_lat + (t2["lat"] - t2["lat"].mean()) * 0.6

            # 2. 小角度随机旋转（制造交叉、缠绕）
            theta = np.radians(np.random.uniform(-25, 25))
            lon_r = t2["lon"] - mean_lon
            lat_r = t2["lat"] - mean_lat
            t2["lon"] = lon_r*np.cos(theta) - lat_r*np.sin(theta) + mean_lon
            t2["lat"] = lon_r*np.sin(theta) + lat_r*np.cos(theta) + mean_lat

            # 3. 微小抖动更自然
            t2["lon"] += np.random.normal(0, 0.00015, len(t2))
            t2["lat"] += np.random.normal(0, 0.00015, len(t2))

            # 写入数据
            t1["source"], t2["source"] = 9001, 9002
            t1["batch"], t2["batch"] = batch_id, batch_id
            scene_rows += [t1, t2]
            corr_rows += [
                {"MMSI":trA["MMSI"].iloc[0],"source":9001,"batch":batch_id,"label":0},
                {"MMSI":trB["MMSI"].iloc[0],"source":9002,"batch":batch_id,"label":0}
            ]
            f1, _ = pad_or_truncate_track(t1)
            f2, _ = pad_or_truncate_track(t2)
            samples.append((f1, f2, 0, common_len))
            batch_id += 1

        scene_df = pd.concat(scene_rows, ignore_index=True) if scene_rows else None
        corr_df = pd.DataFrame(corr_rows) if corr_rows else None
        return samples, scene_df, corr_df

    except Exception as e:
        print(f"❌ 场景{scene_id}处理失败：{e}")
        import traceback
        traceback.print_exc()
        return [], None, None

def main():
    print("="*80)
    print(" 场景0调试版 | 生成+可视化")
    print("="*80)

    # 1. 只处理场景0
    all_samples = []
    samples, scene_df, corr_df = process_single_scene(0)
    all_samples += samples
    
    if scene_df is not None:
        scene_df.to_csv(os.path.join(CSV_OUTPUT_ROOT, f"场景-0.csv"), index=False)
        corr_df.to_csv(os.path.join(CSV_OUTPUT_ROOT, f"关联结果-0.csv"), index=False)
        print(f"\n✅ 场景0 CSV保存完成")

    if len(all_samples) == 0:
        print("❌ 无样本生成")
        return

    # 2. 可视化（核心新增）
    print(f"\n========== 开始可视化场景0 ==========")
    visualize_scene_samples(scene_df, corr_df, 
                           save_path=os.path.join(OUTPUT_ROOT, "scene0_visualization.png"))

    # 3. 简单打包（仅用于调试）
    t1 = np.array([s[0] for s in all_samples])
    t2 = np.array([s[1] for s in all_samples])
    lab = np.array([s[2] for s in all_samples])
    
    print(f"\n" + "="*80)
    print("✅ 场景0处理完成！")
    print(f"总样本对：{len(all_samples)}")
    print(f"正样本：{np.sum(lab==1)} 对")
    print(f"负样本：{np.sum(lab==0)} 对")
    print(f"输出路径：{OUTPUT_ROOT}")
    print("="*80)

if __name__ == "__main__":
    main()