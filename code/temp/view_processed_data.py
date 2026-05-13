import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# ===================== 【仅需修改这里的根路径】 =====================
ROOT_DIR = "/home/yangcq/track_association/data/raw"       # 你的MTAD_data文件夹路径
LINK_DIR = os.path.join(ROOT_DIR, "关联表")  # 关联结果文件夹
MEAS_DIR = os.path.join(ROOT_DIR, "量测场景")  # 场景量测数据文件夹
SEQ_LEN = 100                   # 固定航迹长度100个点
FEATURE_COLS = ["lat", "lon", "vel", "cou"]  # 核心特征列
POS_NEG_RATIO = 1/3             # 正负样本比例 1:3
# ====================================================================

# -------------------------- 【修复：解决中文乱码】 --------------------------
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# -------------------------- 辅助函数：航迹固定长度处理 --------------------------
def pad_or_truncate_traj(traj_df, seq_len=SEQ_LEN, feature_cols=FEATURE_COLS):
    traj_sorted = traj_df.sort_values("time").reset_index(drop=True)
    traj_data = traj_sorted[feature_cols].values
    if len(traj_data) >= seq_len:
        return traj_data[:seq_len]
    else:
        pad_width = ((0, seq_len - len(traj_data)), (0, 0))
        return np.pad(traj_data, pad_width, mode="constant", constant_values=0)

# -------------------------- 核心函数：单场景处理 --------------------------
def process_single_scene(scene_id):
    link_path = os.path.join(LINK_DIR, f"关联结果-{scene_id}.csv")
    meas_path = os.path.join(MEAS_DIR, f"场景-{scene_id}.csv")
    if not os.path.exists(link_path) or not os.path.exists(meas_path):
        print(f"场景{scene_id}文件不存在，跳过")
        return None, None, None, None
    
    df_link = pd.read_csv(link_path)
    df_link.columns = df_link.columns.str.strip().str.lower()
    df_meas = pd.read_csv(meas_path)
    df_meas.columns = df_meas.columns.str.strip().str.lower()
    
    mmsi_source_group = df_link.groupby("mmsi")["source"].unique()
    valid_mmsi_list = [
        mmsi for mmsi, sources in mmsi_source_group.items() 
        if set(sources) == {9001, 9002}
    ]
    if len(valid_mmsi_list) < 2:
        print(f"场景{scene_id}有效mmsi不足，跳过")
        return None, None, None, None
    
    traj_dict = {}
    for mmsi in valid_mmsi_list:
        for source in [9001, 9002]:
            batch_list = df_link[(df_link["mmsi"] == mmsi) & (df_link["source"] == source)]["batch"].unique()
            if len(batch_list) == 0:
                continue
            batch = batch_list[0]
            df_batch = df_meas[df_meas["batch"] == batch].copy()
            if len(df_batch) == 0:
                continue
            traj_data = pad_or_truncate_traj(df_batch)
            traj_dict[(mmsi, source)] = traj_data
    
    positive_pairs = []
    positive_meta = []
    for mmsi in valid_mmsi_list:
        key_9001 = (mmsi, 9001)
        key_9002 = (mmsi, 9002)
        if key_9001 not in traj_dict or key_9002 not in traj_dict:
            continue
        positive_pairs.append([traj_dict[key_9001], traj_dict[key_9002]])
        positive_meta.append({
            "scene_id": scene_id,
            "mmsi": mmsi,
            "source_a": 9001,
            "source_b": 9002,
            "label": 1
        })
    pos_num = len(positive_pairs)
    if pos_num == 0:
        print(f"场景{scene_id}无有效正样本，跳过")
        return None, None, None, None
    
    negative_pairs = []
    negative_meta = []
    key_9001_list = [k for k in traj_dict.keys() if k[1] == 9001]
    key_9002_list = [k for k in traj_dict.keys() if k[1] == 9002]
    if len(key_9001_list) == 0 or len(key_9002_list) == 0:
        print(f"场景{scene_id}缺少信源数据，负样本生成失败")
        return None, None, None, None
    
    neg_target_num = int(pos_num / POS_NEG_RATIO)
    used_pairs = set()
    attempts = 0
    max_attempts = neg_target_num * 10
    
    while len(negative_pairs) < neg_target_num and attempts < max_attempts:
        attempts += 1
        key_a = random.choice(key_9001_list)
        key_b = random.choice(key_9002_list)
        mmsi_a, _ = key_a
        mmsi_b, _ = key_b
        
        if mmsi_a == mmsi_b:
            continue
        pair_key = tuple(sorted([key_a, key_b]))
        if pair_key in used_pairs:
            continue
        
        negative_pairs.append([traj_dict[key_a], traj_dict[key_b]])
        negative_meta.append({
            "scene_id": scene_id,
            "mmsi_a": mmsi_a,
            "mmsi_b": mmsi_b,
            "source_a": 9001,
            "source_b": 9002,
            "label": 0
        })
        used_pairs.add(pair_key)
    
    all_pairs = positive_pairs + negative_pairs
    all_labels = [1]*pos_num + [0]*len(negative_pairs)
    all_meta = positive_meta + negative_meta
    X = np.array(all_pairs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    
    return X, y, all_meta, traj_dict

# -------------------------- 场景0处理 + 【自动区分标签保存图片】 --------------------------
print("="*60)
print("正在处理场景0（正负样本1:3）...")
scene0_X, scene0_y, scene0_meta, scene0_traj_dict = process_single_scene(scene_id=0)

if scene0_X is None:
    print("场景0处理失败")
else:
    pos_num = int(sum(scene0_y))
    neg_num = len(scene0_y) - pos_num
    print(f"场景0处理完成！正{pos_num} 负{neg_num}")

    pos_idx = np.where(scene0_y == 1)[0][15]
    neg_idx = np.where(scene0_y == 0)[0][10]

    # ===================== 【唯一改动：绘图并按标签保存图片】 =====================
    def plot_and_save(traj_a, traj_b, meta, label_type):
        plt.figure(figsize=(10, 6))
        plt.plot(traj_a[:, 1], traj_a[:, 0], 'b-o', markersize=3, linewidth=1.5)
        plt.plot(traj_b[:, 1], traj_b[:, 0], 'r-s', markersize=3, linewidth=1.5)
        plt.scatter(traj_a[0,1], traj_a[0,0], c='blue', s=80, marker='s')
        plt.scatter(traj_a[-1,1], traj_a[-1,0], c='blue', s=80, marker='d')
        plt.scatter(traj_b[0,1], traj_b[0,0], c='red', s=80, marker='s')
        plt.scatter(traj_b[-1,1], traj_b[-1,0], c='red', s=80, marker='d')
        
        plt.xlabel('Lon')
        plt.ylabel('Lat')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()

        # 【区分标签1和0保存】
        if label_type == 1:
            plt.title("Scene 0 | Label=1 (Same MMSI)")
            plt.savefig("scene0_label_1.png", dpi=150)
            print("✅ 已保存：scene0_label_1.png")
        else:
            plt.title("Scene 0 | Label=0 (Different MMSI)")
            plt.savefig("scene0_label_0.png", dpi=150)
            print("✅ 已保存：scene0_label_0.png")
        
        plt.close()

    # 正样本 → 保存为 label_1
    plot_and_save(scene0_X[pos_idx,0], scene0_X[pos_idx,1], scene0_meta[pos_idx], 1)

    # 负样本 → 保存为 label_0
    plot_and_save(scene0_X[neg_idx,0], scene0_X[neg_idx,1], scene0_meta[neg_idx], 0)