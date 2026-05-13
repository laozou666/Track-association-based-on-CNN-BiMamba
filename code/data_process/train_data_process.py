import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ===================== 配置区 =====================
# 仿真数据根目录
SIMULATED_DATA_ROOT = "/home/yangcq/track_association/data/simulated_all_scenes/"
# 输出根目录
OUTPUT_ROOT = "/home/yangcq/track_association/data/track_pairs/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 航迹对参数
MAX_TRACK_LENGTH = 200  # 最大航迹长度
POS_NEG_RATIO = 1/3     # 正负样本比例1:3
FEATURE_COLS = ["lat", "lon", "vel", "cou"]
# 数据集划分比例
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
# 处理范围
START_SCENE = 0
END_SCENE = 4999
# 随机种子（保证结果可复现）
RANDOM_SEED = 42
# =====================================================================

def load_scene_data(scene_id):
    """加载单个场景的仿真数据和关联表"""
    scene_path = os.path.join(SIMULATED_DATA_ROOT, f"场景-{scene_id}.csv")
    corr_path = os.path.join(SIMULATED_DATA_ROOT, f"关联结果-{scene_id}.csv")
    
    if not os.path.exists(scene_path) or not os.path.exists(corr_path):
        return None, None
    
    try:
        scene_df = pd.read_csv(scene_path)
        corr_df = pd.read_csv(corr_path)
        return scene_df, corr_df
    except:
        return None, None

def get_track_by_batch(scene_df, batch_id):
    """根据batch号获取单条航迹"""
    track = scene_df[scene_df["batch"] == batch_id].sort_values("time").reset_index(drop=True)
    return track[FEATURE_COLS].values

def pad_or_truncate_track(track, max_len):
    """对齐航迹长度：超过截断，不足补零"""
    if len(track) >= max_len:
        return track[:max_len]
    else:
        pad_len = max_len - len(track)
        pad = np.zeros((pad_len, track.shape[1]))
        return np.vstack([track, pad])

def generate_positive_pairs(scene_df, corr_df, scaler=None):
    """生成正样本对（同一个MMSI的两条航迹）"""
    pairs = []
    labels = []
    
    for mmsi, group in corr_df.groupby("MMSI"):
        if len(group) != 2:
            continue
        
        batch_9001 = group[group["source"] == 9001]["batch"].iloc[0]
        batch_9002 = group[group["source"] == 9002]["batch"].iloc[0]
        
        track1 = get_track_by_batch(scene_df, batch_9001)
        track2 = get_track_by_batch(scene_df, batch_9002)
        
        if len(track1) == 0 or len(track2) == 0:
            continue
        
        if scaler:
            track1 = scaler.transform(track1)
            track2 = scaler.transform(track2)
        
        track1_padded = pad_or_truncate_track(track1, MAX_TRACK_LENGTH)
        track2_padded = pad_or_truncate_track(track2, MAX_TRACK_LENGTH)
        
        pairs.append((track1_padded, track2_padded))
        labels.append(1)
    
    return pairs, labels

def generate_negative_pairs(scene_df, corr_df, num_needed, scaler=None):
    """生成负样本对（不同MMSI的两条航迹）"""
    pairs = []
    labels = []
    
    batches_9001 = corr_df[corr_df["source"] == 9001]["batch"].tolist()
    batches_9002 = corr_df[corr_df["source"] == 9002]["batch"].tolist()
    mmsi_map = dict(zip(corr_df["batch"], corr_df["MMSI"]))
    
    attempts = 0
    max_attempts = num_needed * 10
    
    while len(pairs) < num_needed and attempts < max_attempts:
        attempts += 1
        
        batch1 = random.choice(batches_9001)
        batch2 = random.choice(batches_9002)
        
        mmsi1 = mmsi_map[batch1]
        mmsi2 = mmsi_map[batch2]
        
        if mmsi1 == mmsi2:
            continue
        
        track1 = get_track_by_batch(scene_df, batch1)
        track2 = get_track_by_batch(scene_df, batch2)
        
        if len(track1) == 0 or len(track2) == 0:
            continue
        
        if scaler:
            track1 = scaler.transform(track1)
            track2 = scaler.transform(track2)
        
        track1_padded = pad_or_truncate_track(track1, MAX_TRACK_LENGTH)
        track2_padded = pad_or_truncate_track(track2, MAX_TRACK_LENGTH)
        
        pairs.append((track1_padded, track2_padded))
        labels.append(0)
    
    return pairs, labels

def fit_scaler(all_tracks):
    """在所有数据上拟合标准化器"""
    scaler = StandardScaler()
    all_data = np.vstack(all_tracks)
    scaler.fit(all_data)
    return scaler

def process_single_scene(scene_id, scaler=None):
    """处理单个场景，生成航迹对"""
    scene_df, corr_df = load_scene_data(scene_id)
    if scene_df is None or corr_df is None:
        return None, None, None
    
    pos_pairs, pos_labels = generate_positive_pairs(scene_df, corr_df, scaler)
    
    if len(pos_pairs) == 0:
        return None, None, None
    
    num_neg_needed = int(len(pos_pairs) / POS_NEG_RATIO)
    neg_pairs, neg_labels = generate_negative_pairs(scene_df, corr_df, num_neg_needed, scaler)
    
    all_pairs = pos_pairs + neg_pairs
    all_labels = pos_labels + neg_labels
    
    combined = list(zip(all_pairs, all_labels))
    random.shuffle(combined)
    all_pairs, all_labels = zip(*combined)
    
    track1_array = np.array([p[0] for p in all_pairs])
    track2_array = np.array([p[1] for p in all_pairs])
    labels_array = np.array(all_labels)
    
    return track1_array, track2_array, labels_array

def split_dataset(track1, track2, labels):
    """
    标准数据集划分：60%训练 / 20%验证 / 20%测试
    """
    # 第一步：先划分训练集和剩余集（80%）
    t1_train, t1_rest, t2_train, t2_rest, lbl_train, lbl_rest = train_test_split(
        track1, track2, labels, 
        test_size=(1 - TRAIN_RATIO), 
        random_state=RANDOM_SEED,
        stratify=labels  # 保证正负样本比例一致
    )
    
    # 第二步：在剩余集中划分验证集和测试集（各50%，即总数据的20%）
    val_test_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    t1_val, t1_test, t2_val, t2_test, lbl_val, lbl_test = train_test_split(
        t1_rest, t2_rest, lbl_rest,
        test_size=(1 - val_test_ratio),
        random_state=RANDOM_SEED,
        stratify=lbl_rest
    )
    
    return {
        "train": (t1_train, t2_train, lbl_train),
        "val": (t1_val, t2_val, lbl_val),
        "test": (t1_test, t2_test, lbl_test)
    }

def print_dataset_stats(name, track1, track2, labels):
    """打印数据集统计信息"""
    total = len(labels)
    pos = np.sum(labels)
    neg = total - pos
    print(f"  {name}:")
    print(f"    总样本数: {total}")
    print(f"    正样本: {pos} ({pos/total*100:.1f}%)")
    print(f"    负样本: {neg} ({neg/total*100:.1f}%)")
    print(f"    航迹形状: {track1.shape}")

if __name__ == "__main__":
    print("="*70)
    print(" 航迹对生成器（含训练/验证/测试集划分）")
    print(f"  参数：正负样本1:3 | 划分比例 {TRAIN_RATIO*100:.0f}%/{VAL_RATIO*100:.0f}%/{TEST_RATIO*100:.0f}%")
    print("="*70)

    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 第一步：收集所有数据，拟合全局标准化器
    print("\n[1/4] 收集所有数据，拟合全局标准化器...")
    all_tracks = []
    valid_scenes = []
    
    for scene_id in range(START_SCENE, END_SCENE + 1):
        if (scene_id + 1) % 100 == 0:
            print(f"  收集进度：{scene_id + 1}/{END_SCENE + 1}")
        
        scene_df, corr_df = load_scene_data(scene_id)
        if scene_df is None:
            continue
        
        valid_scenes.append(scene_id)
        for batch in scene_df["batch"].unique():
            track = get_track_by_batch(scene_df, batch)
            if len(track) > 0:
                all_tracks.append(track)
    
    if len(all_tracks) == 0:
        print("❌ 没有找到有效数据")
        exit()
    
    scaler = fit_scaler(all_tracks)
    print(f"✅ 全局标准化器拟合完成，共 {len(all_tracks)} 条航迹")

    # 第二步：批量生成航迹对
    print("\n[2/4] 批量生成航迹对...")
    all_track1 = []
    all_track2 = []
    all_labels = []
    
    for i, scene_id in enumerate(valid_scenes):
        if (i + 1) % 50 == 0:
            print(f"  生成进度：{i + 1}/{len(valid_scenes)}")
        
        t1, t2, lbl = process_single_scene(scene_id, scaler)
        if t1 is not None:
            all_track1.append(t1)
            all_track2.append(t2)
            all_labels.append(lbl)
    
    final_track1 = np.vstack(all_track1)
    final_track2 = np.vstack(all_track2)
    final_labels = np.hstack(all_labels)
    
    print(f"✅ 航迹对生成完成，总样本数：{len(final_labels)}")

    # 第三步：划分数据集
    print("\n[3/4] 划分训练/验证/测试集...")
    datasets = split_dataset(final_track1, final_track2, final_labels)
    
    print("\n数据集统计：")
    print_dataset_stats("训练集", *datasets["train"])
    print_dataset_stats("验证集", *datasets["val"])
    print_dataset_stats("测试集", *datasets["test"])

    # 第四步：保存结果
    print("\n[4/4] 保存结果...")
    
    # 保存三个数据集
    for split_name, (t1, t2, lbl) in datasets.items():
        np.save(os.path.join(OUTPUT_ROOT, f"track1_{split_name}.npy"), t1)
        np.save(os.path.join(OUTPUT_ROOT, f"track2_{split_name}.npy"), t2)
        np.save(os.path.join(OUTPUT_ROOT, f"labels_{split_name}.npy"), lbl)
    
    # 保存scaler参数
    np.save(os.path.join(OUTPUT_ROOT, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(OUTPUT_ROOT, "scaler_scale.npy"), scaler.scale_)
    
    print("\n" + "="*70)
    print("🎉 全部完成！")
    print(f"  输出目录：{OUTPUT_ROOT}")
    print("  文件列表：")
    print("    ├── track1_train.npy / track2_train.npy / labels_train.npy (训练集)")
    print("    ├── track1_val.npy   / track2_val.npy   / labels_val.npy   (验证集)")
    print("    ├── track1_test.npy  / track2_test.npy  / labels_test.npy  (测试集)")
    print("    ├── scaler_mean.npy  / scaler_scale.npy (标准化器参数)")
    print("  ✅ 所有数据集已按60/20/20比例划分")
    print("  ✅ 每个数据集的正负样本比例保持一致")
    print("  ✅ 所有航迹已对齐长度并标准化")
    print("="*70)