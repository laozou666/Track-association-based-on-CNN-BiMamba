import pandas as pd
import numpy as np
import os
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===================== 路径 =====================
CSV_ROOT = "/home/yangcq/track_association/data/final_dataset/simulated_scenes_csv"
SAVE_DIR = "/home/yangcq/track_association/data/final_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 超参 =====================
INPUT_DIM = 4
MIN_POINTS = 10

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SHORT_THRESH  = 100
LONG_THRESH   = 300

np.random.seed(42)

# =====================================================================
# 【核心】样本对内部插值对齐
# =====================================================================
def interpolate_pair_aligned(f1, f2):
    """
    只在对内部对齐：短的插值到长的长度
    f1: [seq1, 4]
    f2: [seq2, 4]
    return: (f1_aligned, f2_aligned, max_len)
    """
    len1, len2 = len(f1), len(f2)
    max_len = max(len1, len2)
    
    def _interp(feat, target_len):
        if len(feat) == target_len:
            return feat
        t_orig = np.linspace(0, 1, len(feat))
        t_target = np.linspace(0, 1, target_len)
        feat_interp = np.zeros((target_len, INPUT_DIM), dtype=np.float32)
        for i in range(INPUT_DIM):
            try:
                f = interpolate.interp1d(t_orig, feat[:, i], kind='cubic')
                feat_interp[:, i] = f(t_target)
            except:
                f = interpolate.interp1d(t_orig, feat[:, i], kind='linear')
                feat_interp[:, i] = f(t_target)
        return feat_interp
    
    return _interp(f1, max_len), _interp(f2, max_len), max_len

# =====================================================================
# 读取场景CSV
# =====================================================================
def load_scene_pairs(scene_id):
    csv_path  = os.path.join(CSV_ROOT, f"场景-{scene_id}.csv")
    corr_path = os.path.join(CSV_ROOT, f"关联结果-{scene_id}.csv")

    if not os.path.exists(csv_path) or not os.path.exists(corr_path):
        return []

    df = pd.read_csv(csv_path)
    pairs = []

    for batch_id, g in df.groupby("batch"):
        t9001 = g[g["source"] == 9001].sort_values("time")
        t9002 = g[g["source"] == 9002].sort_values("time")

        if len(t9001) < MIN_POINTS or len(t9002) < MIN_POINTS:
            continue

        f1 = t9001[["lat", "lon", "vel", "cou"]].values.astype(np.float32)
        f2 = t9002[["lat", "lon", "vel", "cou"]].values.astype(np.float32)

        label = 1 if (len(pd.unique(t9001["MMSI"])) == 1 and 
                      len(pd.unique(t9002["MMSI"])) == 1 and 
                      t9001["MMSI"].iloc[0] == t9002["MMSI"].iloc[0]) else 0

        pairs.append((f1, f2, label))

    return pairs

# =====================================================================
# 主程序
# =====================================================================
def main():
    print("="*70)
    print("  【样本对内部动态对齐】CSV → NPY 重生成")
    print("  不搞全局统一长度，只在对内部插值对齐")
    print("="*70)

    all_pairs = []
    max_scene = 5000

    for sid in range(max_scene):
        if sid % 500 == 0:
            print(f"处理场景：{sid} ~ {sid+499}")
        pairs = load_scene_pairs(sid)
        all_pairs.extend(pairs)

    print(f"\n总样本对：{len(all_pairs)}")
    if len(all_pairs) == 0:
        print("❌ 无样本")
        return

    # 【核心】对内部对齐
    print("样本对内部动态对齐中...")
    track1_list = []
    track2_list = []
    labels = []
    lengths = []

    for f1, f2, lab in all_pairs:
        f1_aligned, f2_aligned, max_len = interpolate_pair_aligned(f1, f2)
        track1_list.append(f1_aligned)
        track2_list.append(f2_aligned)
        labels.append(lab)
        lengths.append(max_len)

    labels = np.array(labels, dtype=np.float32)
    lengths = np.array(lengths)

    # 划分
    idx = np.arange(len(labels))
    idx_train, idx_tmp = train_test_split(idx, test_size=1-TRAIN_RATIO, stratify=labels, random_state=42)
    idx_val, idx_test     = train_test_split(idx_tmp, test_size=TEST_RATIO/(TEST_RATIO+VAL_RATIO), stratify=labels[idx_tmp], random_state=42)

    len_test = lengths[idx_test]
    idx_short  = idx_test[len_test < SHORT_THRESH]
    idx_medium = idx_test[(len_test >= SHORT_THRESH) & (len_test <= LONG_THRESH)]
    idx_long   = idx_test[len_test > LONG_THRESH]

    # 全局标准化（拼接所有特征）
    print("全局标准化中...")
    all_feats = []
    for i in idx_train:
        all_feats.append(track1_list[i])
        all_feats.append(track2_list[i])
    all_feats = np.concatenate(all_feats)
    
    scaler = StandardScaler().fit(all_feats)
    np.save(os.path.join(SAVE_DIR, "scaler_mean.npy"),  scaler.mean_.astype(np.float32))
    np.save(os.path.join(SAVE_DIR, "scaler_scale.npy"), scaler.scale_.astype(np.float32))

    def norm(x):
        return scaler.transform(x).astype(np.float32)

    # 保存函数（保存list格式，支持变长）
    def save_split(name, idx):
        t1 = [norm(track1_list[i]) for i in idx]
        t2 = [norm(track2_list[i]) for i in idx]
        lab = labels[idx]
        np.save(f"{SAVE_DIR}/track1_{name}.npy", np.array(t1, dtype=object))
        np.save(f"{SAVE_DIR}/track2_{name}.npy", np.array(t2, dtype=object))
        np.save(f"{SAVE_DIR}/labels_{name}.npy", lab)

    save_split("train", idx_train)
    save_split("val",   idx_val)
    save_split("test",  idx_test)
    save_split("test_short",  idx_short)
    save_split("test_medium", idx_medium)
    save_split("test_long",   idx_long)

    print("\n" + "="*70)
    print("✅ 样本对内部动态对齐完成！")
    print(f"保存路径：{SAVE_DIR}")
    print(f"特性：不搞全局统一长度，只在对内部插值对齐")
    print("提示：训练时需要使用 collate_fn 进行动态 batch padding")
    print("="*70)

if __name__ == "__main__":
    main()