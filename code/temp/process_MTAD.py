import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm  # 进度条库（如果没有，pip install tqdm）

# ===================== 【只改这里】你的路径 =====================
ROOT_DIR = "/home/yangcq/data_raw/MTAD_data"   # 必改！你的MTAD_data路径
LINK_DIR = os.path.join(ROOT_DIR, "关联表")
MEAS_DIR = os.path.join(ROOT_DIR, "量测场景")
OUTPUT_DIR = "/home/yangcq/data_process/MTAD"                  # 输出结果文件夹
SEQ_LEN = 100
FEATURE_COLS = ["lat", "lon", "vel", "cou"]
POS_NEG_RATIO = 1/3
# ====================================================================

# 解决绘图问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "训练集"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "验证集"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "测试集"), exist_ok=True)

# -------------------------- 航迹长度统一 --------------------------
def pad_or_truncate_traj(traj_df):
    traj_sorted = traj_df.sort_values("time").reset_index(drop=True)
    traj = traj_sorted[FEATURE_COLS].values
    if len(traj) >= SEQ_LEN:
        return traj[:SEQ_LEN]
    else:
        pad = ((0, SEQ_LEN - len(traj)), (0, 0))
        return np.pad(traj, pad, mode="constant", constant_values=0)

# -------------------------- 单场景处理 --------------------------
def process_single_scene(scene_id):
    link_path = os.path.join(LINK_DIR, f"关联结果-{scene_id}.csv")
    meas_path = os.path.join(MEAS_DIR, f"场景-{scene_id}.csv")

    if not os.path.exists(link_path) or not os.path.exists(meas_path):
        return None, None, None, None

    df_link = pd.read_csv(link_path)
    df_meas = pd.read_csv(meas_path)
    df_link.columns = [str(c).strip().lower() for c in df_link.columns]
    df_meas.columns = [str(c).strip().lower() for c in df_meas.columns]

    # 筛选同时有9001+9002的MMSI
    mmsi_groups = df_link.groupby("mmsi")["source"].unique()
    valid_mmsi = [m for m, s in mmsi_groups.items() if set(s) == {9001, 9002}]
    if len(valid_mmsi) < 2:
        return None, None, None, None

    # 构建航迹字典
    traj_dict = {}
    for mmsi in valid_mmsi:
        for s in [9001, 9002]:
            batch = df_link[(df_link.mmsi == mmsi) & (df_link.source == s)]["batch"].unique()
            if len(batch) == 0:
                continue
            b = batch[0]
            traj_df = df_meas[df_meas.batch == b].copy()
            if len(traj_df) == 0:
                continue
            traj_dict[(mmsi, s)] = pad_or_truncate_traj(traj_df)

    # 正样本
    pos_pairs, pos_meta = [], []
    for mmsi in valid_mmsi:
        k1 = (mmsi, 9001)
        k2 = (mmsi, 9002)
        if k1 in traj_dict and k2 in traj_dict:
            pos_pairs.append([traj_dict[k1], traj_dict[k2]])
            pos_meta.append({
                "scene_id": scene_id, "mmsi": mmsi,
                "source_a": 9001, "source_b": 9002, "label": 1
            })

    pos_num = len(pos_pairs)
    if pos_num == 0:
        return None, None, None, None

    # 负样本 1:3
    neg_target = int(pos_num / POS_NEG_RATIO)
    neg_pairs, neg_meta = [], []
    s1_list = [k for k in traj_dict if k[1] == 9001]
    s2_list = [k for k in traj_dict if k[1] == 9002]
    used = set()
    attempts = 0

    while len(neg_pairs) < neg_target and attempts < neg_target * 10:
        attempts += 1
        ka = random.choice(s1_list)
        kb = random.choice(s2_list)
        ma, sa = ka
        mb, sb = kb
        if ma == mb:
            continue
        key = tuple(sorted([ka, kb]))
        if key in used:
            continue
        neg_pairs.append([traj_dict[ka], traj_dict[kb]])
        neg_meta.append({
            "scene_id": scene_id, "mmsi_a": ma, "mmsi_b": mb,
            "source_a": sa, "source_b": sb, "label": 0
        })
        used.add(key)

    X = np.array(pos_pairs + neg_pairs, dtype=np.float32)
    y = np.array([1]*pos_num + [0]*len(neg_pairs), dtype=np.float32)
    meta = pos_meta + neg_meta
    return X, y, meta, traj_dict

# -------------------------- 批量处理函数 --------------------------
def batch_process(scene_id_list, save_dir, set_name, scaler=None):
    all_X, all_y = [], []
    all_meta = []
    all_traj = []  # 收集航迹用于训练集拟合归一化器

    print(f"\n===== 开始处理【{set_name}】- 共{len(scene_id_list)}个场景 =====")
    for sid in tqdm(scene_id_list, desc=f"Processing {set_name}"):
        X, y, meta, traj_dict = process_single_scene(sid)
        if X is None:
            continue
        all_X.append(X)
        all_y.append(y)
        all_meta.extend(meta)
        [all_traj.append(t) for t in traj_dict.values()]

    if not all_X:
        print(f"❌ {set_name} 无有效数据！")
        return None, None, None, None

    # 合并数据
    X_comb = np.concatenate(all_X, axis=0)
    y_comb = np.concatenate(all_y, axis=0)
    df_meta = pd.DataFrame(all_meta)
    traj_comb = np.concatenate(all_traj, axis=0) if all_traj else None

    # 保存原始数据和元数据
    np.save(os.path.join(save_dir, "X_原始.npy"), X_comb)
    np.save(os.path.join(save_dir, "y_标签.npy"), y_comb)
    df_meta.to_csv(os.path.join(save_dir, "样本元数据.csv"), index=False, encoding="utf-8-sig")

    # 归一化（若传入归一化器）
    if scaler and traj_comb is not None:
        X_norm = scaler.transform(X_comb.reshape(-1, len(FEATURE_COLS))).reshape(X_comb.shape)
        np.save(os.path.join(save_dir, "X_归一化.npy"), X_norm)
        print(f"✅ {set_name} 归一化数据已保存")

    # 统计信息
    pos = int(sum(y_comb))
    neg = len(y_comb) - pos
    print(f"✅ {set_name} 处理完成！")
    print(f"📊 统计：总样本{len(X_comb)} | 正{pos} | 负{neg} | 比例{pos}:{neg}")
    print(f"📈 形状：X={X_comb.shape} | y={y_comb.shape}")
    return X_comb, y_comb, traj_comb, df_meta

# ===================== 主程序：全量5000场景处理 =====================
if __name__ == "__main__":
    print("="*80)
    print("开始全量5000场景处理（正负1:3 | 7:2:1划分）")
    print("="*80)

    # 1. 扫描所有有效场景ID
    print("\n正在扫描所有场景文件...")
    all_scene_ids = []
    for f in os.listdir(LINK_DIR):
        if f.startswith("关联结果-") and f.endswith(".csv"):
            try:
                sid = int(f.split("-")[1].split(".")[0])
                if os.path.exists(os.path.join(MEAS_DIR, f"场景-{sid}.csv")):
                    all_scene_ids.append(sid)
            except:
                continue
    all_scene_ids = sorted(list(set(all_scene_ids)))
    print(f"✅ 共找到 {len(all_scene_ids)} 个有效场景")

    # 2. 场景级7:2:1划分（避免数据泄露）
    train_ids, temp_ids = train_test_split(all_scene_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=1/3, random_state=42)
    print(f"📚 场景划分：训练集{len(train_ids)} | 验证集{len(val_ids)} | 测试集{len(test_ids)}")

    # 3. 处理训练集【仅用训练集拟合归一化器】
    train_X, train_y, train_traj, _ = batch_process(
        train_ids, os.path.join(OUTPUT_DIR, "训练集"), "训练集"
    )

    if train_traj is None:
        print("❌ 训练集无有效航迹，终止处理！")
        exit()

    # 拟合并保存全局归一化器
    print("\n" + "="*80)
    print("拟合全局MinMax归一化器（仅用训练集数据）...")
    print("="*80)
    scaler = MinMaxScaler()
    scaler.fit(train_traj.reshape(-1, len(FEATURE_COLS)))
    scaler_path = os.path.join(OUTPUT_DIR, "全局特征归一化器.save")
    joblib.dump(scaler, scaler_path)
    print(f"✅ 归一化器已保存：{scaler_path}")

    # 训练集数据归一化并保存
    train_X_norm = scaler.transform(train_X.reshape(-1, len(FEATURE_COLS))).reshape(train_X.shape)
    np.save(os.path.join(OUTPUT_DIR, "训练集", "X_归一化.npy"), train_X_norm)
    print(f"✅ 训练集归一化数据已保存")

    # 4. 处理验证集（用训练集归一化器）
    batch_process(val_ids, os.path.join(OUTPUT_DIR, "验证集"), "验证集", scaler=scaler)

    # 5. 处理测试集（用训练集归一化器）
    batch_process(test_ids, os.path.join(OUTPUT_DIR, "测试集"), "测试集", scaler=scaler)

    # 最终汇总
    print("\n" + "="*80)
    print("🎉 全量5000场景处理完成！")
    print("="*80)
    print(f"📁 最终输出目录：{os.path.abspath(OUTPUT_DIR)}")
    print(f"📋 数据说明：")
    print("  - X_原始.npy：未归一化航迹，形状(样本数,2,100,4)")
    print("  - X_归一化.npy：归一化后数据，可直接喂入CNN/LSTM模型")
    print("  - y_标签.npy：0/1标签（0=负样本，1=正样本）")
    print("  - 样本元数据.csv：样本溯源（MMSI/信源/场景ID）")
    print("  - 全局特征归一化器.save：训练集拟合，推理时直接加载")
    print("="*80)