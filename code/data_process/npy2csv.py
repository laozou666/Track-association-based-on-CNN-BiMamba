import numpy as np
import pandas as pd
import os

# ===================== 路径（和你项目100%匹配）=====================
DATA_ROOT = "/home/yangcq/track_association/data/final_dataset"
SAVE_ROOT = os.path.join(DATA_ROOT, "test_csv")

# 加载归一化参数
mean = np.load(os.path.join(DATA_ROOT, "scaler_mean.npy"))
scale = np.load(os.path.join(DATA_ROOT, "scaler_scale.npy"))

# 三类长度
CATEGORIES = ["short", "medium", "long"]

# ===================== 工具函数：反归一化 + 去除补零 =====================
def denorm_and_trim(track_seq):
    real = track_seq * scale + mean
    real = real[~np.all(real == 0, axis=1)]
    return real

# ===================== 生成CSV =====================
def save_track_csv(real_track, save_path, source_id, batch_id):
    time_steps = np.linspace(10, 7000, len(real_track)).astype(int)
    df = pd.DataFrame({
        "MMSI": [f"SHIP-{source_id}-{i}" for i in range(len(real_track))],
        "time": time_steps,
        "lat": real_track[:, 0],
        "lon": real_track[:, 1],
        "vel": real_track[:, 2],
        "cou": real_track[:, 3],
        "source": source_id,
        "batch": batch_id
    })
    df.to_csv(save_path, index=False)

# ===================== 批量还原 + 按标签分类 =====================
for cate in CATEGORIES:
    print(f"\n========== 正在还原 {cate} 测试集（按标签分类）==========")
    
    # 加载npy
    track1 = np.load(os.path.join(DATA_ROOT, f"track1_test_{cate}.npy"), allow_pickle=True)
    track2 = np.load(os.path.join(DATA_ROOT, f"track2_test_{cate}.npy"), allow_pickle=True)
    labels = np.load(os.path.join(DATA_ROOT, f"labels_test_{cate}.npy"), allow_pickle=True)

    print(f"加载完成：{len(track1)} 个航迹对")

    # 创建根目录
    cate_root = os.path.join(SAVE_ROOT, cate)
    os.makedirs(cate_root, exist_ok=True)

    # 创建 label_0 和 label_1 子目录
    label0_dir = os.path.join(cate_root, "label_0")
    label1_dir = os.path.join(cate_root, "label_1")
    os.makedirs(label0_dir, exist_ok=True)
    os.makedirs(label1_dir, exist_ok=True)

    # 统计数量
    cnt0, cnt1 = 0, 0

    # 逐个样本生成CSV
    for idx in range(len(track1)):
        # 反归一化 + 去补零
        t1_real = denorm_and_trim(track1[idx])
        t2_real = denorm_and_trim(track2[idx])

        if len(t1_real) < 5 or len(t2_real) < 5:
            continue

        # 获取标签
        label = int(labels[idx])

        # 确定保存目录
        target_dir = label1_dir if label == 1 else label0_dir
        if label == 1:
            cnt1 += 1
        else:
            cnt0 += 1

        # 保存路径
        t1_path = os.path.join(target_dir, f"{cate}_sample_{idx}_track1.csv")
        t2_path = os.path.join(target_dir, f"{cate}_sample_{idx}_track2.csv")

        # 生成CSV
        save_track_csv(t1_real, t1_path, source_id=9001, batch_id=idx)
        save_track_csv(t2_real, t2_path, source_id=9002, batch_id=idx)

    print(f"✅ {cate} 还原完成！")
    print(f"   不关联 (label=0): {cnt0} 对 → {label0_dir}")
    print(f"   关联 (label=1): {cnt1} 对 → {label1_dir}")

print("\n🎉 全部还原完成！已按标签分类！")
print("📁 输出目录结构：")
print("test_csv/")
print("  ├── short/")
print("  │   ├── label_0/  (不关联样本)")
print("  │   └── label_1/  (关联样本)")
print("  ├── medium/")
print("  │   ├── label_0/")
print("  │   └── label_1/")
print("  └── long/")
print("      ├── label_0/")
print("      └── label_1/")