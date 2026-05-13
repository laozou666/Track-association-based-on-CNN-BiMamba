import numpy as np
import matplotlib.pyplot as plt

# ===================== 改成你的路径 =====================
NPY_DIR = "/home/yangcq/track_association/data/final_dataset"
# ========================================================

print("="*60)
print(" NPY文件直接验证")
print("="*60)

# 1. 加载标签
labels = np.load(f"{NPY_DIR}/labels_train.npy")
t1 = np.load(f"{NPY_DIR}/track1_train.npy")
t2 = np.load(f"{NPY_DIR}/track2_train.npy")

print(f"\n总样本数：{len(labels)}")
print(f"label=1 数量：{np.sum(labels == 1)}")
print(f"label=0 数量：{np.sum(labels == 0)}")

# 2. 打印前10个样本的标签
print(f"\n前10个样本的标签：")
for i in range(10):
    print(f"  样本{i}：label = {labels[i]}")

# 3. 可视化前3个样本，直观验证
print(f"\n可视化前3个样本...")
for i in range(3):
    plt.figure(figsize=(10, 6))
    
    # 反归一化（为了画图好看，假设你有scaler）
    try:
        mean = np.load(f"{NPY_DIR}/scaler_mean.npy")
        scale = np.load(f"{NPY_DIR}/scaler_scale.npy")
        tr1 = t1[i] * scale + mean
        tr2 = t2[i] * scale + mean
    except:
        tr1 = t1[i]
        tr2 = t2[i]
    
    plt.scatter(tr1[:, 1], tr1[:, 0], c='#1f77b4', s=30, label='Track 1')
    plt.scatter(tr2[:, 1], tr2[:, 0], c='#ff7f0e', s=30, label='Track 2')
    plt.title(f"Sample {i} | Label = {labels[i]}", fontsize=16)
    plt.legend()
    plt.savefig(f"check_sample_{i}.png", dpi=150)
    plt.close()
    print(f"  样本{i}已保存：check_sample_{i}.png")

print("\n" + "="*60)
print("✅ 验证完成！")
print("请查看：")
print("  1. 前10个标签打印")
print("  2. check_sample_0/1/2.png 三张图")
print("  如果图里两条航迹明显是同一条船，但label=0，说明标签反了")
print("="*60)