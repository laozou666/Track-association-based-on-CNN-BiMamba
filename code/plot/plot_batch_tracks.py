import pandas as pd
import matplotlib.pyplot as plt
import argparse

# ===================== 配置 =====================
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ===================== 命令行参数 =====================
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str,default="/home/yangcq/track_association/data/final_dataset/simulated_scenes_csv/场景-0.csv", help="要可视化的CSV文件路径")
parser.add_argument("--batch", type=int, required=True, help="要可视化的batch号")
args = parser.parse_args()

# ===================== 加载数据 =====================
df = pd.read_csv(args.csv)

# 筛选当前batch的两条航迹
df_batch = df[df["batch"] == args.batch].copy()
t1 = df_batch[df_batch["source"] == 9001]  # 源A
t2 = df_batch[df_batch["source"] == 9002]  # 源B



print(f"Batch {args.batch} 加载完成：")
print(f"  9001 点数：{len(t1)}")
print(f"  9002 点数：{len(t2)}")

if len(t1) == 0 or len(t2) == 0:
    print("❌ 该batch缺少9001或9002航迹！")
    exit()

# ===================== 绘图 =====================
plt.figure()

# 画航迹散点
plt.scatter(t1["lon"], t1["lat"], c="#1f77b4", s=30, alpha=0.8, label="Track 1 (9001)")
plt.scatter(t2["lon"], t2["lat"], c="#ff7f0e", s=30, alpha=0.8, label="Track 2 (9002)")

# 起点(绿圈) + 终点(红叉)
plt.scatter(t1["lon"].iloc[0], t1["lat"].iloc[0], c="green", s=250, edgecolors="black", label="Start")
plt.scatter(t1["lon"].iloc[-1], t1["lat"].iloc[-1], c="red", s=250, marker="x", linewidth=3, label="End")
plt.scatter(t2["lon"].iloc[0], t2["lat"].iloc[0], c="green", s=250, edgecolors="black")
plt.scatter(t2["lon"].iloc[-1], t2["lat"].iloc[-1], c="red", s=250, marker="x", linewidth=3)

# 样式
plt.title(f"Batch {args.batch} | Track Pair Visualization", fontsize=16)
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存 + 显示
output_dir = "/home/yangcq/track_association/output/figures/track_fig"
save_name = f"batch_{args.batch}_pair.png"
save_path = f"{output_dir}/{save_name}"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"✅ 可视化已保存：{save_name}")