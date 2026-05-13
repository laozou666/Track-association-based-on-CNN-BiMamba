import pandas as pd
import matplotlib.pyplot as plt

# ===================== 【只改这里！】你的两个CSV文件路径 =====================
CSV_PATH_1 = "/home/yangcq/track_association/data/inference/track1.csv"  # 第一条航迹CSV
CSV_PATH_2 = "/home/yangcq/track_association/data/inference/track2.csv"  # 第二条航迹CSV
SAVE_DIR = "/home/yangcq/track_association/output/figures/track_fig"
FIG_NAME = "track_pair_plot.png"
SAVE_PATH = f"{SAVE_DIR}/{FIG_NAME}"
# ===================== 1. 读取CSV + 【核心】强制按time列排序 =====================
def load_and_sort_track(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='time', ascending=True).reset_index(drop=True)  # 锁死时序
    return df

df1 = load_and_sort_track(CSV_PATH_1)
df2 = load_and_sort_track(CSV_PATH_2)

print(f"✅ 航迹1加载完成：{len(df1)} 个点")
print(f"✅ 航迹2加载完成：{len(df2)} 个点")

# ===================== 2. 绘制连续航迹 =====================
plt.figure(figsize=(10, 6))

# 画航迹1（蓝色：线+点）
# plt.plot(df1["lon"], df1["lat"], color="#1f77b4", linewidth=2, label="Track 1")
plt.scatter(df1["lon"], df1["lat"], color="#1f77b4", s=20, alpha=0.7)

# 画航迹2（橙色：线+点）
# plt.plot(df2["lon"], df2["lat"], color="#ff7f0e", linewidth=2, label="Track 2")
plt.scatter(df2["lon"], df2["lat"], color="#ff7f0e", s=20, alpha=0.7)

# 标注起点（绿圈）
plt.scatter(df1["lon"].iloc[0], df1["lat"].iloc[0], color="green", s=150, edgecolor="black", label="Start")
plt.scatter(df2["lon"].iloc[0], df2["lat"].iloc[0], color="green", s=150, edgecolor="black")

# 标注终点（红叉）
plt.scatter(df1["lon"].iloc[-1], df1["lat"].iloc[-1], color="red", s=150, marker="x", linewidth=2, label="End")
plt.scatter(df2["lon"].iloc[-1], df2["lat"].iloc[-1], color="red", s=150, marker="x", linewidth=2)

# ===================== 3. 图表样式 =====================
plt.xlabel("Longitude (°)", fontsize=12)
plt.ylabel("Latitude (°)", fontsize=12)
plt.title("Two Tracks Visualization", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()

# 保存 + 显示
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n✅ 航迹图已保存：{SAVE_PATH}")