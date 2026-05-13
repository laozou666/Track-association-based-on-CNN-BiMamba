import pandas as pd
import matplotlib.pyplot as plt
import argparse

# ===================== 【只需修改这里】 =====================
parser = argparse.ArgumentParser(description='MMSI number to visualize')
parser.add_argument('--MMSI', type=str, required=True, help='MMSI number to visualize')
CSV_PATH = "/home/yangcq/track_association/data/inference/track1.csv"  # 你的CSV路径
TARGET_MMSI = parser.parse_args().MMSI  # 改成你想看的 MMSI 号码
# ==========================================================

# 1. 读取数据
df = pd.read_csv(CSV_PATH)

# 2. 筛选指定MMSI的航迹
track = df[df["MMSI"] == TARGET_MMSI].copy()

# 3. 按时间排序（必须！否则航迹会乱）
if "timestamp" in df.columns:
    track = track.sort_values("timestamp")

# 4. 判断是否有数据
if len(track) == 0:
    print(f"❌ 未找到 MMSI = {TARGET_MMSI} 的航迹")
else:
    print(f"✅ 找到航迹！总点数：{len(track)}")

    # 5. 绘制航迹
    plt.figure(figsize=(10, 6))
    plt.plot(track["lon"], track["lat"], 
             c="blue", 
             lw=1, 
             marker="o", 
             markersize=2, 
             label=f"MMSI: {TARGET_MMSI}")
    
    plt.title(f"Ship Track (MMSI: {TARGET_MMSI})", fontsize=14)
    plt.xlabel("Longitude (经度)", fontsize=12)
    plt.ylabel("Latitude (纬度)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图片 + 显示
    plt.savefig(f"track_mmsi_{TARGET_MMSI}.png", dpi=150)
    #plt.show()