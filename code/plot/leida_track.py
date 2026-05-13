import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 配置区 =====================
# 两个AIS航迹文件路径
TRACK1_PATH = "/home/yangcq/track_association/data/AIS_Trajectory_352003977_2002.csv"
TRACK2_PATH = "/home/yangcq/track_association/data/AIS_Trajectory_352003977_2003.csv"
# 输出图片路径
OUTPUT_PATH = "/home/yangcq/track_association/output/figures/leida_fig.png"
# 航迹颜色
COLOR1 = "blue"
COLOR2 = "red"
# =====================================================================

def load_ais_track(csv_path):
    """加载AIS航迹CSV文件，提取经纬度"""
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在：{csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    # 提取经纬度（匹配你提供的CSV列名）
    lon = df["经度(°)"].values
    lat = df["纬度(°)"].values
    print(f"✅ 加载航迹：{os.path.basename(csv_path)}")
    print(f"   点数：{len(lon)}")
    print(f"   经度范围：{lon.min():.4f} ~ {lon.max():.4f}")
    print(f"   纬度范围：{lat.min():.4f} ~ {lat.max():.4f}")
    return lon, lat

if __name__ == "__main__":
    print("="*70)
    print("双AIS航迹叠加可视化")
    print("="*70)

    # 1. 加载航迹
    print("\n[1/2] 加载航迹数据...")
    lon1, lat1 = load_ais_track(TRACK1_PATH)
    lon2, lat2 = load_ais_track(TRACK2_PATH)
    
    if lon1 is None or lon2 is None:
        exit()

    # 2. 绘制航迹
    print("\n[2/2] 绘制航迹...")
    plt.figure(figsize=(10, 8))
    
    # 绘制航迹1（蓝色）
    plt.plot(lon1, lat1, color=COLOR1, linewidth=2, alpha=0.8, label='Source 2002')
    plt.scatter(lon1, lat1, color=COLOR1, s=15, alpha=0.7)
    plt.scatter(lon1[0], lat1[0], color='green', s=150, marker='o', label='Start')
    plt.scatter(lon1[-1], lat1[-1], color='red', s=150, marker='x', label='End')
    
    # 绘制航迹2（红色）
    plt.plot(lon2, lat2, color=COLOR2, linewidth=2, alpha=0.8, label='Source 2003')
    plt.scatter(lon2, lat2, color=COLOR2, s=15, alpha=0.7)
    plt.scatter(lon2[0], lat2[0], color='green', s=150, marker='o')
    plt.scatter(lon2[-1], lat2[-1], color='red', s=150, marker='x')
    
    # 设置图表属性
    plt.title(f"AIS Track Comparison (MMSI: 352003977)", fontsize=16)
    plt.xlabel("Longitude (°)", fontsize=14)
    plt.ylabel("Latitude (°)", fontsize=14)
    plt.legend(fontsize=12)
    plt.axis('equal')  # 等比例显示，避免航迹变形
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 保存并显示
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print(f"✅ 可视化完成！")
    print(f"✅ 图片已保存到：{OUTPUT_PATH}")
    print("="*70)