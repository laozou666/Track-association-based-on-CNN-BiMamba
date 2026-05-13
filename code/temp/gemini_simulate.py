import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simulate_radar_track(track_df, drop_rate, gap_prob, gap_max_len, 
                         lat_offset, lon_offset, base_noise_std, 
                         start_trim=0, end_trim=0):
    """
    针对单条航迹进行雷达监测仿真
    """
    sim = track_df.copy()
    
    # 1. 模拟航迹长短不一致（雷达探测起止时间不同）
    if end_trim > 0:
        sim = sim.iloc[start_trim:-end_trim]
    else:
        sim = sim.iloc[start_trim:]
        
    # 2. 模拟雷达系统误差（航迹偏移）
    sim['lat'] += lat_offset
    sim['lon'] += lon_offset
    
    # 3. 模拟测量噪声（基础高斯噪声）
    sim['lat'] += np.random.normal(0, base_noise_std, len(sim))
    sim['lon'] += np.random.normal(0, base_noise_std, len(sim))
    
    # 4. 增加突发大噪声点（模拟杂波/野值）
    # 随机选取 5% 的点注入较大噪声
    outlier_count = int(len(sim) * 0.05)
    if outlier_count > 0:
        outlier_idx = np.random.choice(sim.index, size=outlier_count, replace=False)
        sim.loc[outlier_idx, 'lat'] += np.random.normal(0, base_noise_std * 5, len(outlier_idx))
        sim.loc[outlier_idx, 'lon'] += np.random.normal(0, base_noise_std * 5, len(outlier_idx))
        
    # 5. 模拟航迹中断（目标丢失/遮挡）
    mask = np.ones(len(sim), dtype=bool)
    i = 0
    while i < len(sim):
        if np.random.rand() < gap_prob:
            # 随机产生 5 到 gap_max_len 个点的连续中断
            gap = np.random.randint(5, gap_max_len)
            mask[i:i+gap] = False
            i += gap
        else:
            i += 1
    sim = sim[mask]
    
    # 6. 模拟不同雷达采样率/点数不同（随机丢弃一定比例的点）
    sim = sim.sample(frac=1-drop_rate).sort_values('time')
    
    return sim

# ================= 主程序 =================

# 1. 读取数据
file_path = '/home/yangcq/track_association/data/simulated_test/场景-0.csv'
df = pd.read_csv(file_path)

# 2. 选取数据点最多的 MMSI 作为基准
target_mmsi = df['MMSI'].value_counts().idxmax()
base_track = df[df['MMSI'] == target_mmsi].sort_values('time').copy()
print(f"选取的基准 MMSI: {target_mmsi}，原始点数: {len(base_track)}")

# 3. 生成雷达 A 仿真航迹
# 特点：轻微偏移，噪声较小，尾部丢失，采样率较高
radar_a = simulate_radar_track(
    track_df=base_track, 
    drop_rate=0.2,          # 整体随机丢弃20%点
    gap_prob=0.03,          # 产生中断的概率
    gap_max_len=15,         # 最大连续中断点数
    lat_offset=0.015,       # 纬度系统偏移
    lon_offset=-0.010,      # 经度系统偏移
    base_noise_std=0.005,   # 基础噪声强度
    start_trim=0,           # 头部不截断
    end_trim=20             # 尾部截断20个点
)

# 4. 生成雷达 B 仿真航迹
# 特点：偏移较大，噪声较大，头部丢失，采样率较低，中断更频繁
radar_b = simulate_radar_track(
    track_df=base_track, 
    drop_rate=0.4,          # 整体随机丢弃40%点
    gap_prob=0.06,          # 产生中断的概率较高
    gap_max_len=20,         # 最大连续中断点数
    lat_offset=-0.020,      # 纬度系统偏移
    lon_offset=0.025,       # 经度系统偏移
    base_noise_std=0.008,   # 基础噪声强度更高
    start_trim=30,          # 头部截断30个点（雷达B较晚发现目标）
    end_trim=0              # 尾部不截断
)

print(f"雷达 A 仿真点数: {len(radar_a)}")
print(f"雷达 B 仿真点数: {len(radar_b)}")

# 5. 可视化并保存图表（不显示）
plt.figure(figsize=(12, 8))

# 绘制原始真实航迹作为参考（半透明虚线）
plt.plot(base_track['lon'], base_track['lat'], color='gray', 
         linestyle='--', linewidth=2, alpha=0.5, label='True Trajectory')

# 绘制雷达A和雷达B的散点
plt.scatter(radar_a['lon'], radar_a['lat'], color='blue', s=15, 
            label='Radar A (Simulated)', alpha=0.8)
plt.scatter(radar_b['lon'], radar_b['lat'], color='red', s=15, 
            label='Radar B (Simulated)', alpha=0.8, marker='^')

plt.title(f'Simulated Radar Tracks for MMSI: {target_mmsi}', fontsize=14)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# 保存文件并关闭画板
output_filename = f'simulated_tracks_{target_mmsi}.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"仿真图表已成功保存为: {output_filename}")