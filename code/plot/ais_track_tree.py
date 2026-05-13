import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# ===================== 配置区 =====================
# 包含多个MMSI的CSV文件路径
CSV_PATH = "/home/yangcq/track_association/data/AIS/AIS_2024_01_01.csv"
# 图片保存目录
OUTPUT_DIR = "/home/yangcq/track_association/output/figures"
# 要随机挑选的MMSI数量
NUM_TRACKS = 3
# 过滤过短航迹（点数少于该值的不绘制）
MIN_POINTS = 20
# ===============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_csv_smart(file_path):
    """智能读取CSV，自动处理编码问题"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for enc in encodings:
        try:
            print(f"尝试 {enc} 编码...")
            return pd.read_csv(file_path, encoding=enc)
        except:
            continue
    raise ValueError("无法读取CSV文件，请检查文件格式")

def identify_columns(df):
    """自动识别AIS数据的关键列"""
    # 识别MMSI列
    mmsi_col = None
    for col in df.columns:
        if 'mmsi' in col.lower():
            mmsi_col = col
            break
    
    # 识别时间列
    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            time_col = col
            break
    
    # 识别经纬度列
    lat_col = None
    lon_col = None
    for col in df.columns:
        if 'lat' in col.lower():
            lat_col = col
        if 'lon' in col.lower() or 'long' in col.lower():
            lon_col = col
    
    return mmsi_col, time_col, lat_col, lon_col

def plot_single_track(mmsi, track_data, lat_col, lon_col, save_path):
    """绘制单个MMSI的航迹"""
    plt.figure(figsize=(12, 10))
    
    # 绘制航迹
    plt.plot(track_data[lon_col], track_data[lat_col], 
             'b-', linewidth=2, alpha=0.8, label='track')
    plt.scatter(track_data[lon_col], track_data[lat_col], 
                color='blue', s=10, alpha=0.6)
    
    # 标注起点和终点
    plt.scatter(track_data[lon_col].iloc[0], track_data[lat_col].iloc[0], 
                color='green', s=80, marker='o', label='start')
    plt.scatter(track_data[lon_col].iloc[-1], track_data[lat_col].iloc[-1], 
                color='red', s=80, marker='x', label='end')
    
    # 设置图表
    plt.title(f"track_fig\nMMSI: {mmsi}", fontsize=16)
    plt.xlabel("lat (°)", fontsize=14)
    plt.ylabel("lon (°)", fontsize=14)
    plt.legend(fontsize=12)
    plt.axis('equal')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 保存图片
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print(" 随机挑选3条MMSI绘制航迹图")
    print("="*60)
    
    # 1. 读取数据
    print(f"\n[1/5] 读取CSV文件：{CSV_PATH}")
    df = read_csv_smart(CSV_PATH)
    print(f"✅ 读取成功，共 {len(df)} 个点")
    
    # 2. 识别列名
    print("\n[2/5] 识别数据列...")
    mmsi_col, time_col, lat_col, lon_col = identify_columns(df)
    print(f"✅ 识别结果：")
    print(f"   MMSI列: {mmsi_col}")
    print(f"   时间列: {time_col}")
    print(f"   纬度列: {lat_col}")
    print(f"   经度列: {lon_col}")
    
    if not all([mmsi_col, lat_col, lon_col]):
        print("❌ 无法识别关键列，请检查CSV文件列名")
        exit()
    
    # 3. 按MMSI分组，过滤过短航迹
    print("\n[3/5] 处理航迹数据...")
    valid_tracks = []
    for mmsi, group in df.groupby(mmsi_col):
        if len(group) >= MIN_POINTS:
            # 按时间排序
            if time_col:
                group[time_col] = pd.to_datetime(group[time_col], errors='coerce')
                group = group.dropna(subset=[time_col])
                group = group.sort_values(time_col).reset_index(drop=True)
            if len(group) >= MIN_POINTS:
                valid_tracks.append((mmsi, group))
    
    print(f"✅ 有效航迹数量：{len(valid_tracks)}")
    
    if len(valid_tracks) < NUM_TRACKS:
        print(f"❌ 有效航迹数量不足 {NUM_TRACKS} 条，请调整MIN_POINTS参数")
        exit()
    
    # 4. 随机挑选3条MMSI
    print(f"\n[4/5] 随机挑选 {NUM_TRACKS} 条航迹...")
    selected_tracks = random.sample(valid_tracks, NUM_TRACKS)
    print(f"✅ 选中的MMSI：{[mmsi for mmsi, _ in selected_tracks]}")
    
    # 5. 绘制航迹图
    print("\n[5/5] 绘制航迹图...")
    for i, (mmsi, track) in enumerate(selected_tracks):
        save_path = os.path.join(OUTPUT_DIR, f"track_{i+1}_mmsi_{mmsi}.png")
        plot_single_track(mmsi, track, lat_col, lon_col, save_path)
        print(f"  [{i+1}/{NUM_TRACKS}] 已保存：{save_path}")
    
    print("\n" + "="*60)
    print(f"🎉 全部完成！")
    print(f"✅ 共生成 {NUM_TRACKS} 张航迹图")
    print(f"✅ 图片保存目录：{OUTPUT_DIR}")
    print("="*60)