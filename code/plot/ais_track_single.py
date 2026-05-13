import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 配置区 =====================
# 你的单个MMSI CSV文件路径
CSV_PATH = "/home/yangcq/track_association/data/AIS/track3381234.csv"
# 图片保存路径
OUTPUT_PATH = "/home/yangcq/track_association/output/single_track.png"
# ===============================================

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_csv_with_auto_encoding(file_path):
    """自动尝试多种编码读取CSV"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    for enc in encodings:
        try:
            print(f"尝试使用 {enc} 编码读取...")
            df = pd.read_csv(file_path, encoding=enc)
            print(f"✅ 成功使用 {enc} 编码读取！")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用 {enc} 编码时出错：{e}")
            continue
    raise ValueError("无法读取CSV文件，请检查文件格式或手动指定编码")

# 1. 读取数据（自动检测编码）
print(f"正在读取：{CSV_PATH}")
df = read_csv_with_auto_encoding(CSV_PATH)

# 2. 按时间排序（保证航迹顺序正确）
df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
df = df.sort_values("BaseDateTime").reset_index(drop=True)

# 3. 绘制航迹
plt.figure(figsize=(12, 10))

# 绘制航迹线
plt.plot(df["LON"], df["LAT"], 'b-', linewidth=2, alpha=0.8, label='track')
# 绘制航迹点
plt.scatter(df["LON"], df["LAT"], color='blue', s=10, alpha=0.6)

# 标注起点（绿色）和终点（红色）
plt.scatter(df["LON"].iloc[0], df["LAT"].iloc[0], 
            color='green', s=80, marker='o', label='start')
plt.scatter(df["LON"].iloc[-1], df["LAT"].iloc[-1], 
            color='red', s=80, marker='x', label='end')

# 设置图表
mmsi = df["MMSI"].iloc[0]
plt.title(f"track\nMMSI: {mmsi}", fontsize=16)
plt.xlabel("lat (°)", fontsize=14)
plt.ylabel("lon (°)", fontsize=14)
plt.legend(fontsize=12)
plt.axis('equal')  # 等比例显示，避免航迹变形
plt.grid(True, alpha=0.3, linestyle='--')

# 保存图片
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
plt.close()

print("="*50)
print(f"✅ 航迹图已保存：{OUTPUT_PATH}")
print(f"✅ 航迹点数：{len(df)}")
print(f"✅ MMSI：{mmsi}")
print("="*50)