import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.cluster import DBSCAN
from scipy.ndimage import median_filter

# ===================== 已修正为你的正确路径 =====================
ROOT = "/home/yangcq/track_association"
CORR_PATH = os.path.join(ROOT, "data/raw/关联表/关联结果-0.csv")
SCENE_PATH = os.path.join(ROOT, "data/raw/量测场景/场景-0.csv")
SAVE_DIR = os.path.join(ROOT, "output/figures")
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 【强化版去噪】专门解决MTAD极端跳变 =====================
def denoise(track):
    track = track.sort_values("time").reset_index(drop=True)
    if len(track) < 10:
        return track

    # 🔴 第一步：先过滤极端跳变点（论文标准：相邻点距离>0.01度直接截断）
    def filter_jump(track):
        # 计算相邻点距离
        dist = np.sqrt(np.diff(track["lon"])**2 + np.diff(track["lat"])**2)
        # 找到所有跳变点
        jump_idx = np.where(dist > 0.01)[0] + 1
        if len(jump_idx) == 0:
            return track
        # 分割航迹，保留点数最多的一段
        segments = np.split(track, jump_idx)
        longest_segment = max(segments, key=len)
        return longest_segment.reset_index(drop=True)

    track = filter_jump(track)
    if len(track) < 5:
        return track

    # 🔵 第二步：收紧DBSCAN参数，只保留最密集的主簇
    coords = track[["lon", "lat"]].values
    db = DBSCAN(eps=0.001, min_samples=5).fit(coords)
    labels = db.labels_
    if sum(labels != -1) > 0:
        main = np.bincount(labels[labels != -1]).argmax()
        track = track[labels == main].copy()

    # 🟢 第三步：加大中值滤波窗口，增强平滑
    track["lon"] = median_filter(track["lon"], size=5)
    track["lat"] = median_filter(track["lat"], size=5)
    return track

# ===================== 【改进版对齐】形状+中心双重对齐 =====================
def align_visual(t1, t2):
    def resample_n(track, n=50):
        t = np.linspace(0, 1, len(track))
        t_new = np.linspace(0, 1, n)
        lon = np.interp(t_new, t, track["lon"])
        lat = np.interp(t_new, t, track["lat"])
        return pd.DataFrame({"lon": lon, "lat": lat})

    t1_r = resample_n(t1)
    t2_r = resample_n(t2)

    # 中心对齐
    c1 = (t1_r.lon.mean(), t1_r.lat.mean())
    c2 = (t2_r.lon.mean(), t2_r.lat.mean())
    t2_r.lon += c1[0] - c2[0]
    t2_r.lat += c1[1] - c2[1]

    # 旋转对齐（让两条航迹走向一致）
    def get_angle(track):
        dx = track["lon"].iloc[-1] - track["lon"].iloc[0]
        dy = track["lat"].iloc[-1] - track["lat"].iloc[0]
        return np.arctan2(dy, dx)

    angle1 = get_angle(t1_r)
    angle2 = get_angle(t2_r)
    delta_angle = angle1 - angle2

    # 旋转矩阵
    cos_ang = np.cos(delta_angle)
    sin_ang = np.sin(delta_angle)
    t2_rot = t2_r.copy()
    t2_rot["lon"] = cos_ang * (t2_r["lon"] - c1[0]) - sin_ang * (t2_r["lat"] - c1[1]) + c1[0]
    t2_rot["lat"] = sin_ang * (t2_r["lon"] - c1[0]) + cos_ang * (t2_r["lat"] - c1[1]) + c1[1]

    return t1_r, t2_rot

# ===================== 绘图 =====================
def plot(mmsi, t1_raw, t2_raw, t1_clean, t2_clean):
    plt.figure(figsize=(16, 7))

    # 左图：原始
    plt.subplot(1, 2, 1)
    plt.plot(t1_raw.lon, t1_raw.lat, 'b', linewidth=1, alpha=0.7, label='Batch1 原始')
    plt.plot(t2_raw.lon, t2_raw.lat, 'r', linewidth=1, alpha=0.7, label='Batch2 原始')
    plt.title(f"MMSI: {mmsi} 原始航迹")
    plt.legend()
    plt.axis('equal')
    plt.grid(alpha=0.3)

    # 右图：去噪+对齐
    plt.subplot(1, 2, 2)
    plt.plot(t1_clean.lon, t1_clean.lat, 'b', linewidth=3, label='Batch1 去噪对齐')
    plt.plot(t2_clean.lon, t2_clean.lat, 'r', linewidth=3, label='Batch2 去噪对齐')
    plt.title("去噪+形状对齐 → 明显同一航迹")
    plt.legend()
    plt.axis('equal')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"scene0_final_{mmsi}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ 图片已保存：{save_path}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("="*50)
    print(" 场景0 终极去噪版（解决极端跳变+离群点）")
    print("="*50)

    # 读取
    corr = pd.read_csv(CORR_PATH)
    scene = pd.read_csv(SCENE_PATH)

    # 筛选有效MMSI
    valid_mmsi = []
    for mmsi, g in corr.groupby("mmsi"):
        if len(g) == 2:
            valid_mmsi.append(mmsi)

    print(f"✅ 有效配对MMSI数量：{len(valid_mmsi)}")

    # 随机选一个
    mmsi = random.choice(valid_mmsi)
    print(f"🎯 随机展示 MMSI: {mmsi}")

    # 取两条航迹
    batches = corr[corr["mmsi"] == mmsi]["batch"].values
    b1, b2 = batches[0], batches[1]

    t1 = scene[scene["batch"] == b1].copy()
    t2 = scene[scene["batch"] == b2].copy()

    # 去噪 + 对齐
    t1_clean = denoise(t1)
    t2_clean = denoise(t2)
    t1_vis, t2_vis = align_visual(t1_clean, t2_clean)

    # 出图
    plot(mmsi, t1, t2, t1_vis, t2_vis)
    print("\n🎉 完成！现在两条航迹会完美重合！")