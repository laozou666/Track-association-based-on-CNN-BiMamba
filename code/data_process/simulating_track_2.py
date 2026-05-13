import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# ===================== 最终配置区 =====================
REAL_SCENE_ROOT = "/home/yangcq/track_association/data/raw/真实场景/"
OUTPUT_ROOT = "/home/yangcq/track_association/data/hard_scenes_with_negatives/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 完美平衡难度
GAP_RATIO = 0.25          # 25%有断点
LENGTH_DIFF_RATIO = 0.2   # 20%有长度差异
CROSS_RATIO = 0.3         # 30%有X形交叉
NORMAL_RATIO = 0.25       # 25%正常

# 负样本比例
NEGATIVE_RATIO = 0.5      # 50%正样本，50%负样本（平衡）
HARD_NEGATIVE_RATIO = 0.7 # 负样本中70%是“难分负样本”

MIN_TRACK_POINTS = 25

# 只处理场景0
SCENE_ID = 0
# =====================================================================

def simulate_track_from_real_final(truth_track, source_id, effect_type="normal"):
    """
    最终版：偏移量适中，速度航向也加噪声
    """
    track = truth_track.copy().sort_values("time").reset_index(drop=True)

    # 1. 基础采样
    if source_id == 9001:
        keep_ratio = np.random.uniform(0.85, 0.95)
    else:
        keep_ratio = np.random.uniform(0.95, 1.1)
    
    if keep_ratio <= 1.0:
        keep_mask = np.random.choice([True, False], size=len(track), p=[keep_ratio, 1-keep_ratio])
        keep_mask[0] = True
        keep_mask[-1] = True
        track = track[keep_mask].reset_index(drop=True)
    else:
        interp_ratio = keep_ratio
        t = np.linspace(0, 1, len(track))
        t_new = np.linspace(0, 1, int(len(track)*interp_ratio))
        
        track_interp = pd.DataFrame()
        track_interp["time"] = np.interp(t_new, t, track["time"])
        track_interp["lat"] = np.interp(t_new, t, track["lat"])
        track_interp["lon"] = np.interp(t_new, t, track["lon"])
        track_interp["vel"] = np.interp(t_new, t, track["vel"])
        track_interp["cou"] = np.interp(t_new, t, track["cou"])
        track_interp["MMSI"] = track["MMSI"].iloc[0]
        track = track_interp

    # 2. 效果处理
    if effect_type == "gap" and len(track) > 20:
        gap_start = int(len(track)*np.random.uniform(0.4, 0.6))
        gap_length = int(len(track)*np.random.uniform(0.05, 0.10))
        gap_end = gap_start + gap_length
        track = pd.concat([track.iloc[:gap_start], track.iloc[gap_end:]]).reset_index(drop=True)
    
    if effect_type == "length_diff":
        if source_id == 9001:
            crop_ratio = np.random.uniform(0, 0.1)
        else:
            crop_ratio = np.random.uniform(0.05, 0.12)
        start = int(len(track)*crop_ratio)
        end = len(track) - int(len(track)*crop_ratio)
        if end > start + 10:
            track = track.iloc[start:end].reset_index(drop=True)

    # 3. 噪声和偏移（经纬度）
    sys_err_lon = np.random.uniform(-0.004, 0.004)
    sys_err_lat = np.random.uniform(-0.004, 0.004)
    track["lon"] += sys_err_lon
    track["lat"] += sys_err_lat

    noise_std = 0.0001 if source_id == 9001 else 0.00015
    track["lon"] += np.random.normal(0, noise_std, len(track))
    track["lat"] += np.random.normal(0, noise_std, len(track))

    # 4. 【新增】速度和航向也加噪声
    vel_noise = np.random.normal(0, 0.3, len(track))  # 速度噪声：±0.3节
    cou_noise = np.random.normal(0, 2.0, len(track))   # 航向噪声：±2度
    track["vel"] += vel_noise
    track["cou"] += cou_noise
    track["cou"] = track["cou"] % 360  # 航向保持在0-360度

    # 5. 少量离群点
    if len(track) > 20:
        num_outliers = np.random.randint(1, 3)
        outlier_indices = np.random.choice(len(track), num_outliers, replace=False)
        for idx in outlier_indices:
            outlier_dist = np.random.uniform(0.0005, 0.001)
            outlier_angle = np.random.uniform(0, 2*np.pi)
            track.loc[idx, "lon"] += outlier_dist * np.cos(outlier_angle)
            track.loc[idx, "lat"] += outlier_dist * np.sin(outlier_angle)

    return track

def create_x_cross_tracks(track1, track2, num_crosses=2):
    """
    生成X形交叉：两条航迹在中间交叉，然后继续各自方向
    """
    min_len = min(len(track1), len(track2))
    track1 = track1.iloc[:min_len].reset_index(drop=True)
    track2 = track2.iloc[:min_len].reset_index(drop=True)
    
    cross_points = []
    for i in range(num_crosses):
        cross_point = int(min_len * np.random.uniform(0.25 + i*0.25, 0.45 + i*0.25))
        cross_points.append(cross_point)
    cross_points.sort()
    
    cross_offsets = []
    for _ in range(num_crosses):
        offset_lon = np.random.uniform(-0.001, 0.001)
        offset_lat = np.random.uniform(-0.001, 0.001)
        cross_offsets.append((offset_lon, offset_lat))
    
    for i in range(num_crosses):
        start = cross_points[i] - int(min_len*0.05)
        end = cross_points[i] + int(min_len*0.05)
        offset_lon, offset_lat = cross_offsets[i]
        
        track1.loc[start:end, "lon"] += offset_lon
        track1.loc[start:end, "lat"] += offset_lat
        track2.loc[start:end, "lon"] -= offset_lon
        track2.loc[start:end, "lat"] -= offset_lat
    
    return track1, track2

def create_hard_negative_track(truth_track, reference_track):
    """
    【核心新增】生成“难分负样本”：
    把不同目标的航迹强行平移到参考航迹附近，让它们看起来很像
    """
    track = truth_track.copy().sort_values("time").reset_index(drop=True)
    
    # 计算两条航迹的中心
    ref_center_lon = reference_track["lon"].mean()
    ref_center_lat = reference_track["lat"].mean()
    track_center_lon = track["lon"].mean()
    track_center_lat = track["lat"].mean()
    
    # 把当前航迹平移到参考航迹附近（加一点随机偏移，不要完全重合）
    offset_lon = ref_center_lon - track_center_lon + np.random.uniform(-0.003, 0.003)
    offset_lat = ref_center_lat - track_center_lat + np.random.uniform(-0.003, 0.003)
    
    track["lon"] += offset_lon
    track["lat"] += offset_lat
    
    return track

def visualize_pair(truth_track1, truth_track2, radar1, radar2, save_path, 
                   is_positive=True, is_hard_negative=False, 
                   is_crossed=False, num_crosses=0):
    """
    统一可视化函数：支持正样本、负样本、难分负样本
    """
    plt.figure(figsize=(16, 10))
    
    # 绘制真实航迹（灰色）
    if truth_track1 is not None:
        plt.scatter(truth_track1["lon"], truth_track1["lat"], color='gray', s=15, alpha=0.2, label='Ground Truth 1')
    if truth_track2 is not None and truth_track2 is not truth_track1:
        plt.scatter(truth_track2["lon"], truth_track2["lat"], color='darkgray', s=15, alpha=0.2, label='Ground Truth 2')
    
    # 绘制仿真航迹
    plt.scatter(radar1["lon"], radar1["lat"], color='#1f77b4', s=30, alpha=0.8, label='Source 9001')
    plt.scatter(radar1["lon"].iloc[0], radar1["lat"].iloc[0], color='green', s=150, marker='o', edgecolors='black', label='Start')
    plt.scatter(radar1["lon"].iloc[-1], radar1["lat"].iloc[-1], color='red', s=150, marker='x', linewidths=2, label='End')
    
    plt.scatter(radar2["lon"], radar2["lat"], color='#ff7f0e', s=30, alpha=0.8, label='Source 9002')
    plt.scatter(radar2["lon"].iloc[0], radar2["lat"].iloc[0], color='green', s=150, marker='o', edgecolors='black')
    plt.scatter(radar2["lon"].iloc[-1], radar2["lat"].iloc[-1], color='red', s=150, marker='x', linewidths=2)
    
    # 生成标题
    title = f'Scene {SCENE_ID} - '
    if is_positive:
        title += 'POSITIVE PAIR (Same Target)'
    else:
        if is_hard_negative:
            title += 'HARD NEGATIVE PAIR (Different Targets, Similar Area)'
        else:
            title += 'EASY NEGATIVE PAIR (Different Targets)'
    
    if is_crossed:
        title += f' [X-CROSS x{num_crosses}]'
    
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel('Longitude (°)', fontsize=14)
    plt.ylabel('Latitude (°)', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.axis('equal')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ 可视化保存到：{save_path}")
    plt.show()
    plt.close()

def process_scene_0_with_negatives():
    input_path = os.path.join(REAL_SCENE_ROOT, f"场景-{SCENE_ID}.csv")
    output_scene_path = os.path.join(OUTPUT_ROOT, f"场景-{SCENE_ID}_with_negatives.csv")
    output_corr_path = os.path.join(OUTPUT_ROOT, f"关联结果-{SCENE_ID}_with_negatives.csv")
    vis_dir = os.path.join(OUTPUT_ROOT, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"❌ 文件不存在：{input_path}")
        return

    print("="*80)
    print(f" 完善版仿真 - 场景 {SCENE_ID} (含正/负样本)")
    print("  ✅ 正样本：X形交叉、断点、长度差异")
    print("  ✅ 负样本：70%难分负样本（平移到同一区域）")
    print("  ✅ 速度和航向也加噪声")
    print("  ✅ 完整可视化对比")
    print("="*80)

    try:
        truth_scene = pd.read_csv(input_path)
        all_mmsi = truth_scene["MMSI"].unique().tolist()
        print(f"\n✅ 加载场景 {SCENE_ID}")
        print(f"   总MMSI数：{len(all_mmsi)}")

        # 过滤过短航迹
        valid_tracks = []
        for mmsi, group in truth_scene.groupby("MMSI"):
            if len(group) >= MIN_TRACK_POINTS:
                valid_tracks.append((mmsi, group.sort_values("time").reset_index(drop=True)))
        
        print(f"   有效航迹数：{len(valid_tracks)}")

        # 打乱顺序
        random.shuffle(valid_tracks)
        
        # 计算正/负样本数量
        num_total = len(valid_tracks)
        num_positive = int(num_total * (1 - NEGATIVE_RATIO))
        num_negative = num_total - num_positive
        
        print(f"\n   正样本对数：{num_positive}")
        print(f"   负样本对数：{num_negative}")
        print(f"   其中难分负样本：{int(num_negative * HARD_NEGATIVE_RATIO)}")

        # 生成数据
        simulated_scene = []
        correlation_table = []
        batch_id = 0
        vis_count = 0

        # ===================== 1. 生成正样本对 =====================
        print(f"\n[1/2] 生成正样本对...")
        positive_tracks = valid_tracks[:num_positive]
        
        # 按比例分配效果
        random.shuffle(positive_tracks)
        n_gap = int(len(positive_tracks)*GAP_RATIO)
        n_length_diff = int(len(positive_tracks)*LENGTH_DIFF_RATIO)
        n_cross = int(len(positive_tracks)*CROSS_RATIO)
        
        gap_tracks = positive_tracks[:n_gap]
        length_diff_tracks = positive_tracks[n_gap:n_gap+n_length_diff]
        cross_tracks = positive_tracks[n_gap+n_length_diff:n_gap+n_length_diff+n_cross]
        normal_tracks = positive_tracks[n_gap+n_length_diff+n_cross:]

        # 正常正样本
        for mmsi, truth_track in normal_tracks:
            radar1 = simulate_track_from_real_final(truth_track, 9001, "normal")
            radar2 = simulate_track_from_real_final(truth_track, 9002, "normal")
            radar1["batch"] = batch_id
            radar2["batch"] = batch_id + 1
            batch_id += 2
            simulated_scene.append(radar1)
            simulated_scene.append(radar2)
            correlation_table.append({"MMSI": mmsi, "source": 9001, "batch": radar1["batch"].iloc[0], "label": 1})
            correlation_table.append({"MMSI": mmsi, "source": 9002, "batch": radar2["batch"].iloc[0], "label": 1})
            
            if vis_count < 3:
                vis_path = os.path.join(vis_dir, f"vis_{SCENE_ID}_pos_normal_{vis_count}.png")
                visualize_pair(truth_track, truth_track, radar1, radar2, vis_path, is_positive=True)
                vis_count += 1

        # 断点正样本
        for mmsi, truth_track in gap_tracks:
            radar1 = simulate_track_from_real_final(truth_track, 9001, "gap")
            radar2 = simulate_track_from_real_final(truth_track, 9002, "gap")
            radar1["batch"] = batch_id
            radar2["batch"] = batch_id + 1
            batch_id += 2
            simulated_scene.append(radar1)
            simulated_scene.append(radar2)
            correlation_table.append({"MMSI": mmsi, "source": 9001, "batch": radar1["batch"].iloc[0], "label": 1})
            correlation_table.append({"MMSI": mmsi, "source": 9002, "batch": radar2["batch"].iloc[0], "label": 1})
            
            if vis_count < 6:
                vis_path = os.path.join(vis_dir, f"vis_{SCENE_ID}_pos_gap_{vis_count}.png")
                visualize_pair(truth_track, truth_track, radar1, radar2, vis_path, is_positive=True)
                vis_count += 1

        # 长度差异正样本
        for mmsi, truth_track in length_diff_tracks:
            radar1 = simulate_track_from_real_final(truth_track, 9001, "length_diff")
            radar2 = simulate_track_from_real_final(truth_track, 9002, "length_diff")
            radar1["batch"] = batch_id
            radar2["batch"] = batch_id + 1
            batch_id += 2
            simulated_scene.append(radar1)
            simulated_scene.append(radar2)
            correlation_table.append({"MMSI": mmsi, "source": 9001, "batch": radar1["batch"].iloc[0], "label": 1})
            correlation_table.append({"MMSI": mmsi, "source": 9002, "batch": radar2["batch"].iloc[0], "label": 1})
            
            if vis_count < 9:
                vis_path = os.path.join(vis_dir, f"vis_{SCENE_ID}_pos_length_{vis_count}.png")
                visualize_pair(truth_track, truth_track, radar1, radar2, vis_path, is_positive=True)
                vis_count += 1

        # X形交叉正样本
        for mmsi, truth_track in cross_tracks:
            radar1 = simulate_track_from_real_final(truth_track, 9001, "normal")
            radar2 = simulate_track_from_real_final(truth_track, 9002, "normal")
            num_crosses = np.random.randint(1, 3)
            radar1, radar2 = create_x_cross_tracks(radar1, radar2, num_crosses)
            radar1["batch"] = batch_id
            radar2["batch"] = batch_id + 1
            batch_id += 2
            simulated_scene.append(radar1)
            simulated_scene.append(radar2)
            correlation_table.append({"MMSI": mmsi, "source": 9001, "batch": radar1["batch"].iloc[0], "label": 1})
            correlation_table.append({"MMSI": mmsi, "source": 9002, "batch": radar2["batch"].iloc[0], "label": 1})
            
            if vis_count < 12:
                vis_path = os.path.join(vis_dir, f"vis_{SCENE_ID}_pos_cross_{vis_count}.png")
                visualize_pair(truth_track, truth_track, radar1, radar2, vis_path, 
                              is_positive=True, is_crossed=True, num_crosses=num_crosses)
                vis_count += 1

        # ===================== 2. 生成负样本对 =====================
        print(f"\n[2/2] 生成负样本对...")
        negative_tracks = valid_tracks[num_positive:]
        
        for i in range(0, len(negative_tracks) - 1, 2):
            if i + 1 >= len(negative_tracks):
                break
                
            mmsi1, truth_track1 = negative_tracks[i]
            mmsi2, truth_track2 = negative_tracks[i + 1]
            
            # 决定是难分负样本还是简单负样本
            is_hard = np.random.rand() < HARD_NEGATIVE_RATIO
            
            if is_hard:
                # 难分负样本：把两条航迹都仿真后，平移到一起
                radar1 = simulate_track_from_real_final(truth_track1, 9001, "normal")
                radar2 = simulate_track_from_real_final(truth_track2, 9002, "normal")
                # 把radar2平移到radar1附近
                radar2 = create_hard_negative_track(radar2, radar1)
            else:
                # 简单负样本：各自仿真，不平移
                radar1 = simulate_track_from_real_final(truth_track1, 9001, "normal")
                radar2 = simulate_track_from_real_final(truth_track2, 9002, "normal")
            
            radar1["batch"] = batch_id
            radar2["batch"] = batch_id + 1
            batch_id += 2
            simulated_scene.append(radar1)
            simulated_scene.append(radar2)
            correlation_table.append({"MMSI": mmsi1, "source": 9001, "batch": radar1["batch"].iloc[0], "label": 0})
            correlation_table.append({"MMSI": mmsi2, "source": 9002, "batch": radar2["batch"].iloc[0], "label": 0})
            
            # 可视化负样本
            if vis_count < 15:
                vis_path = os.path.join(vis_dir, f"vis_{SCENE_ID}_neg_{'hard' if is_hard else 'easy'}_{vis_count}.png")
                visualize_pair(truth_track1, truth_track2, radar1, radar2, vis_path, 
                              is_positive=False, is_hard_negative=is_hard)
                vis_count += 1

        # 保存结果
        final_scene = pd.concat(simulated_scene, ignore_index=True)
        final_corr = pd.DataFrame(correlation_table)
        
        final_scene.to_csv(output_scene_path, index=False, encoding="utf-8")
        final_corr.to_csv(output_corr_path, index=False, encoding="utf-8")
        
        print("\n" + "="*80)
        print(f"✅ 场景 {SCENE_ID} 完善版仿真完成！")
        print(f"   输出场景：{output_scene_path}")
        print(f"   输出关联表：{output_corr_path}")
        print(f"   总样本对数：{len(final_corr) // 2}")
        print(f"   正样本数：{int(np.sum(final_corr['label'] == 1)) // 2}")
        print(f"   负样本数：{int(np.sum(final_corr['label'] == 0)) // 2}")
        print(f"   可视化保存到：{vis_dir}")
        print("="*80)

    except Exception as e:
        print(f"❌ 处理失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_scene_0_with_negatives()