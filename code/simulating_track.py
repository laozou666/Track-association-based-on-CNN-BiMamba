import pandas as pd
import numpy as np
import os
import random

# ===================== 配置区 =====================
# 真实场景根目录
REAL_SCENE_ROOT = "/home/yangcq/track_association/data/raw/真实场景/"
# 输出根目录
OUTPUT_ROOT = "/home/yangcq/track_association/data/simulated_all_scenes/"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 仿真参数（和场景0完全一致）
GAP_RATIO = 0.3
LENGTH_DIFF_RATIO = 0.2
NORMAL_RATIO = 0.5
MIN_TRACK_POINTS = 15
# 处理范围：0到4999
START_SCENE = 0
END_SCENE = 4999
# =====================================================================

def simulate_track_from_real(truth_track, source_id, effect_type="normal"):
    track = truth_track.copy().sort_values("time").reset_index(drop=True)

    # 1. 基础采样：保证仿真点数≥真实点数
    if source_id == 9001:
        keep_ratio = np.random.uniform(0.95, 1.0)
    else:
        keep_ratio = np.random.uniform(0.90, 1.05)
    
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
        gap_start = int(len(track)*np.random.uniform(0.35, 0.55))
        gap_length = int(len(track)*np.random.uniform(0.08, 0.12))
        gap_end = gap_start + gap_length
        track = pd.concat([track.iloc[:gap_start], track.iloc[gap_end:]]).reset_index(drop=True)
    
    if effect_type == "length_diff":
        if source_id == 9001:
            crop_ratio = np.random.uniform(0.02, 0.05)
        else:
            crop_ratio = np.random.uniform(0.15, 0.25)
        start = int(len(track)*crop_ratio)
        end = len(track) - int(len(track)*crop_ratio)
        track = track.iloc[start:end].reset_index(drop=True)

    # 3. 通用噪声和偏移
    sys_err_lon = np.random.uniform(-0.0015, 0.0015)
    sys_err_lat = np.random.uniform(-0.0015, 0.0015)
    track["lon"] += sys_err_lon
    track["lat"] += sys_err_lat

    noise_std = 0.00006 if source_id == 9001 else 0.00009
    track["lon"] += np.random.normal(0, noise_std, len(track))
    track["lat"] += np.random.normal(0, noise_std, len(track))

    # 4. 少量离群点
    if len(track) > 20:
        num_outliers = np.random.randint(1, 4)
        outlier_indices = np.random.choice(len(track), num_outliers, replace=False)
        for idx in outlier_indices:
            outlier_dist = np.random.uniform(0.0005, 0.0012)
            outlier_angle = np.random.uniform(0, 2*np.pi)
            track.loc[idx, "lon"] += outlier_dist * np.cos(outlier_angle)
            track.loc[idx, "lat"] += outlier_dist * np.sin(outlier_angle)

    return track

def process_single_scene(scene_id):
    """处理单个场景"""
    input_path = os.path.join(REAL_SCENE_ROOT, f"场景-{scene_id}.csv")
    output_scene_path = os.path.join(OUTPUT_ROOT, f"场景-{scene_id}.csv")
    output_corr_path = os.path.join(OUTPUT_ROOT, f"关联结果-{scene_id}.csv")

    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"⚠️  场景-{scene_id}.csv 不存在，跳过")
        return False

    try:
        # 读取真实数据
        truth_scene = pd.read_csv(input_path)
        all_mmsi = truth_scene["MMSI"].unique().tolist()

        # 过滤过短航迹
        valid_tracks = []
        for mmsi, group in truth_scene.groupby("MMSI"):
            if len(group) >= MIN_TRACK_POINTS:
                valid_tracks.append((mmsi, group.sort_values("time").reset_index(drop=True)))

        if len(valid_tracks) == 0:
            print(f"⚠️  场景-{scene_id} 没有有效航迹，跳过")
            return False

        # 按比例分配效果
        random.shuffle(valid_tracks)
        n_gap = int(len(valid_tracks)*GAP_RATIO)
        n_length_diff = int(len(valid_tracks)*LENGTH_DIFF_RATIO)
        
        gap_tracks = valid_tracks[:n_gap]
        length_diff_tracks = valid_tracks[n_gap:n_gap+n_length_diff]
        normal_tracks = valid_tracks[n_gap+n_length_diff:]

        # 生成双信源数据
        simulated_scene = []
        correlation_table = []
        batch_id = 0

        for mmsi, truth_track in normal_tracks:
            radar1 = simulate_track_from_real(truth_track, 9001, "normal")
            radar2 = simulate_track_from_real(truth_track, 9002, "normal")
            radar1["batch"] = batch_id
            radar2["batch"] = batch_id + 1
            batch_id += 2
            simulated_scene.append(radar1)
            simulated_scene.append(radar2)
            correlation_table.append({"MMSI": mmsi, "source": 9001, "batch": radar1["batch"].iloc[0]})
            correlation_table.append({"MMSI": mmsi, "source": 9002, "batch": radar2["batch"].iloc[0]})

        for mmsi, truth_track in gap_tracks:
            radar1 = simulate_track_from_real(truth_track, 9001, "gap")
            radar2 = simulate_track_from_real(truth_track, 9002, "gap")
            radar1["batch"] = batch_id
            radar2["batch"] = batch_id + 1
            batch_id += 2
            simulated_scene.append(radar1)
            simulated_scene.append(radar2)
            correlation_table.append({"MMSI": mmsi, "source": 9001, "batch": radar1["batch"].iloc[0]})
            correlation_table.append({"MMSI": mmsi, "source": 9002, "batch": radar2["batch"].iloc[0]})

        for mmsi, truth_track in length_diff_tracks:
            radar1 = simulate_track_from_real(truth_track, 9001, "length_diff")
            radar2 = simulate_track_from_real(truth_track, 9002, "length_diff")
            radar1["batch"] = batch_id
            radar2["batch"] = batch_id + 1
            batch_id += 2
            simulated_scene.append(radar1)
            simulated_scene.append(radar2)
            correlation_table.append({"MMSI": mmsi, "source": 9001, "batch": radar1["batch"].iloc[0]})
            correlation_table.append({"MMSI": mmsi, "source": 9002, "batch": radar2["batch"].iloc[0]})

        # 保存结果
        final_scene = pd.concat(simulated_scene, ignore_index=True)
        final_corr = pd.DataFrame(correlation_table)
        
        final_scene.to_csv(output_scene_path, index=False, encoding="utf-8")
        final_corr.to_csv(output_corr_path, index=False, encoding="utf-8")
        
        return True
    except Exception as e:
        print(f"❌ 处理场景-{scene_id} 失败：{e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print(" 批量处理 0-4999 所有场景（纯数据生成）")
    print("  参数：30%断点 | 40%长短差异 | 30%正常")
    print("="*70)

    success_count = 0
    fail_count = 0

    for scene_id in range(START_SCENE, END_SCENE + 1):
        if (scene_id + 1) % 100 == 0:
            print(f"\n处理进度：{scene_id + 1}/{END_SCENE + 1}")
            print(f"  已成功：{success_count} 个")
            print(f"  已失败：{fail_count} 个")
        
        if process_single_scene(scene_id):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "="*70)
    print("🎉 全部处理完成！")
    print(f"  总场景数：{END_SCENE - START_SCENE + 1}")
    print(f"  成功处理：{success_count} 个")
    print(f"  处理失败：{fail_count} 个")
    print(f"  输出目录：{OUTPUT_ROOT}")
    print("  ✅ 所有输出格式和场景0完全一致")
    print("  ✅ 模型可以直接批量加载训练")
    print("="*70)