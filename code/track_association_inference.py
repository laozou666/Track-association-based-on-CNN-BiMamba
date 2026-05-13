import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import re
import glob
import matplotlib.pyplot as plt
import argparse
from matplotlib.patches import Patch

# ===================== 配置区 =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "output", "model")
SCALER_DIR = os.path.join(PROJECT_ROOT, "data", "final_dataset")
TEST_CSV_ROOT = os.path.join(PROJECT_ROOT, "data", "final_dataset", "test_set_csvs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "result")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 超参数（必须和训练时一致：当前数据集固定长度 350）
MAX_SEQ_LEN = 350
INPUT_DIM = 4  # lat, lon, vel, cou
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVAILABLE_MODELS = ['cnn_mamba', 'cnn_mamba_tuned', 'cnn_lstm', 'cnn_bilstm',
                    'bilstm', 'ann', 'cnn', 'lstm', 'bigru']

# 调优后的 CNN-Mamba：来自 2D 网格搜索的最优配置（92.38% val acc）
TUNED_CNN_MAMBA_CFG = {
    'n_cnn':    4,
    'n_mamba':  1,
    'dropout':  0.4,
    'ckpt':     os.path.join(PROJECT_ROOT, "output", "grid_search", "g_cnn4_drop04_best.pth"),
}
AVAILABLE_LENGTHS = ['short', 'medium', 'long']
AVAILABLE_LABELS = [0, 1]
DEFAULT_MODEL = "cnn_bilstm"

print(f"Using device: {DEVICE}")

# ===================== 命令行参数解析 =====================
parser = argparse.ArgumentParser(
    description='End-to-end Track Association Inference (interactive or CLI mode)'
)
parser.add_argument('--label', type=int, default=None, choices=AVAILABLE_LABELS,
                    help='Ground-truth label of the test sample (0=not associated, 1=associated)')
parser.add_argument('--length', type=str, default=None, choices=AVAILABLE_LENGTHS,
                    help='Length category of the test sample (short / medium / long)')
parser.add_argument('--sample', type=str, default=None,
                    help='Sample (batch) number, e.g. 10059')
parser.add_argument('--model', type=str, default=None, choices=AVAILABLE_MODELS,
                    help='Model to use for inference')
# 兼容旧版：也可以直接传两个 CSV 路径
parser.add_argument('--track1', type=str, default=None, help='(Optional) Path to track1 CSV, skips interactive mode')
parser.add_argument('--track2', type=str, default=None, help='(Optional) Path to track2 CSV, skips interactive mode')
args = parser.parse_args()


# ===================== 交互式输入 =====================
def _prompt_choice(prompt_text, options, default=None, cast=str):
    """通用的选择型 prompt，带校验和默认值。"""
    opts_str = " / ".join(str(o) for o in options)
    default_hint = f" [默认: {default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt_text} ({opts_str}){default_hint}: ").strip()
        if raw == "" and default is not None:
            return cast(default)
        try:
            val = cast(raw)
        except ValueError:
            print(f"  ⚠️  无法解析输入 '{raw}'，请重新输入。")
            continue
        if val in options:
            return val
        print(f"  ⚠️  '{val}' 不在可选列表 {options} 中，请重新输入。")


def _list_sample_ids(label, length):
    """扫描 test_set_csvs/label_X/length/ 下的 sample_{id}_track1.csv，返回排序后的 id 列表。"""
    folder = os.path.join(TEST_CSV_ROOT, f"label_{label}", length)
    if not os.path.isdir(folder):
        return []
    pattern = re.compile(r'^sample_(\d+)_track1\.csv$')
    ids = []
    for name in os.listdir(folder):
        m = pattern.match(name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _prompt_sample(label, length):
    """让用户选择样本编号，显示可用数量和示例。"""
    ids = _list_sample_ids(label, length)
    if not ids:
        print(f"❌ 在 label_{label}/{length} 下没有找到任何样本！")
        return None
    print(f"  → 该类别下共 {len(ids)} 个样本，编号范围 [{ids[0]}, {ids[-1]}]")
    preview = ids[:5] + (['...'] + ids[-3:] if len(ids) > 8 else [])
    print(f"  → 示例编号: {preview}")
    while True:
        raw = input(f"请输入样本编号 (batch number) [默认: {ids[0]}]: ").strip()
        if raw == "":
            return ids[0]
        try:
            sid = int(raw)
        except ValueError:
            print(f"  ⚠️  '{raw}' 不是整数，请重新输入。")
            continue
        if sid in ids:
            return sid
        print(f"  ⚠️  编号 {sid} 不存在于该类别，请从可用列表中选择。")


def interactive_select(args):
    """根据 args 的缺失情况，交互补齐 label / length / sample / model。"""
    need_input = (args.label is None or args.length is None
                  or args.sample is None or args.model is None)
    if need_input:
        print("\n" + "-" * 80)
        print(" 请在下方依次选择：测试集标签 / 长度 / 样本编号 / 推理模型")
        print("-" * 80)

    label = args.label
    if label is None:
        label = _prompt_choice("请选择标签", AVAILABLE_LABELS, default=1, cast=int)

    length = args.length
    if length is None:
        length = _prompt_choice("请选择序列长度", AVAILABLE_LENGTHS, default='medium', cast=str)

    sample_id = None
    if args.sample is not None:
        try:
            sample_id = int(args.sample)
        except ValueError:
            print(f"⚠️  --sample '{args.sample}' 不是整数，进入交互选择。")
    if sample_id is None:
        sample_id = _prompt_sample(label, length)
        if sample_id is None:
            return None

    model_name = args.model
    if model_name is None:
        model_name = _prompt_choice("请选择推理模型", AVAILABLE_MODELS,
                                    default=DEFAULT_MODEL, cast=str)

    t1_path = os.path.join(TEST_CSV_ROOT, f"label_{label}", length,
                           f"sample_{sample_id}_track1.csv")
    t2_path = os.path.join(TEST_CSV_ROOT, f"label_{label}", length,
                           f"sample_{sample_id}_track2.csv")

    if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
        print(f"❌ 对应文件不存在：\n   {t1_path}\n   {t2_path}")
        return None

    return {
        'label': label,
        'length': length,
        'sample': sample_id,
        'model': model_name,
        'track1': t1_path,
        'track2': t2_path,
    }

# ===================== 1. 数据加载与预处理 =====================
def load_and_preprocess_track(csv_path, scaler_mean, scaler_scale):
    """
    Load AIS CSV and preprocess it into model input format
    Returns: (original_lon, original_lat, processed_track)
    """
    print(f"\nLoading track: {os.path.basename(csv_path)}")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} points")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return None, None, None
    
    # Extract features (match training features: lat, lon, vel, cou)
    try:
        # Map column names (adapt to your CSV format)
        col_mapping = {
            'lat': ['纬度(°)', 'lat', 'Latitude', 'latitude'],
            'lon': ['经度(°)', 'lon', 'Longitude', 'longitude'],
            'vel': ['速度(kn)', 'vel', 'Speed', 'speed', '速度'],
            'cou': ['船艏向(°)', 'cou', 'Heading', 'heading', '船艏向']
        }
        
        # Find matching columns
        feature_cols = {}
        for feat, possible_names in col_mapping.items():
            for name in possible_names:
                if name in df.columns:
                    feature_cols[feat] = name
                    break
        
        if len(feature_cols) < 4:
            print(f"❌ Missing columns. Found: {list(df.columns)}")
            print(f"   Need columns for: lat, lon, vel, cou")
            return None, None, None
        
        # Extract data
        lat = df[feature_cols['lat']].values.astype(np.float32)
        lon = df[feature_cols['lon']].values.astype(np.float32)
        vel = df[feature_cols['vel']].values.astype(np.float32)
        cou = df[feature_cols['cou']].values.astype(np.float32)
        
        # Stack features
        track = np.stack([lat, lon, vel, cou], axis=1)  # Shape: (N, 4)
        print(f"✅ Extracted features: lat, lon, vel, cou")
        
    except Exception as e:
        print(f"❌ Failed to extract features: {e}")
        return None, None, None
    
    # Save original coordinates for plotting
    original_lon = lon.copy()
    original_lat = lat.copy()
    
    # 1. Normalize (using training scaler)
    track = (track - scaler_mean) / scaler_scale
    
    # 2. Pad or truncate to MAX_SEQ_LEN
    if len(track) > MAX_SEQ_LEN:
        track = track[:MAX_SEQ_LEN]
        print(f"⚠️  Track truncated to {MAX_SEQ_LEN} points")
    elif len(track) < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - len(track)
        track = np.pad(track, ((0, pad_len), (0, 0)), mode='constant')
        print(f"⚠️  Track padded with {pad_len} zeros")
    
    return original_lon, original_lat, track.astype(np.float32)

# ===================== 2. 模型加载 =====================
def load_model_for_inference(model_name):
    """Load trained model for inference"""
    print(f"\nLoading model: {model_name}")
    
    try:
        # --- 特例：调优后的 CNN-Mamba（grid search 最优，92.38% val acc）---
        if model_name == "cnn_mamba_tuned":
            # 确保能找到 code/ablation_cnn_mamba.py
            code_dir = os.path.dirname(os.path.abspath(__file__))
            if code_dir not in sys.path:
                sys.path.insert(0, code_dir)
            from ablation_cnn_mamba import AblationCNNMamba
            cfg = TUNED_CNN_MAMBA_CFG
            model = AblationCNNMamba(
                input_dim=INPUT_DIM,
                n_cnn=cfg['n_cnn'],
                n_mamba=cfg['n_mamba'],
                dropout=cfg['dropout'],
            ).to(DEVICE)
            model.eval()
            ckpt = cfg['ckpt']
            if not os.path.exists(ckpt):
                print(f"❌ Tuned CNN-Mamba weights not found: {ckpt}")
                return None
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"✅ Tuned CNN-Mamba loaded "
                  f"(n_cnn={cfg['n_cnn']}, n_mamba={cfg['n_mamba']}, "
                  f"dropout={cfg['dropout']}): {ckpt}")
            return model

        # --- 其他 baseline 模型（使用 utils/ 下的实现）---
        if model_name == "cnn_mamba":
            from utils.model_cnn_mamba import CNNTrajectoryMatcher as Model
        elif model_name == "cnn_lstm":
            from utils.model_cnn_lstm import CNNTrajectoryMatcher_LSTM as Model
        elif model_name == "cnn_bilstm":
            from utils.model_cnn_bilstm import CNNTrajectoryMatcher_BiLSTM as Model
        elif model_name == "bilstm":
            from utils.model_bilstm import TrajectoryMatcher_BiLSTM as Model
        elif model_name == "ann":
            from utils.model_ann import TrajectoryMatcher_ANN as Model
        elif model_name == "cnn":
            from utils.model_cnn import TrajectoryMatcher_CNN as Model
        elif model_name == "lstm":
            from utils.model_lstm import TrajectoryMatcher_LSTM as Model
        elif model_name == "bigru":
            from utils.model_bigru import TrajectoryMatcher_BiGRU as Model
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = Model().to(DEVICE)
        model.eval()

        model_path = os.path.join(MODEL_DIR, f"best_{model_name}.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"✅ Model weights loaded: {model_path}")
        else:
            print(f"❌ Model weights not found: {model_path}")
            return None

        return model
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

# ===================== 3. 推理函数 =====================
def run_inference(model, track1, track2, device):
    """Run inference on a single track pair"""
    model.eval()
    
    # Convert to tensors and add batch dimension
    t1 = torch.tensor(track1).unsqueeze(0).to(device)
    t2 = torch.tensor(track2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(t1, t2)
        prob = torch.sigmoid(output).item()
    
    is_associated = prob > THRESHOLD
    return prob, is_associated

# ===================== 4. 结果可视化 =====================
# def plot_result(lon1, lat1, lon2, lat2, prob, is_associated, model_name, save_path):
#     """
#     Plot the two tracks as scatter points only (no lines)
#     All text in English
#     """
#     fig = plt.figure(figsize=(12, 14))
#     gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
#     # 1. Plot tracks (top) - ONLY SCATTER POINTS, NO LINES
#     ax_track = fig.add_subplot(gs[0])
    
#     # Plot Track 1 (Blue scatter only)
#     ax_track.scatter(lon1, lat1, color='#1f77b4', s=20, alpha=0.8, label='Track 1 (Source A)')
#     ax_track.scatter(lon1[0], lat1[0], color='green', s=250, marker='o', edgecolors='black', label='Start')
#     ax_track.scatter(lon1[-1], lat1[-1], color='red', s=250, marker='x', linewidths=3, label='End')
    
#     # Plot Track 2 (Orange scatter only)
#     ax_track.scatter(lon2, lat2, color='#ff7f0e', s=20, alpha=0.8, label='Track 2 (Source B)')
#     ax_track.scatter(lon2[0], lat2[0], color='green', s=250, marker='o', edgecolors='black')
#     ax_track.scatter(lon2[-1], lat2[-1], color='red', s=250, marker='x', linewidths=3)
    
#     # Track plot properties
#     ax_track.set_title('Track Pair Visualization', fontsize=18, pad=20)
#     ax_track.set_xlabel('Longitude (°)', fontsize=14)
#     ax_track.set_ylabel('Latitude (°)', fontsize=14)
#     ax_track.legend(fontsize=12, loc='upper left')
#     ax_track.axis('equal')
#     ax_track.grid(True, alpha=0.3, linestyle='--')
#     ax_track.tick_params(axis='both', which='major', labelsize=12)
    
#     # 2. Plot result panel (bottom)
#     ax_result = fig.add_subplot(gs[1])
#     ax_result.axis('tight')
#     ax_result.axis('off')
    
#     # Determine result text and color
#     if is_associated:
#         result_text = "ASSOCIATED (Same Target)"
#         result_color = '#2ca02c'  # Green
#     else:
#         result_text = "NOT ASSOCIATED (Different Targets)"
#         result_color = '#d62728'  # Red
    
#     # Create result display
#     result_content = [
#         ["Model Used:", model_name],
#         ["Association Probability:", f"{prob:.4f}"],
#         ["Threshold:", f"{THRESHOLD:.2f}"],
#         ["Final Decision:", result_text]
#     ]
    
#     # Create table
#     table = ax_result.table(
#         cellText=result_content,
#         colLabels=["Item", "Value"],
#         loc='center',
#         cellLoc='left',
#         colWidths=[0.3, 0.5],
#         fontsize=14
#     )
#     table.auto_set_font_size(False)
#     table.set_fontsize(14)
#     table.scale(1, 2.5)
    
#     # Highlight the final decision row
#     for (i, j), cell in table.get_celld().items():
#         if i == 4:  # Final decision row
#             cell.set_facecolor(result_color)
#             cell.set_text_props(weight='bold', color='white')
#         elif i == 0:  # Header
#             cell.set_facecolor('#f0f0f0')
#             cell.set_text_props(weight='bold')
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"\n✅ Result visualization saved to: {save_path}")
#     plt.show()
#     plt.close()
def plot_result(lon1, lat1, lon2, lat2, prob, is_associated, model_name, save_path):
    """
    修复版：取消等比例 + 自适应坐标 + 放大航迹
    彻底解决直线问题
    """
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    ax_track = fig.add_subplot(gs[0])

    # 画航迹（纯散点）
    ax_track.scatter(lon1, lat1, color='#1f77b4', s=30, alpha=0.9, label='Track 1 (Source A)')
    ax_track.scatter(lon2, lat2, color='#ff7f0e', s=30, alpha=0.9, label='Track 2 (Source B)')
    
    # 起点终点
    ax_track.scatter(lon1[0], lat1[0], c='green', s=250, edgecolor='k', label='Start')
    ax_track.scatter(lon1[-1], lat1[-1], c='red', s=250, marker='x', linewidth=3, label='End')
    ax_track.scatter(lon2[0], lat2[0], c='green', s=250, edgecolor='k')
    ax_track.scatter(lon2[-1], lat2[-1], c='red', s=250, marker='x', linewidth=3)

    # ===================== 核心修复：自适应坐标，取消等比例 =====================
    all_lon = np.concatenate([lon1, lon2])
    all_lat = np.concatenate([lat1, lat2])
    # 自动放大边界，留出边距
    margin_lon = (all_lon.max() - all_lon.min()) * 0.1
    margin_lat = (all_lat.max() - all_lat.min()) * 0.1
    ax_track.set_xlim(all_lon.min()-margin_lon, all_lon.max()+margin_lon)
    ax_track.set_ylim(all_lat.min()-margin_lat, all_lat.max()+margin_lat)
    # ❌ 永久删除 axis('equal')

    # 样式
    ax_track.set_title('Track Pair Visualization', fontsize=18, pad=20)
    ax_track.set_xlabel('Longitude (°)', fontsize=14)
    ax_track.set_ylabel('Latitude (°)', fontsize=14)
    ax_track.legend(fontsize=12)
    ax_track.grid(True, alpha=0.3, linestyle='--')
    ax_track.tick_params(labelsize=12)

    # 结果面板（不变）
    ax_result = fig.add_subplot(gs[1])
    ax_result.axis('tight')
    ax_result.axis('off')

    result_text = "ASSOCIATED" if is_associated else "NOT ASSOCIATED"
    result_color = '#2ca02c' if is_associated else '#d62728'

    result_content = [
        ["Model Used:", model_name],
        ["Probability:", f"{prob:.4f}"],
        ["Threshold:", f"{THRESHOLD:.2f}"],
        ["Decision:", result_text]
    ]

    table = ax_result.table(
        cellText=result_content,
        colLabels=["Item", "Value"],
        loc='center',
        cellLoc='left',
        colWidths=[0.3, 0.5],
        fontsize=14
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)

    for (i, j), cell in table.get_celld().items():
        if i == 4:
            cell.set_facecolor(result_color)
            cell.set_text_props(weight='bold', color='white')
        elif i == 0:
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 80)
    print(" End-to-End Track Association Inference")
    print("=" * 80)

    # -------- 决定样本与模型 --------
    if args.track1 and args.track2:
        # 兼容：直接传两个 CSV
        selection = {
            'label': None,
            'length': None,
            'sample': None,
            'model': args.model if args.model is not None else DEFAULT_MODEL,
            'track1': args.track1,
            'track2': args.track2,
        }
    else:
        selection = interactive_select(args)
        if selection is None:
            exit(1)

    print("\n[Selection]")
    for k, v in selection.items():
        print(f"  {k:8s}: {v}")

    # 1. Load scaler parameters
    print("\n[1/5] Loading scaler parameters...")
    try:
        scaler_mean = np.load(os.path.join(SCALER_DIR, "scaler_mean.npy"))
        scaler_scale = np.load(os.path.join(SCALER_DIR, "scaler_scale.npy"))
        print("✅ Scaler parameters loaded")
    except Exception as e:
        print(f"❌ Failed to load scaler: {e}")
        print("   Please ensure scaler_mean.npy and scaler_scale.npy exist")
        exit()

    # 2. Load and preprocess tracks
    print("\n[2/5] Loading and preprocessing tracks...")
    lon1, lat1, track1 = load_and_preprocess_track(selection['track1'], scaler_mean, scaler_scale)
    lon2, lat2, track2 = load_and_preprocess_track(selection['track2'], scaler_mean, scaler_scale)

    if track1 is None or track2 is None:
        exit()

    # 3. Load model
    print("\n[3/5] Loading trained model...")
    model = load_model_for_inference(selection['model'])
    if model is None:
        exit()

    # 4. Run inference
    print("\n[4/5] Running inference...")
    prob, is_associated = run_inference(model, track1, track2, DEVICE)

    print(f"\n{'=' * 80}")
    print(f" INFERENCE RESULT")
    print(f"{'=' * 80}")
    print(f" Model:       {selection['model']}")
    if selection['label'] is not None:
        gt_text = "ASSOCIATED" if selection['label'] == 1 else "NOT ASSOCIATED"
        print(f" Ground truth: {gt_text}  (label={selection['label']})")
        print(f" Split:        label_{selection['label']} / {selection['length']} / sample_{selection['sample']}")
    print(f" Probability: {prob:.4f}")
    print(f" Threshold:   {THRESHOLD:.2f}")
    print("")
    if is_associated:
        print(f" ✅ DECISION: ASSOCIATED (Same Target)")
    else:
        print(f" ❌ DECISION: NOT ASSOCIATED (Different Targets)")
    # 与 ground truth 对比
    if selection['label'] is not None:
        correct = int(is_associated) == int(selection['label'])
        print(f" {'🎯 Correct' if correct else '❗ Wrong'} prediction vs. ground truth.")
    print(f"{'=' * 80}")

    # 5. Plot and save result
    print("\n[5/5] Generating visualization...")
    if selection['label'] is not None:
        output_filename = (f"result_{selection['model']}_label{selection['label']}"
                           f"_{selection['length']}_s{selection['sample']}.png")
    else:
        output_filename = (f"result_{os.path.basename(selection['track1']).split('.')[0]}"
                           f"_{os.path.basename(selection['track2']).split('.')[0]}.png")
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    plot_result(lon1, lat1, lon2, lat2, prob, is_associated, selection['model'], output_path)
    print(f"✅ Saved visualization to: {output_path}")

    print("\n🎉 Inference complete!")