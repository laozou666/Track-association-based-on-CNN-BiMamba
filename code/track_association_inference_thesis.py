# -*- coding: utf-8 -*-
"""
航迹关联推理脚本（中文版）

在 track_association_inference.py 基础上，将终端提示与插图标注改为中文；
插图版式与原版一致，无图题；同时输出 PNG（300 DPI）与 SVG。
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch

# ===================== 配置区 =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "output", "model")
SCALER_DIR = os.path.join(PROJECT_ROOT, "data", "final_dataset")
TEST_CSV_ROOT = os.path.join(SCALER_DIR, "test_set_csvs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "thesis_inference")
FONT_PATH = os.path.join(PROJECT_ROOT, "SIMHEI.ttf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SEQ_LEN = 350
INPUT_DIM = 4
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVAILABLE_MODELS = [
    "cnn_mamba_tuned", "cnn_mamba", "cnn_lstm", "cnn_bilstm",
    "bilstm", "ann", "cnn", "lstm", "bigru",
]
TUNED_CNN_MAMBA_CFG = {
    "n_cnn": 4,
    "n_mamba": 1,
    "dropout": 0.4,
    "ckpt": os.path.join(PROJECT_ROOT, "output", "grid_search", "g_cnn4_drop04_best.pth"),
}
AVAILABLE_LENGTHS = ["short", "medium", "long"]
AVAILABLE_LABELS = [0, 1]
DEFAULT_MODEL = "cnn_mamba_tuned"

MODEL_CN = {
    "cnn_mamba_tuned": "cnn_mamba_tuned",
    "cnn_mamba": "cnn_mamba",
    "cnn_lstm": "cnn_lstm",
    "cnn_bilstm": "cnn_bilstm",
    "bilstm": "bilstm",
    "ann": "ann",
    "cnn": "cnn",
    "lstm": "lstm",
    "bigru": "bigru",
}

print(f"使用设备: {DEVICE}")


def setup_chinese_font():
    if os.path.isfile(FONT_PATH):
        font_manager.fontManager.addfont(FONT_PATH)
        prop = font_manager.FontProperties(fname=FONT_PATH)
        plt.rcParams["font.family"] = prop.get_name()
        plt.rcParams["font.sans-serif"] = [prop.get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.dpi"] = 300


# ===================== 命令行参数 =====================
parser = argparse.ArgumentParser(description="航迹关联端到端推理（中文版）")
parser.add_argument("--label", type=int, default=None, choices=AVAILABLE_LABELS,
                    help="真值标签：0=非关联，1=关联")
parser.add_argument("--length", type=str, default=None, choices=AVAILABLE_LENGTHS,
                    help="航迹长度：short / medium / long")
parser.add_argument("--sample", type=str, default=None, help="样本编号，如 10004")
parser.add_argument("--model", type=str, default=None, choices=AVAILABLE_MODELS,
                    help="推理模型")
parser.add_argument("--track1", type=str, default=None, help="航迹1 CSV（跳过交互）")
parser.add_argument("--track2", type=str, default=None, help="航迹2 CSV")
parser.add_argument("--show", action="store_true", help="保存后显示窗口")
args = parser.parse_args()


def _prompt_choice(prompt_text, options, default=None, cast=str):
    opts_str = " / ".join(str(o) for o in options)
    default_hint = f" [默认: {default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt_text} ({opts_str}){default_hint}: ").strip()
        if raw == "" and default is not None:
            return cast(default)
        try:
            val = cast(raw)
        except ValueError:
            print(f"  无法解析输入 '{raw}'，请重新输入。")
            continue
        if val in options:
            return val
        print(f"  '{val}' 不在可选列表 {options} 中，请重新输入。")


def _list_sample_ids(label, length):
    folder = os.path.join(TEST_CSV_ROOT, f"label_{label}", length)
    if not os.path.isdir(folder):
        return []
    pattern = re.compile(r"^sample_(\d+)_track1\.csv$")
    ids = []
    for name in os.listdir(folder):
        m = pattern.match(name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _prompt_sample(label, length):
    ids = _list_sample_ids(label, length)
    if not ids:
        print(f"在 label_{label}/{length} 下没有找到任何样本！")
        return None
    print(f"  -> 该类别下共 {len(ids)} 个样本，编号范围 [{ids[0]}, {ids[-1]}]")
    preview = ids[:5] + (["..."] + ids[-3:] if len(ids) > 8 else [])
    print(f"  -> 示例编号: {preview}")
    while True:
        raw = input(f"请输入样本编号 [默认: {ids[0]}]: ").strip()
        if raw == "":
            return ids[0]
        try:
            sid = int(raw)
        except ValueError:
            print(f"  '{raw}' 不是整数，请重新输入。")
            continue
        if sid in ids:
            return sid
        print(f"  编号 {sid} 不存在于该类别，请从可用列表中选择。")


def interactive_select(cli_args):
    need_input = (cli_args.label is None or cli_args.length is None
                  or cli_args.sample is None or cli_args.model is None)
    if need_input:
        print("\n" + "-" * 80)
        print(" 请在下方依次选择：测试集标签 / 长度 / 样本编号 / 推理模型")
        print("-" * 80)

    label = cli_args.label
    if label is None:
        label = _prompt_choice("请选择标签", AVAILABLE_LABELS, default=1, cast=int)

    length = cli_args.length
    if length is None:
        length = _prompt_choice("请选择序列长度", AVAILABLE_LENGTHS, default="medium", cast=str)

    sample_id = None
    if cli_args.sample is not None:
        try:
            sample_id = int(cli_args.sample)
        except ValueError:
            print(f" --sample '{cli_args.sample}' 不是整数，进入交互选择。")
    if sample_id is None:
        sample_id = _prompt_sample(label, length)
        if sample_id is None:
            return None

    model_name = cli_args.model
    if model_name is None:
        model_name = _prompt_choice("请选择推理模型", AVAILABLE_MODELS,
                                    default=DEFAULT_MODEL, cast=str)

    t1_path = os.path.join(TEST_CSV_ROOT, f"label_{label}", length,
                           f"sample_{sample_id}_track1.csv")
    t2_path = os.path.join(TEST_CSV_ROOT, f"label_{label}", length,
                           f"sample_{sample_id}_track2.csv")
    if not (os.path.exists(t1_path) and os.path.exists(t2_path)):
        print(f"对应文件不存在：\n   {t1_path}\n   {t2_path}")
        return None

    return {
        "label": label, "length": length, "sample": sample_id,
        "model": model_name, "track1": t1_path, "track2": t2_path,
    }


def load_and_preprocess_track(csv_path, scaler_mean, scaler_scale):
    print(f"\n加载航迹: {os.path.basename(csv_path)}")
    try:
        df = pd.read_csv(csv_path)
        print(f"已加载 {len(df)} 个点")
    except Exception as e:
        print(f"加载 CSV 失败: {e}")
        return None, None, None

    col_mapping = {
        "lat": ["纬度(°)", "lat", "Latitude", "latitude"],
        "lon": ["经度(°)", "lon", "Longitude", "longitude"],
        "vel": ["速度(kn)", "vel", "Speed", "speed", "速度"],
        "cou": ["船艏向(°)", "cou", "Heading", "heading", "船艏向"],
    }
    feature_cols = {}
    for feat, possible_names in col_mapping.items():
        for name in possible_names:
            if name in df.columns:
                feature_cols[feat] = name
                break
    if len(feature_cols) < 4:
        print(f"缺少列。当前列: {list(df.columns)}")
        return None, None, None

    lat = df[feature_cols["lat"]].values.astype(np.float32)
    lon = df[feature_cols["lon"]].values.astype(np.float32)
    vel = df[feature_cols["vel"]].values.astype(np.float32)
    cou = df[feature_cols["cou"]].values.astype(np.float32)
    track = np.stack([lat, lon, vel, cou], axis=1)
    print("已提取特征: lat, lon, vel, cou")

    original_lon, original_lat = lon.copy(), lat.copy()
    track = (track - scaler_mean) / scaler_scale
    if len(track) > MAX_SEQ_LEN:
        track = track[:MAX_SEQ_LEN]
        original_lon = original_lon[:MAX_SEQ_LEN]
        original_lat = original_lat[:MAX_SEQ_LEN]
        print(f"航迹已截断至 {MAX_SEQ_LEN} 点")
    elif len(track) < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - len(track)
        track = np.pad(track, ((0, pad_len), (0, 0)), mode="constant")
        print(f"航迹已零填充 {pad_len} 点")

    return original_lon, original_lat, track.astype(np.float32)


def load_model_for_inference(model_name):
    print(f"\n加载模型: {model_name}")
    try:
        if model_name == "cnn_mamba_tuned":
            if CODE_DIR not in sys.path:
                sys.path.insert(0, CODE_DIR)
            from ablation_cnn_mamba import AblationCNNMamba
            cfg = TUNED_CNN_MAMBA_CFG
            model = AblationCNNMamba(
                input_dim=INPUT_DIM,
                n_cnn=cfg["n_cnn"],
                n_mamba=cfg["n_mamba"],
                dropout=cfg["dropout"],
            ).to(DEVICE)
            ckpt = cfg["ckpt"]
            if not os.path.exists(ckpt):
                print(f"权重未找到: {ckpt}")
                return None
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
            model.eval()
            print(f"已加载调优 CNN-Mamba (n_cnn={cfg['n_cnn']}, "
                  f"n_mamba={cfg['n_mamba']}, dropout={cfg['dropout']}): {ckpt}")
            return model

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
            raise ValueError(f"未知模型: {model_name}")

        model = Model().to(DEVICE)
        model_path = os.path.join(MODEL_DIR, f"best_{model_name}.pth")
        if not os.path.exists(model_path):
            print(f"权重未找到: {model_path}")
            return None
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        model.eval()
        print(f"已加载模型权重: {model_path}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def run_inference(model, track1, track2, device):
    model.eval()
    t1 = torch.tensor(track1).unsqueeze(0).to(device)
    t2 = torch.tensor(track2).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(t1, t2)
        prob = torch.sigmoid(output).item()
    return prob, prob > THRESHOLD


def plot_result(lon1, lat1, lon2, lat2, prob, is_associated, model_name, save_path, show=False):
    setup_chinese_font()
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_track = fig.add_subplot(gs[0])

    ax_track.scatter(lon1, lat1, color="#1f77b4", s=30, alpha=0.9, label="航迹1（源A）")
    ax_track.scatter(lon2, lat2, color="#ff7f0e", s=30, alpha=0.9, label="航迹2（源B）")
    ax_track.scatter(lon1[0], lat1[0], c="green", s=250, edgecolor="k", label="起点")
    ax_track.scatter(lon1[-1], lat1[-1], c="red", s=250, marker="x", linewidths=3, label="终点")
    ax_track.scatter(lon2[0], lat2[0], c="green", s=250, edgecolor="k")
    ax_track.scatter(lon2[-1], lat2[-1], c="red", s=250, marker="x", linewidths=3)

    all_lon = np.concatenate([lon1, lon2])
    all_lat = np.concatenate([lat1, lat2])
    margin_lon = (all_lon.max() - all_lon.min()) * 0.1
    margin_lat = (all_lat.max() - all_lat.min()) * 0.1
    ax_track.set_xlim(all_lon.min() - margin_lon, all_lon.max() + margin_lon)
    ax_track.set_ylim(all_lat.min() - margin_lat, all_lat.max() + margin_lat)

    ax_track.set_xlabel("经度 (°)", fontsize=14)
    ax_track.set_ylabel("纬度 (°)", fontsize=14)
    ax_track.legend(fontsize=12)
    ax_track.grid(True, alpha=0.3, linestyle="--")
    ax_track.tick_params(labelsize=12)

    ax_result = fig.add_subplot(gs[1])
    ax_result.axis("off")

    if is_associated:
        result_text = "关联（同一目标）"
        result_color = "#2ca02c"
    else:
        result_text = "非关联（不同目标）"
        result_color = "#d62728"

    result_content = [
        ["使用模型:", MODEL_CN.get(model_name, model_name)],
        ["关联概率:", f"{prob:.4f}"],
        ["判定阈值:", f"{THRESHOLD:.2f}"],
        ["推理结论:", result_text],
    ]

    table = ax_result.table(
        cellText=result_content,
        colLabels=["项目", "数值"],
        loc="center",
        cellLoc="left",
        colWidths=[0.3, 0.5],
        fontsize=14,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)

    for (i, j), cell in table.get_celld().items():
        if i == 4:
            cell.set_facecolor(result_color)
            cell.set_text_props(weight="bold", color="white")
        elif i == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    base, _ = os.path.splitext(save_path)
    png_path = base + ".png"
    svg_path = base + ".svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"\n结果图已保存:")
    print(f"  PNG (300 DPI): {png_path}")
    print(f"  SVG:           {svg_path}")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print(" 航迹关联端到端推理")
    print("=" * 80)

    if args.track1 and args.track2:
        selection = {
            "label": args.label,
            "length": args.length,
            "sample": int(args.sample) if args.sample else None,
            "model": args.model if args.model is not None else DEFAULT_MODEL,
            "track1": args.track1,
            "track2": args.track2,
        }
    else:
        selection = interactive_select(args)
        if selection is None:
            exit(1)

    print("\n[选择结果]")
    for k, v in selection.items():
        print(f"  {k:8s}: {v}")

    print("\n[1/5] 加载标准化参数...")
    try:
        scaler_mean = np.load(os.path.join(SCALER_DIR, "scaler_mean.npy"))
        scaler_scale = np.load(os.path.join(SCALER_DIR, "scaler_scale.npy"))
        print("标准化参数已加载")
    except Exception as e:
        print(f"加载标准化参数失败: {e}")
        exit()

    print("\n[2/5] 加载并预处理航迹...")
    lon1, lat1, track1 = load_and_preprocess_track(
        selection["track1"], scaler_mean, scaler_scale)
    lon2, lat2, track2 = load_and_preprocess_track(
        selection["track2"], scaler_mean, scaler_scale)
    if track1 is None or track2 is None:
        exit()

    print("\n[3/5] 加载训练模型...")
    model = load_model_for_inference(selection["model"])
    if model is None:
        exit()

    print("\n[4/5] 执行推理...")
    prob, is_associated = run_inference(model, track1, track2, DEVICE)

    print(f"\n{'=' * 80}")
    print(" 推理结果")
    print(f"{'=' * 80}")
    print(f" 模型:       {selection['model']}")
    if selection["label"] is not None:
        gt_text = "关联" if selection["label"] == 1 else "非关联"
        print(f" 真值标签:   {gt_text}  (label={selection['label']})")
        print(f" 样本路径:   label_{selection['label']} / "
              f"{selection['length']} / sample_{selection['sample']}")
    print(f" 关联概率:   {prob:.4f}")
    print(f" 判定阈值:   {THRESHOLD:.2f}")
    print("")
    if is_associated:
        print(" 判定结果: 关联（同一目标）")
    else:
        print(" 判定结果: 非关联（不同目标）")
    if selection["label"] is not None:
        correct = int(is_associated) == int(selection["label"])
        print(f" 与真值对比: {'预测正确' if correct else '预测错误'}")
    print(f"{'=' * 80}")

    print("\n[5/5] 生成可视化...")
    if selection["label"] is not None:
        output_filename = (
            f"result_{selection['model']}_label{selection['label']}_"
            f"{selection['length']}_s{selection['sample']}.png"
        )
    else:
        output_filename = (
            f"result_{os.path.basename(selection['track1']).split('.')[0]}_"
            f"{os.path.basename(selection['track2']).split('.')[0]}.png"
        )
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    plot_result(lon1, lat1, lon2, lat2, prob, is_associated,
                selection["model"], output_path, show=args.show)
    print("\n推理完成。")
