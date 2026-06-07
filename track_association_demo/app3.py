# -*- coding: utf-8 -*-
"""航迹关联演示系统（科研简洁版 · 结果页内展开）"""

import base64
import io
import os
import socket

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from model_cnn_mamba import AblationCNNMamba

matplotlib.use("Agg")

# ==================== 路径与参数 ====================
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_DATA_DIR = os.path.join(DEMO_DIR, "track_data")
MODEL_PATH = os.path.join(DEMO_DIR, "g_cnn4_drop04_best.pth")
SCALER_MEAN_PATH = os.path.join(DEMO_DIR, "scaler_mean.npy")
SCALER_SCALE_PATH = os.path.join(DEMO_DIR, "scaler_scale.npy")

MAX_SEQ_LEN = 350
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRESET_SAMPLES = {
    "示例1：关联航迹（同一目标）": {
        "track1": os.path.join(TRACK_DATA_DIR, "positive_track1.csv"),
        "track2": os.path.join(TRACK_DATA_DIR, "positive_track2.csv"),
        "label": 1,
    },
    "示例2：非关联航迹（不同目标）": {
        "track1": os.path.join(TRACK_DATA_DIR, "negative_track1.csv"),
        "track2": os.path.join(TRACK_DATA_DIR, "negative_track2.csv"),
        "label": 0,
    },
}

COL_MAPPING = {
    "lat": ["纬度(°)", "lat", "Latitude", "latitude"],
    "lon": ["经度(°)", "lon", "Longitude", "longitude"],
    "vel": ["速度(kn)", "vel", "SOG", "sog", "Speed", "speed", "速度", "对地航速"],
    "cou": ["船艏向(°)", "cou", "COG", "cog", "Heading", "heading", "船艏向", "对地航向"],
}

# ==================== 简洁科研样式 ====================
CUSTOM_CSS = """
body, .gradio-container {
    background: #f8f9fa !important;
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
}
.main-card {
    max-width: 860px;
    margin: 1.8rem auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    padding: 2rem 2.2rem;
    border: 1px solid #e9ecef;
}
.main-title {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #0b2b44 !important;
    text-align: center;
    margin-bottom: 0.2rem !important;
}
.sub-title {
    font-size: 0.95rem !important;
    color: #5f6b7a;
    text-align: center;
    margin-bottom: 1.8rem !important;
}
#run_btn {
    background: #1e5a99 !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: background 0.2s;
}
#run_btn:hover {
    background: #15416e !important;
}
.upload-box {
    border: 1px solid #dee2e6;
    border-radius: 10px;
}
#preset_select {
    border-radius: 10px !important;
    border: 1px solid #ced4da !important;
}
.result-container {
    max-width: 860px;
    margin: 1.2rem auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    padding: 1.8rem 2rem;
    border: 1px solid #d1d9e0;
}
.result-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0b2b44;
    margin-bottom: 1.2rem;
}
.info-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1.2rem;
}
.info-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #eee;
}
.info-table td:first-child {
    font-weight: 600;
    color: #333;
    width: 30%;
}
.badge-ok { color: #0e6b0e; font-weight: 700; }
.badge-no { color: #b91c1c; font-weight: 700; }
"""

# ==================== 模型加载 ====================
def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

def load_model():
    model = AblationCNNMamba(input_dim=4, n_cnn=4, n_mamba=1, dropout=0.4).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

MODEL = load_model()
SCALER_MEAN = np.load(SCALER_MEAN_PATH).astype(np.float32)
SCALER_SCALE = np.load(SCALER_SCALE_PATH).astype(np.float32)
setup_chinese_font()

def _warmup():
    dummy = torch.zeros(1, MAX_SEQ_LEN, 4, device=DEVICE)
    with torch.inference_mode():
        MODEL(dummy, dummy)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
_warmup()

# ==================== 数据处理 ====================
def _resolve_upload(upload):
    if upload is None: return None
    return upload.name if hasattr(upload, "name") else str(upload)

def _find_column(df, candidates):
    for name in candidates:
        if name in df.columns: return name
    return None

def load_track(csv_path):
    df = pd.read_csv(csv_path)
    cols = {}
    for feat, names in COL_MAPPING.items():
        col = _find_column(df, names)
        if col is None:
            raise ValueError(f"CSV缺少字段 {feat}，现有列：{list(df.columns)}")
        cols[feat] = col
    lat = df[cols["lat"]].values.astype(np.float32)
    lon = df[cols["lon"]].values.astype(np.float32)
    vel = df[cols["vel"]].values.astype(np.float32)
    cou = df[cols["cou"]].values.astype(np.float32)

    plot_df = pd.DataFrame({"lat": lat, "lon": lon, "vel": vel, "cou": cou})
    track = np.stack([lat, lon, vel, cou], axis=1)
    track = (track - SCALER_MEAN) / SCALER_SCALE
    if len(track) > MAX_SEQ_LEN:
        track = track[:MAX_SEQ_LEN]
        plot_df = plot_df.iloc[:MAX_SEQ_LEN].copy()
    elif len(track) < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - len(track)
        track = np.pad(track, ((0, pad_len),(0,0)), mode="constant")
    return plot_df, track.astype(np.float32)

def predict(track1, track2):
    t1 = torch.from_numpy(track1).unsqueeze(0).to(DEVICE)
    t2 = torch.from_numpy(track2).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logit = MODEL(t1, t2)
        prob = torch.sigmoid(logit.view(-1)[0]).item()
    return prob, prob >= THRESHOLD

# ==================== 绘图 ====================
def _view_bounds(df_a, df_b, pad_ratio=0.22):
    all_lon = np.concatenate([df_a["lon"], df_b["lon"]])
    all_lat = np.concatenate([df_a["lat"], df_b["lat"]])
    span_lon = max(all_lon.max() - all_lon.min(), 0.01)
    span_lat = max(all_lat.max() - all_lat.min(), 0.01)
    pad_lon = max(span_lon * pad_ratio, 0.015)
    pad_lat = max(span_lat * pad_ratio, 0.015)
    return (all_lon.min() - pad_lon, all_lon.max() + pad_lon,
            all_lat.min() - pad_lat, all_lat.max() + pad_lat)

def plot_result(df_a, df_b, prob, is_assoc):
    view = _view_bounds(df_a, df_b)
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.plot(df_a["lon"], df_a["lat"], color="#0b2b44", linewidth=2.2, label="航迹1 (源A)")
    ax.scatter(df_a["lon"], df_a["lat"], color="#0b2b44", s=18, zorder=3)
    ax.plot(df_b["lon"], df_b["lat"], color="#cc4c02", linewidth=2.2, label="航迹2 (源B)")
    ax.scatter(df_b["lon"], df_b["lat"], color="#cc4c02", s=18, zorder=3)

    ax.scatter(df_a["lon"].iloc[0], df_a["lat"].iloc[0], c="#2a7f2a", s=100, edgecolors="white", zorder=5, label="起点")
    ax.scatter(df_a["lon"].iloc[-1], df_a["lat"].iloc[-1], c="#c42e2e", s=100, marker="X", edgecolors="white", zorder=5, label="终点")
    ax.scatter(df_b["lon"].iloc[0], df_b["lat"].iloc[0], c="#2a7f2a", s=100, edgecolors="white", zorder=5)
    ax.scatter(df_b["lon"].iloc[-1], df_b["lat"].iloc[-1], c="#c42e2e", s=100, marker="X", edgecolors="white", zorder=5)

    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_xlabel("经度 (°)", fontsize=11, color="#333")
    ax.set_ylabel("纬度 (°)", fontsize=11, color="#333")
    ax.grid(True, alpha=0.3, linestyle="--", color="#999")
    result_str = "关联（同一目标）" if is_assoc else "非关联（不同目标）"
    ax.set_title(f"航迹关联结果：{result_str} | 概率 {prob:.4f}", fontsize=13, fontweight="bold", color="#0b2b44")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return fig

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ==================== 核心推理（输出为图片base64+文本） ====================
def run_inference(preset_name, file_a, file_b, progress=gr.Progress()):
    ground_truth = None
    if preset_name and preset_name != "自定义上传":
        preset = PRESET_SAMPLES[preset_name]
        path_a, path_b = preset["track1"], preset["track2"]
        ground_truth = preset["label"]
        source_hint = f"预设样本：{preset_name}"
    else:
        path_a = _resolve_upload(file_a)
        path_b = _resolve_upload(file_b)
        if not path_a or not path_b:
            return (
                None,  # 图像
                "<div class='result-container' style='color:#b91c1c;'>请先上传两条航迹 CSV 或选择内置示例。</div>",
                gr.update(visible=True)  # 显示结果区域
            )
        source_hint = "用户上传数据"

    try:
        progress(0.1, desc="加载航迹...")
        df_a, track_a = load_track(path_a)
        df_b, track_b = load_track(path_b)

        progress(0.5, desc="模型推理...")
        prob, is_assoc = predict(track_a, track_b)

        progress(0.8, desc="绘制结果...")
        fig = plot_result(df_a, df_b, prob, is_assoc)
        img_b64 = fig_to_base64(fig)

        # 构建HTML结果面板
        result_text = "关联（同一目标）" if is_assoc else "非关联（不同目标）"
        badge_class = "badge-ok" if is_assoc else "badge-no"
        gt_html = ""
        if ground_truth is not None:
            gt_text = "关联" if ground_truth == 1 else "非关联"
            gt_html = f"<tr><td>数据集真值</td><td>{gt_text}</td></tr>"

        html_result = f"""
        <div class="result-container">
            <div class="result-title">航迹关联分析结果</div>
            <div style="color:#5f6b7a; margin-bottom:1.2rem;">{source_hint}</div>
            <div style="text-align:center;">
                <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #e0e0e0; border-radius:8px;">
            </div>
            <table class="info-table">
                <tr><td>关联判定</td><td><span class="{badge_class}">{result_text}</span></td></tr>
                <tr><td>关联概率</td><td>{prob:.4f}</td></tr>
                <tr><td>判定阈值</td><td>{THRESHOLD:.4f}</td></tr>
                <tr><td>使用模型</td><td>CNN-BiMamba（调优版）</td></tr>
                <tr><td>推理设备</td><td>{DEVICE}</td></tr>
                {gt_html}
            </table>
        </div>
        """
        progress(1.0, desc="完成")
        return (
            None,  # 不再单独使用 gr.Image，而是嵌入 HTML
            html_result,
            gr.update(visible=True)
        )
    except Exception as exc:
        return (
            None,
            f"<div class='result-container' style='color:#b91c1c;'>推理失败：{exc}</div>",
            gr.update(visible=True)
        )

# ==================== UI 构建 ====================
def _find_free_port(start=7860, end=7870):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
    return start

def build_ui():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="gray",
        font=gr.themes.GoogleFont("Inter"),
    ).set(body_background_fill="#f8f9fa")

    with gr.Blocks(title="航迹关联演示系统", css=CUSTOM_CSS, theme=theme) as demo:
        # 主输入卡片
        with gr.Column(elem_classes=["main-card"]):
            gr.Markdown(
                '<h1 class="main-title">端到端的多源航迹关联系统</h1>'
                '<p class="sub-title">基于 CNN-BiMamba 的深度学习模型</p>'
            )
            preset = gr.Dropdown(
                choices=["自定义上传"] + list(PRESET_SAMPLES.keys()),
                value=list(PRESET_SAMPLES.keys())[0],
                label="航迹数据上传（CSV）",
                elem_id="preset_select",
            )
            with gr.Row():
                file_a = gr.File(label="航迹 A (CSV)", file_types=[".csv"], elem_classes=["upload-box"])
                file_b = gr.File(label="航迹 B (CSV)", file_types=[".csv"], elem_classes=["upload-box"])
            run_btn = gr.Button("开始推理", elem_id="run_btn")
            gr.Markdown(
                "📄 CSV 需包含：纬度、经度、航速、航向（支持中英文列名）"
            )

        # 隐藏的结果区域（推理后自动展开）
        result_area = gr.Column(visible=False)
        with result_area:
            # 我们不再使用 gr.Image，而是用一个 gr.HTML 承载完整结果（图片+表格）
            result_html = gr.HTML()

        # 绑定事件
        run_btn.click(
            fn=run_inference,
            inputs=[preset, file_a, file_b],
            outputs=[gr.Image(visible=False), result_html, result_area],  # 第一个输出为哑元，兼容旧接口
        )

    return demo

if __name__ == "__main__":
    print(f"🚢 运行设备: {DEVICE}")
    port = int(os.environ.get("GRADIO_SERVER_PORT", _find_free_port()))
    ui = build_ui()
    ui.queue(default_concurrency_limit=1)
    ui.launch(server_name="0.0.0.0", server_port=port)