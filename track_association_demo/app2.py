# -*- coding: utf-8 -*-
"""航迹关联 Web 演示系统（Gradio）—— 毕业答辩展示用（无背景地图版）。"""

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

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_DATA_DIR = os.path.join(DEMO_DIR, "track_data")
MODEL_PATH = os.path.join(DEMO_DIR, "g_cnn4_drop04_best.pth")
SCALER_MEAN_PATH = os.path.join(DEMO_DIR, "scaler_mean.npy")
SCALER_SCALE_PATH = os.path.join(DEMO_DIR, "scaler_scale.npy")

MAX_SEQ_LEN = 350
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRESET_SAMPLES = {
    "示例 1：关联航迹（同一目标，sample_10009）": {
        "track1": os.path.join(TRACK_DATA_DIR, "positive_track1.csv"),
        "track2": os.path.join(TRACK_DATA_DIR, "positive_track2.csv"),
        "label": 1,
    },
    "示例 2：非关联航迹（不同目标，sample_6322）": {
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

CUSTOM_CSS = """
.result-overlay {
    position: fixed !important;
    inset: 0 !important;
    z-index: 9999 !important;
    overflow-y: auto !important;
    padding: 1rem 1.2rem 1.5rem !important;
    background: rgba(245, 248, 252, 0.98) !important;
}
.result-title {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #1a365d !important;
}
.result-card {
    background: #ffffff;
    border: 1px solid #cbd5e0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    line-height: 1.9;
    color: #2d3748;
}
.badge-ok {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: #38a169;
    color: white;
    font-weight: 700;
}
.badge-no {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: #e53e3e;
    color: white;
    font-weight: 700;
}
#close-btn {
    max-width: 140px;
}
"""


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP", "SimHei", "Microsoft YaHei", "DejaVu Sans",
    ]
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

RESULT_CACHE = {}

setup_chinese_font()


def _warmup_model():
    dummy = torch.zeros(1, MAX_SEQ_LEN, 4, device=DEVICE)
    with torch.inference_mode():
        MODEL(dummy, dummy)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


_warmup_model()
print("[INFO] 模型预热完成，后续推理会更快")


def _resolve_upload_path(upload):
    if upload is None:
        return None
    if isinstance(upload, str):
        return upload
    if hasattr(upload, "name"):
        return upload.name
    return str(upload)


def _find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def load_and_preprocess_track(csv_path, scaler_mean, scaler_scale):
    df = pd.read_csv(csv_path)
    feature_cols = {}
    for feat, possible_names in COL_MAPPING.items():
        col = _find_column(df, possible_names)
        if col is None:
            raise ValueError(f"CSV 缺少必要字段（{feat}）。当前列: {list(df.columns)}")
        feature_cols[feat] = col

    lat = df[feature_cols["lat"]].values.astype(np.float32)
    lon = df[feature_cols["lon"]].values.astype(np.float32)
    vel = df[feature_cols["vel"]].values.astype(np.float32)
    cou = df[feature_cols["cou"]].values.astype(np.float32)

    plot_df = pd.DataFrame({"lat": lat, "lon": lon, "vel": vel, "cou": cou})
    track = np.stack([lat, lon, vel, cou], axis=1)
    track = (track - scaler_mean) / scaler_scale

    if len(track) > MAX_SEQ_LEN:
        track = track[:MAX_SEQ_LEN]
        plot_df = plot_df.iloc[:MAX_SEQ_LEN].copy()
    elif len(track) < MAX_SEQ_LEN:
        pad_len = MAX_SEQ_LEN - len(track)
        track = np.pad(track, ((0, pad_len), (0, 0)), mode="constant")

    return plot_df, track.astype(np.float32)


def run_inference(track1, track2):
    t1 = torch.from_numpy(track1).unsqueeze(0).to(DEVICE)
    t2 = torch.from_numpy(track2).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logit = MODEL(t1, t2)
        prob = torch.sigmoid(logit.view(-1)[0]).item()
    return prob, prob >= THRESHOLD


def _view_bounds(df_a, df_b, padding_ratio=0.22):
    all_lon = np.concatenate([df_a["lon"].values, df_b["lon"].values])
    all_lat = np.concatenate([df_a["lat"].values, df_b["lat"].values])
    lon_span = max(all_lon.max() - all_lon.min(), 0.01)
    lat_span = max(all_lat.max() - all_lat.min(), 0.01)
    pad_lon = max(lon_span * padding_ratio, 0.015)
    pad_lat = max(lat_span * padding_ratio, 0.015)
    return (
        all_lon.min() - pad_lon,
        all_lon.max() + pad_lon,
        all_lat.min() - pad_lat,
        all_lat.max() + pad_lat,
    )


def _style_track(ax, lon, lat, color, label):
    ax.plot(lon, lat, color=color, linewidth=2.2, alpha=0.92, zorder=3)
    ax.scatter(
        lon, lat, color=color, s=22, edgecolors="white", linewidths=0.5,
        zorder=4, label=label,
    )


def plot_result(df_a, df_b, prob, is_associated):
    """绘制航迹关联结果（纯白背景，无地图底图）。"""
    view = _view_bounds(df_a, df_b)

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="white")

    _style_track(ax, df_a["lon"], df_a["lat"], "#0066cc", "航迹1（源A）")
    _style_track(ax, df_b["lon"], df_b["lat"], "#ff6600", "航迹2（源B）")

    ax.scatter(
        df_a["lon"].iloc[0], df_a["lat"].iloc[0], c="#00aa44", s=110,
        edgecolors="white", linewidths=1.2, zorder=5, label="起点",
    )
    ax.scatter(
        df_a["lon"].iloc[-1], df_a["lat"].iloc[-1], c="#cc0000", s=110,
        marker="X", edgecolors="white", linewidths=1.0, zorder=5, label="终点",
    )
    ax.scatter(df_b["lon"].iloc[0], df_b["lat"].iloc[0], c="#00aa44", s=110,
               edgecolors="white", linewidths=1.2, zorder=5)
    ax.scatter(df_b["lon"].iloc[-1], df_b["lat"].iloc[-1], c="#cc0000", s=110,
               marker="X", edgecolors="white", linewidths=1.0, zorder=5)

    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_xlabel("经度 (°)", fontsize=12)
    ax.set_ylabel("纬度 (°)", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--", color="#666666")

    result_text = "关联（同一目标）" if is_associated else "非关联（不同目标）"
    title = f"航迹关联结果：{result_text}  |  关联概率 {prob:.4f}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    fig.tight_layout()
    return fig


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_result_html(prob, is_associated, source_hint, ground_truth=None):
    badge = "badge-ok" if is_associated else "badge-no"
    result_text = "关联（同一目标）" if is_associated else "非关联（不同目标）"
    html = f"""
<div class="result-card">
  <div style="margin-bottom:0.6rem;color:#4a5568;">{source_hint}</div>
  <div class="{badge}" style="margin-bottom:0.8rem;">{result_text}</div>
  <div><b>关联概率：</b>{prob:.4f}</div>
  <div><b>判定阈值：</b>{THRESHOLD:.4f}</div>
  <div><b>使用模型：</b>CNN-BiMamba（调优版）</div>
  <div><b>推理设备：</b>{DEVICE}</div>
"""
    if ground_truth is not None:
        gt_text = "关联" if ground_truth == 1 else "非关联"
        html += f"<div><b>数据集真值：</b>{gt_text}</div>"
    html += "</div>"
    return html


def predict_association(preset_name, file_a, file_b, progress=gr.Progress()):
    ground_truth = None

    if preset_name and preset_name != "自定义上传":
        preset = PRESET_SAMPLES[preset_name]
        path_a, path_b = preset["track1"], preset["track2"]
        ground_truth = preset["label"]
        source_hint = f"当前样本：{preset_name}"
        cache_key = ("preset", preset_name)
    else:
        path_a = _resolve_upload_path(file_a)
        path_b = _resolve_upload_path(file_b)
        if not path_a or not path_b:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                "<div class='result-card'>请先上传两条航迹 CSV，或选择内置示例。</div>",
            )
        source_hint = "当前样本：用户上传"
        cache_key = ("upload", path_a, path_b)

    if cache_key in RESULT_CACHE:
        progress(1.0, desc="命中缓存，立即展示")
        cached = RESULT_CACHE[cache_key]
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            cached["image"],
            cached["html"],
        )

    try:
        progress(0.2, desc="读取航迹数据...")
        df_a, track_a = load_and_preprocess_track(path_a, SCALER_MEAN, SCALER_SCALE)
        df_b, track_b = load_and_preprocess_track(path_b, SCALER_MEAN, SCALER_SCALE)

        progress(0.55, desc="模型推理中...")
        prob, is_associated = run_inference(track_a, track_b)

        progress(0.85, desc="绘制结果地图...")
        fig = plot_result(df_a, df_b, prob, is_associated)
        map_image = fig_to_pil(fig)
        result_html = build_result_html(prob, is_associated, source_hint, ground_truth)

        RESULT_CACHE[cache_key] = {"image": map_image, "html": result_html}
        progress(1.0, desc="完成")

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            map_image,
            result_html,
        )
    except Exception as exc:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            f"<div class='result-card' style='color:#c53030;'>运行出错：{exc}</div>",
        )


def back_to_main():
    return gr.update(visible=True), gr.update(visible=False)


def _find_free_port(start=7860, end=7870):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
    return start


def build_demo():
    with gr.Blocks(title="航迹关联演示系统", css=CUSTOM_CSS) as demo:
        with gr.Column(visible=True) as main_page:
            gr.Markdown(
                """
# 基于 CNN-BiMamba 的多源 AIS 航迹关联演示系统

上传两条 AIS 航迹 CSV，或使用内置测试样本。推理完成后将**弹出结果页面**，
在自定义坐标格网上展示双源航迹与关联判别结果。
                """
            )
            preset = gr.Dropdown(
                choices=["自定义上传"] + list(PRESET_SAMPLES.keys()),
                value=list(PRESET_SAMPLES.keys())[0],
                label="选择演示样本",
            )
            with gr.Row():
                file_a = gr.File(label="航迹 A CSV（自定义上传时使用）", file_types=[".csv"])
                file_b = gr.File(label="航迹 B CSV（自定义上传时使用）", file_types=[".csv"])
            run_btn = gr.Button("开始关联推理", variant="primary")
            gr.Markdown(
                """
**CSV 字段要求：** 至少包含纬度、经度、航速、航向四列。
系统将自动适配字段名（支持中/英文），并以经纬度为坐标绘制航迹。
                """
            )

        with gr.Column(visible=False, elem_classes=["result-overlay"]) as result_page:
            with gr.Row():
                close_btn = gr.Button("← 返回", elem_id="close-btn", scale=0)
                gr.Markdown('<div class="result-title">航迹关联结果</div>')
            with gr.Row():
                with gr.Column(scale=3):
                    result_map = gr.Image(label="航迹地图", type="pil", height=560, show_label=False)
                with gr.Column(scale=1):
                    result_panel = gr.HTML()

        run_btn.click(
            fn=predict_association,
            inputs=[preset, file_a, file_b],
            outputs=[main_page, result_page, result_map, result_panel],
        )
        close_btn.click(fn=back_to_main, outputs=[main_page, result_page])

    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    port = int(os.environ.get("GRADIO_SERVER_PORT", _find_free_port()))
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=port)