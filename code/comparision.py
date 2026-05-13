import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===================== 1. 配置与参数 =====================
parser = argparse.ArgumentParser(description='Multi-Model Comparison on Test Sets')
parser.add_argument('--data_dir', type=str, 
                    default='/home/yangcq/track_association/data/final_dataset',
                    help='Path to the dataset directory')
parser.add_argument('--model_dir', type=str,
                    default='/home/yangcq/track_association/output/model',
                    help='Path to the trained models directory')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===================== 2. 动态 Padding Collate_fn =====================
def collate_fn(batch):
    t1_list, t2_list, lab_list = [], [], []
    max_len = 0
    for t1, t2, _ in batch:
        max_len = max(max_len, len(t1), len(t2))

    for t1, t2, lab in batch:
        pad1 = max_len - len(t1)
        if pad1 > 0:
            t1_pad = np.pad(t1, ((0, pad1), (0, 0)), mode='edge')
        else:
            t1_pad = t1

        pad2 = max_len - len(t2)
        if pad2 > 0:
            t2_pad = np.pad(t2, ((0, pad2), (0, 0)), mode='edge')
        else:
            t2_pad = t2

        t1_list.append(t1_pad)
        t2_list.append(t2_pad)
        lab_list.append(lab)

    return (torch.FloatTensor(np.array(t1_list)),
            torch.FloatTensor(np.array(t2_list)),
            torch.FloatTensor(np.array(lab_list)))

# ===================== 3. Dataset 支持变长数据 =====================
class AISTrackPairDataset(Dataset):
    def __init__(self, track1_path, track2_path, labels_path):
        self.track1 = np.load(track1_path, allow_pickle=True)
        self.track2 = np.load(track2_path, allow_pickle=True)
        self.labels = np.load(labels_path).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.track1[idx], self.track2[idx], self.labels[idx]

# ===================== 4. 模型加载 =====================
def load_model(model_name, model_dir, device):
    print(f"\nLoading model: {model_name.upper()}...")
    try:
        if model_name == 'cnn_mamba':
            from utils.model_cnn_mamba import CNNTrajectoryMatcher as Model
        elif model_name == 'cnn_lstm':
            from utils.model_cnn_lstm import CNNTrajectoryMatcher_LSTM as Model
        elif model_name == 'cnn_bilstm':
            from utils.model_cnn_bilstm import CNNTrajectoryMatcher_BiLSTM as Model
        elif model_name == 'bilstm':
            from utils.model_bilstm import TrajectoryMatcher_BiLSTM as Model
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = Model().to(device)
        model_path = os.path.join(model_dir, f"best_{model_name}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Successfully loaded {model_name}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None

# ===================== 5. 评估函数 =====================
def evaluate_model(model, loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for traj_a, traj_b, labels in tqdm(loader, desc="Evaluating"):
            traj_a, traj_b, labels = traj_a.to(device), traj_b.to(device), labels.to(device)
            outputs = model(traj_a, traj_b).squeeze(-1)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    acc = np.mean(all_labels == all_preds)

    tp = np.sum((all_labels == 1) & (all_preds == 1))
    fp = np.sum((all_labels == 0) & (all_preds == 1))
    fn = np.sum((all_labels == 1) & (all_preds == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"Accuracy": acc, "F1": f1, "Precision": precision, "Recall": recall}

# ===================== 6. 画图并保存 =====================
def plot_and_save_results(results, model_names, test_set_names, save_path="model_comparison.png"):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.figure(figsize=(14, 10))

    metrics = ["Accuracy", "F1"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bar_width = 0.2
    index = np.arange(len(test_set_names))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 1, i+1)

        for j, model in enumerate(model_names):
            values = [results[model][ts][metric] for ts in test_set_names]
            plt.bar(index + j*bar_width, values, bar_width, label=model, color=colors[j])

        plt.xlabel('Test Set')
        plt.ylabel(metric)
        plt.title(f'{metric} Comparison Across Models')
        plt.xticks(index + bar_width * (len(model_names)-1) / 2, test_set_names)
        plt.ylim(0.8, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Comparison chart saved to: {save_path}")

# ===================== 7. 主程序 =====================
def main():
    print("="*80)
    print(" Multi-Model Evaluation on 3 Test Sets (Short/Medium/Long)")
    print("="*80)

    model_names = ['cnn_mamba', 'cnn_lstm', 'cnn_bilstm', 'bilstm']

    test_sets = {
        "Full Test": ("track1_test.npy", "track2_test.npy", "labels_test.npy"),
        "Short Track": ("track1_test_short.npy", "track2_test_short.npy", "labels_test_short.npy"),
        "Medium Track": ("track1_test_medium.npy", "track2_test_medium.npy", "labels_test_medium.npy"),
        "Long Track": ("track1_test_long.npy", "track2_test_long.npy", "labels_test_long.npy"),
    }
    test_set_names = list(test_sets.keys())

    print("\n[1/4] Checking files...")
    for t1, t2, lab in test_sets.values():
        for f in [t1, t2, lab]:
            fp = os.path.join(args.data_dir, f)
            if not os.path.exists(fp):
                print("Missing", fp)
                return
    print("✅ All test files exist")

    print("\n[2/4] Loading models...")
    models = {}
    for name in model_names:
        m = load_model(name, args.model_dir, DEVICE)
        if m: models[name] = m
    if not models: return

    print("\n[3/4] Starting evaluation...")
    results = {name: {} for name in models}

    for set_name, (t1f, t2f, labf) in test_sets.items():
        print(f"\n--- {set_name} ---")
        t1p = os.path.join(args.data_dir, t1f)
        t2p = os.path.join(args.data_dir, t2f)
        labp = os.path.join(args.data_dir, labf)

        ds = AISTrackPairDataset(t1p, t2p, labp)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        for name, model in models.items():
            res = evaluate_model(model, loader, DEVICE)
            results[name][set_name] = res
            print(f"{name:15} Acc:{res['Accuracy']:.4f}  F1:{res['F1']:.4f}")

    print("\n[4/4] Generating comparison chart...")
    plot_and_save_results(results, model_names, test_set_names)

    print("\n" + "="*80)
    print(" FINAL RESULTS TABLE")
    print("="*80)
    for name in models:
        for ts in test_set_names:
            r = results[name][ts]
            print(f"{name:15} | {ts:15} | Acc:{r['Accuracy']:.4f}  F1:{r['F1']:.4f}")

if __name__ == "__main__":
    main()