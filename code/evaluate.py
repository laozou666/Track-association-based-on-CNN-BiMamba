import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score,
                             classification_report)
import seaborn as sns

# ===================== 导入你的模型和数据集类 =====================
# 注意：确保这些文件在正确的位置
try:
    from utils.model import CNNTrajectoryMatcher
    print("✅ 成功导入模型 CNNTrajectoryMatcher")
except ImportError:
    print("⚠️  未找到 utils.model，使用之前定义的模型")
    # 这里复制你之前的模型定义，确保可以独立运行
    import torch.nn as nn
    from mamba_ssm import Mamba

    class CNNFeatureExtractor(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=64):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim*2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        
        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.conv_layers(x)
            return x.permute(0, 2, 1)

    class MambaEncoder(nn.Module):
        def __init__(self, input_dim=64, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.mamba = Mamba(
                d_model=input_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.norm = nn.LayerNorm(input_dim)
        
        def forward(self, x):
            x = self.norm(x)
            x = self.mamba(x)
            return x[:, -1, :]

    class CNNTrajectoryMatcher(nn.Module):
        def __init__(self, input_dim=4, cnn_hidden=64, mamba_d_state=16):
            super().__init__()
            self.cnn_extractor = CNNFeatureExtractor(input_dim=input_dim, hidden_dim=cnn_hidden)
            self.mamba_encoder = MambaEncoder(input_dim=cnn_hidden, d_state=mamba_d_state)
            self.classifier = nn.Sequential(
                nn.Linear(cnn_hidden * 2, cnn_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(cnn_hidden, 1),
            )
        
        def forward(self, traj_a, traj_b):
            feat_a = self.cnn_extractor(traj_a)
            feat_b = self.cnn_extractor(traj_b)
            global_a = self.mamba_encoder(feat_a)
            global_b = self.mamba_encoder(feat_b)
            combined = torch.cat([global_a, global_b], dim=1)
            return self.classifier(combined).squeeze(-1)

# ===================== 数据集类（和训练保持一致） =====================
class AISTrackPairDataset(torch.utils.data.Dataset):
    def __init__(self, track1_path, track2_path, labels_path):
        self.track1 = np.load(track1_path).astype(np.float32)
        self.track2 = np.load(track2_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.track1[idx], self.track2[idx], self.labels[idx]

# ===================== 路径配置 =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "track_pairs")

# 测试集路径
TEST_TRACK1 = os.path.join(DATA_DIR, "track1_test.npy")
TEST_TRACK2 = os.path.join(DATA_DIR, "track2_test.npy")
TEST_LABELS = os.path.join(DATA_DIR, "labels_test.npy")

# 模型路径
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "model", "cnn_mamba_ais_real.pth")
# 混淆矩阵保存路径
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_ROOT, "output", "figures", "confusion_matrix.png")
os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)

# 超参数
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5  # 分类阈值

# ===================== 核心评估函数 =====================
def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    在测试集上评估模型
    返回：所有预测结果、所有真实标签、指标字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # 保存概率值，用于PR曲线等
    
    with torch.no_grad():
        for traj_a, traj_b, labels in test_loader:
            traj_a, traj_b, labels = traj_a.to(device), traj_b.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(traj_a, traj_b)
            probs = torch.sigmoid(outputs)  # 转换为概率
            preds = (probs > threshold).float()  # 二分类预测
            
            # 收集结果
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    metrics = {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, zero_division=0),
        "Recall": recall_score(all_labels, all_preds, zero_division=0),
        "F1": f1_score(all_labels, all_preds, zero_division=0),
        "ConfusionMatrix": confusion_matrix(all_labels, all_preds)
    }
    
    return all_preds, all_labels, all_probs, metrics

def plot_confusion_matrix(cm, save_path, class_names=["不关联 (0)", "关联 (1)"]):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                annot_kws={"size": 16})
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.title('混淆矩阵', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"✅ 混淆矩阵已保存到：{save_path}")
    plt.show()
    plt.close()

def print_evaluation_report(metrics):
    """打印详细的评估报告"""
    print("\n" + "="*80)
    print(" 模型评估报告")
    print("="*80)
    
    cm = metrics["ConfusionMatrix"]
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n【混淆矩阵】")
    print(f"  真负例 (TN): {tn:>6}  |  假正例 (FP): {fp:>6}")
    print(f"  假负例 (FN): {fn:>6}  |  真正例 (TP): {tp:>6}")
    
    print(f"\n【核心指标】")
    print(f"  准确率 (Accuracy):  {metrics['Accuracy']:.4f}  ({metrics['Accuracy']*100:.2f}%)")
    print(f"  精确率 (Precision): {metrics['Precision']:.4f}  ({metrics['Precision']*100:.2f}%)")
    print(f"  召回率 (Recall):    {metrics['Recall']:.4f}  ({metrics['Recall']*100:.2f}%)")
    print(f"  F1 分数 (F1-Score): {metrics['F1']:.4f}  ({metrics['F1']*100:.2f}%)")
    
    print(f"\n【指标解释】")
    print(f"  准确率：整体预测正确的比例")
    print(f"  精确率：预测为'关联'的样本中，真正关联的比例（避免误报）")
    print(f"  召回率：真正关联的样本中，被正确预测的比例（避免漏报）")
    print(f"  F1分数：精确率和召回率的调和平均，综合指标")
    print("="*80)

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("="*80)
    print(" CNN-Mamba 航迹关联模型 - 测试集评估")
    print("="*80)
    
    # 1. 检查文件是否存在
    print("\n[1/4] 检查文件...")
    file_list = [TEST_TRACK1, TEST_TRACK2, TEST_LABELS, MODEL_PATH]
    for f in file_list:
        if not os.path.exists(f):
            print(f"❌ 文件不存在：{f}")
            exit()
    print("✅ 所有文件均存在")
    
    # 2. 加载测试集
    print("\n[2/4] 加载测试集...")
    test_dataset = AISTrackPairDataset(TEST_TRACK1, TEST_TRACK2, TEST_LABELS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ 测试集加载完成，共 {len(test_dataset)} 个样本")
    print(f"   正样本数：{np.sum(test_dataset.labels == 1)}")
    print(f"   负样本数：{np.sum(test_dataset.labels == 0)}")
    
    # 3. 加载模型
    print("\n[3/4] 加载模型...")
    model = CNNTrajectoryMatcher().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ 模型加载成功：{MODEL_PATH}")
    print(f"   模型参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 4. 执行评估
    print("\n[4/4] 开始评估...")
    preds, labels, probs, metrics = evaluate_model(model, test_loader, DEVICE, THRESHOLD)
    
    # 5. 打印报告
    print_evaluation_report(metrics)
    
    # 6. 绘制混淆矩阵
    plot_confusion_matrix(metrics["ConfusionMatrix"], CONFUSION_MATRIX_PATH)
    
    print("\n🎉 评估完成！")