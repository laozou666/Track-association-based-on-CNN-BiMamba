import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

# ===================== 路径配置 =====================
PROJECT_ROOT = "/home/yangcq/track_association"  # 你的项目根目录路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TEST_X = os.path.join(DATA_DIR, "test", "X_归一化.npy")
TEST_Y = os.path.join(DATA_DIR, "test", "y_标签.npy")
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "model", "cnn_mamba_best.pth")
# 预测结果保存路径
PRED_SAVE_DIR = os.path.join(PROJECT_ROOT, "output", "predictions")
os.makedirs(PRED_SAVE_DIR, exist_ok=True)
PRED_LABELS_PATH = os.path.join(PRED_SAVE_DIR, "test_pred_labels.npy")
PRED_PROBS_PATH = os.path.join(PRED_SAVE_DIR, "test_pred_probs.npy")
TRUE_LABELS_PATH = os.path.join(PRED_SAVE_DIR, "test_true_labels.npy")
# ====================================================================

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

# -------------------------- 数据加载器 --------------------------
class MTADDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        traj_a = torch.tensor(self.X[idx, 0], dtype=torch.float32)
        traj_b = torch.tensor(self.X[idx, 1], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return traj_a, traj_b, label

# 验证文件存在
def check_file_exists():
    file_list = [TEST_X, TEST_Y, MODEL_PATH]
    for f in file_list:
        if not os.path.exists(f):
            print(f"❌ 文件不存在：{f}")
            return False
    print("✅ 所有文件均存在，路径正确！")
    return True

if not check_file_exists():
    exit()

# 加载数据
print("正在加载测试数据...")
test_dataset = MTADDataset(TEST_X, TEST_Y)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"测试集加载完成！共 {len(test_dataset)} 个样本")

# -------------------------- 模型定义（和训练时完全一致） --------------------------
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
            nn.Sigmoid()
        )
    
    def forward(self, traj_a, traj_b):
        feat_a = self.cnn_extractor(traj_a)
        feat_b = self.cnn_extractor(traj_b)
        global_a = self.mamba_encoder(feat_a)
        global_b = self.mamba_encoder(feat_b)
        combined = torch.cat([global_a, global_b], dim=1)
        return self.classifier(combined).squeeze(-1)

# -------------------------- 加载模型并推理 --------------------------
print("\n正在加载最佳模型...")
model = CNNTrajectoryMatcher().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ 模型加载成功！")

# 测试集推理
all_true_labels = []
all_pred_labels = []
all_pred_probs = []

print("\n开始测试集推理...")
with torch.no_grad():
    for traj_a, traj_b, labels in tqdm(test_loader, desc="推理中"):
        traj_a, traj_b, labels = traj_a.to(DEVICE), traj_b.to(DEVICE), labels.to(DEVICE)
        outputs = model(traj_a, traj_b)
        preds = (outputs > 0.5).float()
        
        all_true_labels.extend(labels.cpu().numpy())
        all_pred_labels.extend(preds.cpu().numpy())
        all_pred_probs.extend(outputs.cpu().numpy())

# 保存预测结果
np.save(TRUE_LABELS_PATH, np.array(all_true_labels))
np.save(PRED_LABELS_PATH, np.array(all_pred_labels))
np.save(PRED_PROBS_PATH, np.array(all_pred_probs))

print("\n✅ 测试推理完成！")
print(f"预测结果已保存到：{PRED_SAVE_DIR}")
print("  - test_true_labels.npy (真实标签)")
print("  - test_pred_labels.npy (预测标签)")
print("  - test_pred_probs.npy (预测概率)")