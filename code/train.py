import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# ===================== 1. 命令行参数解析 =====================
parser = argparse.ArgumentParser(description='Train different trajectory matching models')
parser.add_argument('--model', type=str, required=True, 
                    choices=['cnn_mamba', 'cnn_lstm', 'cnn_bilstm', 'bilstm','cnn','lstm','bigru','ann'],
                    help='Which model to train: mamba, cnn_lstm, cnn_bilstm,  bilstm, cnn, lstm, bigru')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=None, help='Learning rate (optional, auto-set if not provided)')
args = parser.parse_args()

# ===================== 2. 根据参数选择模型和路径 =====================
MODEL_NAME = args.model
print(f"🎯 选择训练模型：{MODEL_NAME.upper()}")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 【修改1】指向新的变长数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "final_dataset")

TRAIN_TRACK1 = os.path.join(DATA_DIR, "track1_train.npy")
TRAIN_TRACK2 = os.path.join(DATA_DIR, "track2_train.npy")
TRAIN_LABELS = os.path.join(DATA_DIR, "labels_train.npy")
VAL_TRACK1 = os.path.join(DATA_DIR, "track1_val.npy")
VAL_TRACK2 = os.path.join(DATA_DIR, "track2_val.npy")
VAL_LABELS = os.path.join(DATA_DIR, "labels_val.npy")

SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "model", f"best_{MODEL_NAME}.pth")
SAVE_CURVE_PATH = os.path.join(PROJECT_ROOT, "output", "figures", f"training_curve_{MODEL_NAME}.png")
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SAVE_CURVE_PATH), exist_ok=True)

BATCH_SIZE = args.batch_size
VAL_BATCH_SIZE = 16
EPOCHS = args.epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

# 【修改2】自适应学习率（保留之前的修复）
if args.lr is not None:
    LR = args.lr
else:
    if MODEL_NAME == 'cnn_mamba':
        LR = 2e-4
    elif MODEL_NAME == 'cnn_lstm':
        LR = 3e-4
    elif MODEL_NAME == 'cnn_bilstm':
        LR = 1e-3
    elif MODEL_NAME == 'bilstm':
        LR = 2e-4
    # ========== 【新增】4个对比模型的学习率 ==========
    elif MODEL_NAME == 'ann':
        LR = 1e-3      # 纯MLP：结构简单，学习率稍大
    elif MODEL_NAME == 'cnn':
        LR = 3e-4      # 纯CNN：中等学习率
    elif MODEL_NAME == 'lstm':
        LR = 1e-3      # 单向LSTM：需要更大学习率穿透padding噪声
    elif MODEL_NAME == 'bigru':
        LR = 1e-3      # BiGRU：轻量化双向，学习率稍大
    # ==================================================
print(f"设置学习率：{LR}")

# ===================== 3. 导入模型 =====================
print(f"\n正在导入模型 {MODEL_NAME}...")
try:
    if MODEL_NAME == 'cnn_mamba':
        from utils.model_cnn_mamba import CNNTrajectoryMatcher as Model
    elif MODEL_NAME == 'cnn_lstm':
        from utils.model_cnn_lstm import CNNTrajectoryMatcher_LSTM as Model
    elif MODEL_NAME == 'cnn_bilstm':
        from utils.model_cnn_bilstm import CNNTrajectoryMatcher_BiLSTM as Model
    elif MODEL_NAME == 'bilstm':
        from utils.model_bilstm import TrajectoryMatcher_BiLSTM as Model
    elif MODEL_NAME == 'ann':
        from utils.model_ann import TrajectoryMatcher_ANN as Model
    elif MODEL_NAME == 'cnn':           
        from utils.model_cnn import TrajectoryMatcher_CNN as Model
    elif MODEL_NAME == 'lstm':
        from utils.model_lstm import TrajectoryMatcher_LSTM as Model
    elif MODEL_NAME == 'bigru':
        from utils.model_bigru import TrajectoryMatcher_BiGRU as Model
    else:
        raise ValueError(f"Unknown model: {MODEL_NAME}")
    print(f"✅ 成功导入模型")
except ImportError as e:
    print(f"❌ 导入模型失败：{e}")
    print("请确保对应的模型文件存在于 utils/ 目录下")
    exit()

# ===================== 4. 【修改3】适配变长数据的 Dataset =====================
class AISTrackPairDataset(Dataset):
    """适配变长双航迹输入的数据集类"""
    def __init__(self, track1_path, track2_path, labels_path):
        # 加载 object 格式的变长 npy
        self.track1 = np.load(track1_path, allow_pickle=True)
        self.track2 = np.load(track2_path, allow_pickle=True)
        self.labels = np.load(labels_path).astype(np.float32)
        
        print(f"数据集加载完成：")
        print(f"  样本数：{len(self.labels)}")
        print(f"  正样本数：{np.sum(self.labels == 1)}")
        print(f"  负样本数：{np.sum(self.labels == 0)}") 
      
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 返回原始 numpy array，由 collate_fn 处理 padding
        return self.track1[idx], self.track2[idx], self.labels[idx]

# ===================== 5. 【修改4】核心：动态 batch padding collate_fn =====================
def collate_fn(batch):
    """
    动态 batch padding：
    1. 只补到当前 batch 最长长度
    2. 用 mode='edge' 复制最后一点，绝不补0
    """
    t1_list, t2_list, lab_list = [], [], []
    
    # 找当前 batch 最长长度
    max_len = 0
    for t1, t2, _ in batch:
        max_len = max(max_len, len(t1), len(t2))
    
    # 对每个样本 padding
    for t1, t2, lab in batch:
        # padding track1
        pad1 = max_len - len(t1)
        if pad1 > 0:
            t1_pad = np.pad(t1, ((0, pad1), (0, 0)), mode='edge')
        else:
            t1_pad = t1
        
        # padding track2
        pad2 = max_len - len(t2)
        if pad2 > 0:
            t2_pad = np.pad(t2, ((0, pad2), (0, 0)), mode='edge')
        else:
            t2_pad = t2
        
        t1_list.append(t1_pad)
        t2_list.append(t2_pad)
        lab_list.append(lab)
    
    # 转成 Tensor
    return (torch.FloatTensor(np.array(t1_list)),
            torch.FloatTensor(np.array(t2_list)),
            torch.FloatTensor(np.array(lab_list)))

# ===================== 验证文件存在 =====================
def check_file_exists():
    file_list = [TRAIN_TRACK1, TRAIN_TRACK2, TRAIN_LABELS,
                 VAL_TRACK1, VAL_TRACK2, VAL_LABELS]
    for f in file_list:
        if not os.path.exists(f):
            print(f"❌ 文件不存在：{f}")
            return False
    print("✅ 所有数据集文件均存在，路径正确！")
    return True

if not check_file_exists():
    exit()

# ===================== 6. 【修改5】加载数据 + DataLoader 加 collate_fn =====================
print("\n正在加载数据...")
train_dataset = AISTrackPairDataset(TRAIN_TRACK1, TRAIN_TRACK2, TRAIN_LABELS)
val_dataset = AISTrackPairDataset(VAL_TRACK1, VAL_TRACK2, VAL_LABELS)

# 【关键】加上 collate_fn=collate_fn
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
print(f"\n数据加载完成！")
print(f"  训练集：{len(train_dataset)} 个样本")
print(f"  验证集：{len(val_dataset)} 个样本")

# ===================== 计算正负样本权重 =====================
train_labels = np.load(TRAIN_LABELS)
pos_num = np.sum(train_labels == 1)
neg_num = np.sum(train_labels == 0)
pos_weight = torch.tensor([neg_num / pos_num], device=DEVICE)
print(f"\n正负样本统计：")
print(f"  正样本：{pos_num}")
print(f"  负样本：{neg_num}")
print(f"  正样本权重：{pos_weight.item():.2f}")

# ===================== 指标计算 =====================
try:
    from utils.metrics import calculate_all_metrics
    print("✅ 成功导入指标计算函数")
except ImportError:
    print("⚠️  未找到 utils.metrics，使用简单的准确率计算")
    def calculate_all_metrics(labels, preds, verbose=True):
        acc = np.mean(labels == preds)
        return {"Accuracy": acc, "F1": acc}

# ===================== 初始化模型 =====================
print("\n正在初始化模型...")
model = Model().to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
print("✅ 模型初始化完成！")
print(f"模型参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

# ===================== 早停法 =====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  早停法计数器：{self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

# ===================== 7. 【修改6】训练/验证函数（保留 squeeze + 梯度裁剪） =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for traj_a, traj_b, labels in tqdm(loader, desc="训练中"):
        traj_a, traj_b, labels = traj_a.to(device), traj_b.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 【修复】输出 squeeze
        outputs = model(traj_a, traj_b).squeeze(-1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        # 【修复】梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * traj_a.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for traj_a, traj_b, labels in tqdm(loader, desc="验证中"):
            traj_a, traj_b, labels = traj_a.to(device), traj_b.to(device), labels.to(device)
            
            # 【修复】验证时也 squeeze
            outputs = model(traj_a, traj_b).squeeze(-1)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * traj_a.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_preds)

# ===================== 主训练流程 =====================
print("\n" + "="*80)
print(f"开始训练 {MODEL_NAME.upper()} 航迹关联模型")
print("="*80)

train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_loss = float('inf')
early_stopping = EarlyStopping(patience=5, verbose=True)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)
    
    # 训练
    train_loss, train_labels, train_preds = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    train_metrics = calculate_all_metrics(train_labels, train_preds, verbose=False)
    train_losses.append(train_loss)
    train_accs.append(train_metrics["Accuracy"])
    
    # 验证
    val_loss, val_labels, val_preds = validate(model, val_loader, criterion, DEVICE)
    val_metrics = calculate_all_metrics(val_labels, val_preds, verbose=False)
    val_losses.append(val_loss)
    val_accs.append(val_metrics["Accuracy"])
    
    # 学习率调整
    scheduler.step(val_loss)
    
    # 打印结果
    print(f"训练 Loss: {train_loss:.4f} | Acc: {train_metrics['Accuracy']:.4f} | F1: {train_metrics.get('F1', 0):.4f}")
    print(f"验证 Loss: {val_loss:.4f} | Acc: {val_metrics['Accuracy']:.4f} | F1: {val_metrics.get('F1', 0):.4f}")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print(f"✅ 最佳模型已更新（验证Loss: {best_val_loss:.4f}）")
    
    # 检查早停法
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("\n" + "="*80)
        print(f"⏹️  早停法触发！在第 {epoch+1} 轮停止训练")
        print(f"最佳验证Loss: {best_val_loss:.4f}")
        print("="*80)
        break

# ===================== 绘制训练曲线 =====================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training & Validation Loss ({MODEL_NAME.upper()})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Training & Validation Accuracy ({MODEL_NAME.upper()})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_CURVE_PATH, dpi=150)
print(f"\n训练曲线已保存到：{SAVE_CURVE_PATH}")
plt.close()

print("\n🎉 模型训练全部完成！")
print(f"  最佳模型已保存到：{SAVE_MODEL_PATH}")