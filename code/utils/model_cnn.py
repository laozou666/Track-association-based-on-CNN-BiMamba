import torch
import torch.nn as nn
class TrajectoryMatcher_CNN(nn.Module):
    def __init__(self, input_dim=4, cnn_hidden=64):
        super().__init__()
        self.model_name = "CNN"
        
        # 1D CNN 特征提取器
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden, cnn_hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_hidden * 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden * 2, cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(cnn_hidden * 2, cnn_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(cnn_hidden, 1),
        )
    
    def forward(self, traj_a, traj_b):
        # 输入形状: (Batch, SeqLen, 4) -> Conv1D需要 (Batch, 4, SeqLen)
        x_a = traj_a.permute(0, 2, 1)
        x_b = traj_b.permute(0, 2, 1)
        
        feat_a = self.conv_layers(x_a)  # (Batch, Hidden, SeqLen)
        feat_b = self.conv_layers(x_b)
        
        # 全局平均池化，得到全局特征
        global_a = torch.mean(feat_a, dim=2)  # (Batch, Hidden)
        global_b = torch.mean(feat_b, dim=2)  # (Batch, Hidden)
        
        combined = torch.cat([global_a, global_b], dim=1)  # (Batch, Hidden*2)
        return self.classifier(combined).squeeze(-1)
