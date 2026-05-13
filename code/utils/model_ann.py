import torch
import torch.nn as nn

# =====================================================================
# 1. ANN (MLP) - 纯全连接，无视时序
# =====================================================================
class TrajectoryMatcher_ANN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super().__init__()
        self.model_name = "ANN"
        # 简单的MLP，先对每条航迹做全局均值池化，再拼接分类
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 两条航迹的均值拼接
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, traj_a, traj_b):
        # 对每条航迹取时间维度的均值（无视时序，只看整体统计）
        global_a = torch.mean(traj_a, dim=1)  # (Batch, 4)
        global_b = torch.mean(traj_b, dim=1)  # (Batch, 4)
        
        combined = torch.cat([global_a, global_b], dim=1)  # (Batch, 8)
        return self.classifier(combined).squeeze(-1)
