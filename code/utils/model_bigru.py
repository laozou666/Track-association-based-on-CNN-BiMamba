import torch    
import torch.nn as nn
class TrajectoryMatcher_BiGRU(nn.Module):
    def __init__(self, input_dim=4, rnn_hidden=128):
        super().__init__()
        self.model_name = "Bi-GRU"
        self.bigru = nn.GRU(input_dim, rnn_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 4, rnn_hidden * 2),  # 双向*2 + 两条航迹*2
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden * 2, 1),
        )
    
    def forward(self, traj_a, traj_b):
        # GRU只有隐藏状态h，没有细胞状态c
        _, h_a = self.bigru(traj_a)
        _, h_b = self.bigru(traj_b)
        
        # 双向GRU，拼接前向和后向的最后一层
        global_a = torch.cat([h_a[-2, :, :], h_a[-1, :, :]], dim=1)  # (Batch, Hidden*2)
        global_b = torch.cat([h_b[-2, :, :], h_b[-1, :, :]], dim=1)  # (Batch, Hidden*2)
        
        combined = torch.cat([global_a, global_b], dim=1)  # (Batch, Hidden*4)
        return self.classifier(combined).squeeze(-1)