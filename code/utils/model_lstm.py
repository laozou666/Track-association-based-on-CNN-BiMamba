import torch
import torch.nn as nn
class TrajectoryMatcher_LSTM(nn.Module):
    def __init__(self, input_dim=4, rnn_hidden=128):
        super().__init__()
        self.model_name = "LSTM"
        self.lstm = nn.LSTM(input_dim, rnn_hidden, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),  # 两条航迹拼接
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden, 1),
        )
    
    def forward(self, traj_a, traj_b):
        _, (h_a, _) = self.lstm(traj_a)
        _, (h_b, _) = self.lstm(traj_b)
        
        # 单向LSTM，只取最后一层的隐藏状态
        global_a = h_a[-1, :, :]  # (Batch, Hidden)
        global_b = h_b[-1, :, :]  # (Batch, Hidden)
        
        combined = torch.cat([global_a, global_b], dim=1)  # (Batch, Hidden*2)
        return self.classifier(combined).squeeze(-1)
