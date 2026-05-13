import torch
import torch.nn as nn

class TrajectoryMatcher_BiLSTM(nn.Module):
    def __init__(self, input_dim=4, rnn_hidden=128):
        super().__init__()
        self.model_name = "BiLSTM (No CNN)"
        self.bilstm = nn.LSTM(input_dim, rnn_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 4, rnn_hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden * 2, 1),
        )
    
    def forward(self, traj_a, traj_b):
        _, (h_a, _) = self.bilstm(traj_a)
        _, (h_b, _) = self.bilstm(traj_b)
        
        global_a = torch.cat([h_a[-2, :, :], h_a[-1, :, :]], dim=1)
        global_b = torch.cat([h_b[-2, :, :], h_b[-1, :, :]], dim=1)
        
        combined = torch.cat([global_a, global_b], dim=1)
        return self.classifier(combined).squeeze(-1)