import torch
import torch.nn as nn

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

class CNNTrajectoryMatcher_LSTM(nn.Module):
    def __init__(self, input_dim=4, cnn_hidden=64, rnn_hidden=128):
        super().__init__()
        self.model_name = "CNN-LSTM"
        self.cnn_extractor = CNNFeatureExtractor(input_dim=input_dim, hidden_dim=cnn_hidden)
        self.lstm = nn.LSTM(cnn_hidden, rnn_hidden, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(rnn_hidden, 1),
        )
    
    def forward(self, traj_a, traj_b):
        feat_a = self.cnn_extractor(traj_a)
        feat_b = self.cnn_extractor(traj_b)
        
        _, (h_a, _) = self.lstm(feat_a)
        _, (h_b, _) = self.lstm(feat_b)
        
        global_a = h_a[-1, :, :]
        global_b = h_b[-1, :, :]
        
        combined = torch.cat([global_a, global_b], dim=1)
        return self.classifier(combined).squeeze(-1)