import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class LightCNNEmbedder(nn.Module):
    """
    轻量级 CNN：仅做局部特征嵌入，不破坏时序结构。
    只用 1 层残差块（4 -> d_model），保留 Mamba 的学习空间。
    """
    def __init__(self, input_dim=4, d_model=64, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(d_model)
        self.proj  = nn.Conv1d(input_dim, d_model, kernel_size=1)  # residual 投影
        self.drop  = nn.Dropout(dropout)
        self.output_dim = d_model

    def forward(self, x):
        # x: (B, T, C)  ->  conv 需要 (B, C, T)
        x = x.permute(0, 2, 1)
        res = self.proj(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + res)
        return out.permute(0, 2, 1)   # (B, T, d_model)


class BiMambaBlock(nn.Module):
    """
    双向 Mamba 编码器，输出整条序列的特征。
    修复点：
      - 聚合改为 mean-pool + max-pool（不再只取最后一步），
        规避 edge-padding 重复值对最终状态的污染。
      - d_state 加大到 32，增强长程记忆。
    """
    def __init__(self, d_model=64, d_state=32, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm_fwd = nn.LayerNorm(d_model)
        self.norm_bwd = nn.LayerNorm(d_model)

        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)
        # mean-pool + max-pool 各 d_model，正向 + 反向 -> 4 * d_model
        self.output_dim = d_model * 4

    def _pool(self, out):
        """mean-pool 与 max-pool 拼接，比单取最后步更鲁棒。"""
        mean_f = out.mean(dim=1)                     # (B, d_model)
        max_f  = out.max(dim=1).values               # (B, d_model)
        return torch.cat([mean_f, max_f], dim=-1)    # (B, 2*d_model)

    def forward(self, x):
        # 正向
        fwd = self.mamba_fwd(self.norm_fwd(x))
        feat_fwd = self._pool(fwd)                   # (B, 2*d_model)

        # 反向（翻转时间轴）
        x_rev = torch.flip(x, dims=[1])
        bwd = self.mamba_bwd(self.norm_bwd(x_rev))
        feat_bwd = self._pool(bwd)                   # (B, 2*d_model)

        out = torch.cat([feat_fwd, feat_bwd], dim=-1)  # (B, 4*d_model)
        return self.drop(out)


class CNNTrajectoryMatcher(nn.Module):
    """
    改进版 CNN + 双向 Mamba 航迹关联模型。

    关键改动：
    1. 轻量 CNN（1 残差块），避免过度压缩时序信息。
    2. mean-pool + max-pool 替代 last-step，消除 edge-padding 干扰。
    3. 跨航迹交互特征 [a, b, a-b, |a-b|]，让模型直接学差异。
    4. d_state=32，增强 Mamba 长程记忆能力。
    """
    def __init__(self,
                 input_dim=4,
                 d_model=64,
                 mamba_d_state=32,
                 mamba_d_conv=4,
                 mamba_expand=2,
                 dropout=0.2):
        super().__init__()
        self.model_name = "CNN1D_BiMamba_v2"

        # 1. 轻量 CNN 嵌入层
        self.cnn = LightCNNEmbedder(input_dim=input_dim, d_model=d_model,
                                    dropout=dropout * 0.5)

        # 2. 双向 Mamba 编码器
        self.bimamba = BiMambaBlock(d_model=d_model, d_state=mamba_d_state,
                                    d_conv=mamba_d_conv, expand=mamba_expand,
                                    dropout=dropout * 0.5)

        enc_dim = self.bimamba.output_dim   # 4 * d_model = 256

        # 3. 分类头：输入 = [a, b, a-b, |a-b|] = 4 * enc_dim
        fusion_dim = enc_dim * 4
        hidden = enc_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def _encode(self, traj):
        feat = self.cnn(traj)          # (B, T, d_model)
        return self.bimamba(feat)      # (B, enc_dim)

    def forward(self, traj_a, traj_b):
        g_a = self._encode(traj_a)    # (B, enc_dim)
        g_b = self._encode(traj_b)

        # 跨航迹交互：拼接原始特征 + 差值 + 绝对差值
        combined = torch.cat([g_a, g_b, g_a - g_b, torch.abs(g_a - g_b)], dim=-1)
        return self.classifier(combined).squeeze(-1)


# ==========================================
# 快速验证
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    batch_size, seq_len, input_dim = 8, 350, 4
    traj_a = torch.randn(batch_size, seq_len, input_dim).to(device)
    traj_b = torch.randn(batch_size, seq_len, input_dim).to(device)

    model = CNNTrajectoryMatcher(input_dim=4, d_model=64, mamba_d_state=32, dropout=0.2).to(device)
    output = model(traj_a, traj_b)

    print("✅ CNN+Mamba v2 测试成功！")
    print(f"  输入: traj_a={traj_a.shape}, traj_b={traj_b.shape}")
    print(f"  输出: {output.shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")