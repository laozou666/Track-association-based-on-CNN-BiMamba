"""CNN + BiMamba track association model (demo inference only)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 64
D_STATE = 32
D_CONV = 4
EXPAND = 2


class LinearEmbedder(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return F.gelu(self.norm(self.proj(x)))


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        res = self.proj(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = F.gelu(self.bn2(self.conv2(out)) + res)
        return out.permute(0, 2, 1)


class BiMambaSeqLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        from mamba_ssm import Mamba

        self.norm_fwd = nn.LayerNorm(d_model)
        self.norm_bwd = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model, d_state, d_conv, expand)
        self.mamba_bwd = Mamba(d_model, d_state, d_conv, expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        fwd = self.mamba_fwd(self.norm_fwd(x))
        bwd = torch.flip(
            self.mamba_bwd(self.norm_bwd(torch.flip(x, dims=[1]))),
            dims=[1],
        )
        return self.drop(x + (fwd + bwd) * 0.5)


class AblationCNNMamba(nn.Module):
    """Grid-search best config: n_cnn=4, n_mamba=1, dropout=0.4."""

    def __init__(
        self,
        input_dim=4,
        d_model=D_MODEL,
        n_cnn=4,
        n_mamba=1,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        dropout=0.4,
    ):
        super().__init__()

        if n_cnn == 0:
            self.embedder = LinearEmbedder(input_dim, d_model)
        else:
            blocks = [CNNBlock(input_dim, d_model, dropout * 0.5)]
            for _ in range(n_cnn - 1):
                blocks.append(CNNBlock(d_model, d_model, dropout * 0.5))
            self.embedder = nn.Sequential(*blocks)

        self.mamba_layers = nn.ModuleList([
            BiMambaSeqLayer(d_model, d_state, d_conv, expand, dropout * 0.5)
            for _ in range(n_mamba)
        ])

        enc_dim = d_model * 2
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

    def _encode(self, x):
        h = self.embedder(x)
        for layer in self.mamba_layers:
            h = layer(h)
        return torch.cat([h.mean(1), h.max(1).values], dim=-1)

    def forward(self, traj_a, traj_b):
        g_a = self._encode(traj_a)
        g_b = self._encode(traj_b)
        comb = torch.cat([g_a, g_b, g_a - g_b, torch.abs(g_a - g_b)], dim=-1)
        return self.classifier(comb).squeeze(-1)
