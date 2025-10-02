# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=11, padding=5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=9, padding=4), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, out_dim)
        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        z = self.net(x.unsqueeze(1)).squeeze(-1)  # (B, 128)
        z = self.fc(z)                            # (B, out_dim)
        p = self.proj(z)
        return z, p

def nt_xent(p_i, p_j, temperature=0.2):
    B = p_i.shape[0]
    z = torch.cat([F.normalize(p_i, dim=1), F.normalize(p_j, dim=1)], dim=0)
    sim = torch.matmul(z, z.T)
    mask = (~torch.eye(2*B, dtype=torch.bool, device=sim.device)).float()
    sim = sim / temperature
    exp_sim = torch.exp(sim) * mask
    pos_exp = torch.cat([
        torch.exp((F.normalize(p_i, dim=1) * F.normalize(p_j, dim=1)).sum(dim=1)/temperature),
        torch.exp((F.normalize(p_j, dim=1) * F.normalize(p_i, dim=1)).sum(dim=1)/temperature)
    ])
    denom = exp_sim.sum(dim=1)
    loss = -torch.log(pos_exp / denom).mean()
    return loss
