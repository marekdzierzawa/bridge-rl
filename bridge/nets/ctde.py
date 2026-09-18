"""Actor-critic bidding networks: the actor sees its own hand and the auction, the critic sees all four hands."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge.env.bridge_env import NUM_ACTIONS, OBS_SIZE

N_HIDDEN_CARDS = 156
N_PARTNER_FEATURES = 5


class _ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class _Trunk(nn.Module):
    def __init__(self, in_dim, h=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, h)
        self.res1 = _ResBlock(h, dropout)
        self.res2 = _ResBlock(h, dropout)

    def forward(self, x):
        return self.res2(self.res1(F.relu(self.proj(x))))


class BidActor(nn.Module):
    def __init__(self, obs_dim=OBS_SIZE, h=512):
        super().__init__()
        self.tulow = _Trunk(obs_dim, h)
        self.glowa_polityki = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Linear(256, NUM_ACTIONS))
        self.glowa_kart = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Linear(256, N_HIDDEN_CARDS))
        self.glowa_partnera = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Linear(128, N_PARTNER_FEATURES))

    def forward(self, obs):
        h = self.tulow(obs)
        return self.glowa_polityki(h), self.glowa_kart(h), self.glowa_partnera(h)

    def logits(self, obs, mask):
        return self.glowa_polityki(self.tulow(obs)).masked_fill(mask == 0, -1e9)


class BidCritic(nn.Module):
    def __init__(self, obs_dim=OBS_SIZE, h=512):
        super().__init__()
        self.tulow = _Trunk(obs_dim + N_HIDDEN_CARDS, h)
        self.glowa = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, obs, hidden):
        return self.glowa(self.tulow(torch.cat([obs, hidden], dim=-1))).squeeze(-1)
