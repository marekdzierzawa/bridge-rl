"""Networks: card play (card value with a belief head) and the regret and policy networks of Deep CFR."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge.env.bridge_env import NUM_ACTIONS, OBS_SIZE
from bridge.env.play_features import CARD_FEAT_DIM


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class PlayQNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_SIZE, card_dim=CARD_FEAT_DIM, h=512):
        super().__init__()
        self.proj = nn.Linear(obs_dim, h)
        self.res1 = ResBlock(h)
        self.res2 = ResBlock(h)
        self.belief_head = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Linear(256, 156))
        self.card_enc = nn.Sequential(
            nn.Linear(card_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(h + 128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _trunk(self, obs):
        return self.res2(self.res1(F.relu(self.proj(obs))))

    def forward(self, obs, card):
        z = self._trunk(obs)
        return (self.q_head(torch.cat([z, self.card_enc(card)], -1)).squeeze(-1),
                self.belief_head(z))

    @torch.no_grad()
    def q_all(self, obs_1d, card_mat):
        z = self._trunk(obs_1d.unsqueeze(0)).expand(card_mat.shape[0], -1)
        return self.q_head(torch.cat([z, self.card_enc(card_mat)], -1)).squeeze(-1)


class HierAdvantageNetwork(nn.Module):
    def __init__(self, input_dim=OBS_SIZE, num_actions=NUM_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.head_cat = nn.Linear(256, 4)
        self.head_strain = nn.Linear(256, 5)
        self.head_act = nn.Linear(256, num_actions)

    def forward(self, state):
        h = self.trunk(state)
        return self.head_cat(h), self.head_strain(h), self.head_act(h)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=OBS_SIZE, num_actions=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, state):
        return self.net(state)
