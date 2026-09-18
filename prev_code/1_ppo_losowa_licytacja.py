import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from bridge_env import BridgeEnv, NUM_ACTIONS, OBS_SIZE
from models import PlayNetwork
from config import DEVICE


class PlayCriticNetwork(nn.Module):
    def __init__(self, input_dim=OBS_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state):
        return self.net(state)

class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.masks = []
        self.players = []
        self.rewards = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.masks.clear()
        self.players.clear()
        self.rewards.clear()

# ============================================================================
# LOGIKA TRENINGU PPO
# ============================================================================

def generate_random_auction(env):
    while env.phase == 0 and not env.terminated:
        mask = env.get_action_mask()
        legal_actions = np.flatnonzero(mask)
        action = np.random.choice(legal_actions)
        env.step(action)

def train_play_ppo(epochs=500, episodes_per_epoch=128, ppo_epochs=4, clip_eps=0.2):
    print("\n" + "="*50)
    print("FAZA 0: Trening PPO (Umiejętność rozgrywki kart)")
    print("="*50)
    
    actor = PlayNetwork().to(DEVICE)
    critic = PlayCriticNetwork().to(DEVICE)
    
    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)
    
    buffer = PPOBuffer()
    pbar = tqdm(range(1, epochs + 1), desc="Trening PPO")
    
    for epoch in pbar:
        actor.eval()
        critic.eval()
        
        for _ in range(episodes_per_epoch):
            env = BridgeEnv()
            env.reset()
            generate_random_auction(env)
            
            if env.terminated: 
                continue
                
            episode_start_idx = len(buffer.states)
            
            while not env.terminated:
                player = env.current_player
                obs = env.observe(player)
                mask = env.get_action_mask()
                
                state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE)
                
                with torch.no_grad():
                    value = critic(state_t)
                    
                    probs = actor(state_t)
                    
                    valid_probs = probs * mask_t
                    valid_probs = valid_probs / (valid_probs.sum() + 1e-8)
                    
                    dist = Categorical(valid_probs)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                
                buffer.states.append(obs)
                buffer.actions.append(action.item())
                buffer.logprobs.append(logprob.item())
                buffer.values.append(value.item())
                buffer.masks.append(mask)
                buffer.players.append(player)
                
                env.step(action.item())
                
            for i in range(episode_start_idx, len(buffer.states)):
                p = buffer.players[i]
                buffer.rewards.append(env.rewards[p])

        if len(buffer.states) == 0:
            continue
            
        actor.train()
        critic.train()
        
        old_states = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=DEVICE)
        old_actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32, device=DEVICE)
        old_logprobs = torch.tensor(np.array(buffer.logprobs), dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(np.array(buffer.rewards), dtype=torch.float32, device=DEVICE)
        old_values = torch.tensor(np.array(buffer.values), dtype=torch.float32, device=DEVICE).squeeze()
        masks = torch.tensor(np.array(buffer.masks), dtype=torch.float32, device=DEVICE)
        
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(ppo_epochs):
            values = critic(old_states).squeeze()
            probs = actor(old_states)
            
            valid_probs = probs * masks
            valid_probs = valid_probs / (valid_probs.sum(dim=-1, keepdim=True) + 1e-8)
            dist = Categorical(valid_probs)
            
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            critic_loss = nn.MSELoss()(values, rewards)
            
            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()
            
            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()
            
        pbar.set_postfix({"Act Loss": f"{actor_loss.item():.4f}", "Crit Loss": f"{critic_loss.item():.4f}"})
        buffer.clear()
        
    os.makedirs("models", exist_ok=True)
    torch.save(actor.state_dict(), "models/play_ppo_policy.pth")
    print("\n-> Zapisano wytrenowaną sieć PlayNetwork (PPO) w 'models/play_ppo_policy.pth'.")

if __name__ == "__main__":
    train_play_ppo(epochs=2000, episodes_per_epoch=128)