import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures

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

    def __len__(self):
        return len(self.states)


def get_hcp(card_id: int) -> int:
    rank = card_id % 13
    if rank == 12: return 4
    elif rank == 11: return 3
    elif rank == 10: return 2
    elif rank == 9: return 1
    return 0

def generate_random_auction(env):
    ns_hcp = sum(get_hcp(c) for c in env.hands[0]) + sum(get_hcp(c) for c in env.hands[2])
    
    declarer_side = 0 if ns_hcp >= 20 else 1
    points = ns_hcp if declarer_side == 0 else (40 - ns_hcp)
    
    if points >= 33: target_level = np.random.randint(6, 8)     # Szlemik / Szlem
    elif points >= 25: target_level = np.random.randint(4, 6)   # Końcówka
    elif points >= 20: target_level = np.random.randint(1, 4)   # Częściówka
    else: target_level = 1
    
    PASS_ACTION = 35 
    
    while env.phase == 0 and not env.terminated:
        mask = env.get_action_mask()
        legal_actions = np.flatnonzero(mask)
        
        current_side = env.current_player % 2
        
        if current_side != declarer_side:
            action = PASS_ACTION if PASS_ACTION in legal_actions else legal_actions[0]
        else:
            last_bid = env.auction.last_bid
            current_level = (last_bid // 5 + 1) if last_bid is not None else 0
            
            if current_level >= target_level and PASS_ACTION in legal_actions:
                action = PASS_ACTION
            else:
                bids = [a for a in legal_actions if a < PASS_ACTION]
                if bids:
                    bids = bids[:4] 
                    action = np.random.choice(bids)
                else:
                    action = PASS_ACTION if PASS_ACTION in legal_actions else legal_actions[0]
                    
        env.step(action)



_WORKER_ACTOR = None
_WORKER_CRITIC = None


def _worker_init():
    global _WORKER_ACTOR, _WORKER_CRITIC
    torch.set_num_threads(1)
    seed = (os.getpid() * int(time.time() * 1000)) % (2**31 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _WORKER_ACTOR = PlayNetwork()
    _WORKER_CRITIC = PlayCriticNetwork()
    _WORKER_ACTOR.eval()
    _WORKER_CRITIC.eval()


def _worker_set_weights(actor_sd, critic_sd):

    _WORKER_ACTOR.load_state_dict(actor_sd)
    _WORKER_CRITIC.load_state_dict(critic_sd)


def worker_collect(episodes, actor_sd, critic_sd):

    _worker_set_weights(actor_sd, critic_sd)
    local_actor = _WORKER_ACTOR
    local_critic = _WORKER_CRITIC

    local_buf = PPOBuffer()
    local_stats = {i: 0 for i in range(1, 8)}
    local_stats["Passed_Out"] = 0
    local_tricks = {i: [] for i in range(1, 8)}

    for _ in range(episodes):
        env = BridgeEnv()
        env.reset()
        generate_random_auction(env)

        if env.terminated:
            local_stats["Passed_Out"] += 1
            continue

        local_stats[env.level] += 1
        episode_start_idx = len(local_buf.states)

        while not env.terminated:
            player = env.current_player
            obs = env.observe(player)
            mask = env.get_action_mask()

            state_t = torch.tensor(obs, dtype=torch.float32)
            mask_t = torch.tensor(mask, dtype=torch.float32)

            with torch.no_grad():
                value = local_critic(state_t)
                logits = local_actor(state_t)
                logits = logits.masked_fill(mask_t == 0, -1e9)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                logprob = dist.log_prob(action)

            local_buf.states.append(obs)
            local_buf.actions.append(action.item())
            local_buf.logprobs.append(logprob.item())
            local_buf.values.append(value.item())
            local_buf.masks.append(mask)
            local_buf.players.append(player)

            env.step(action.item())

        if env.level is not None:
            local_tricks[env.level].append(env.info.get("tricks_taken", 0))

        for i in range(episode_start_idx, len(local_buf.states)):
            p = local_buf.players[i]
            side = p % 2
            tricks_taken = env.tricks_won[side]

            reward = (tricks_taken - 6.5) / 6.5

            local_buf.rewards.append(reward)

    return local_buf, local_stats, local_tricks


def ppo_update(actor, critic, opt_actor, opt_critic, buffer,
                ppo_epochs=8, clip_eps=0.2, minibatch_size=512,
                max_grad_norm=0.5, target_kl=0.02):
    old_states = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=DEVICE)
    old_actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32, device=DEVICE)
    old_logprobs = torch.tensor(np.array(buffer.logprobs), dtype=torch.float32, device=DEVICE)
    rewards = torch.tensor(np.array(buffer.rewards), dtype=torch.float32, device=DEVICE)
    old_values = torch.tensor(np.array(buffer.values), dtype=torch.float32, device=DEVICE).squeeze(-1)
    masks = torch.tensor(np.array(buffer.masks), dtype=torch.float32, device=DEVICE)

    advantages = rewards - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = old_states.shape[0]
    minibatch_size = min(minibatch_size, n)
    last_actor_loss = None
    last_critic_loss = None

    for epoch_i in range(ppo_epochs):
        perm = torch.randperm(n, device=DEVICE)
        epoch_kl = 0.0
        n_batches = 0

        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]

            b_states = old_states[idx]
            b_actions = old_actions[idx]
            b_old_logprobs = old_logprobs[idx]
            b_rewards = rewards[idx]
            b_advantages = advantages[idx]
            b_masks = masks[idx]

            values = critic(b_states).squeeze(-1)
            logits = actor(b_states)

            logits = logits.masked_fill(b_masks == 0, -1e9)
            probs = torch.softmax(logits, dim=-1)
            
            dist = Categorical(probs)
            new_logprobs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            log_ratio = new_logprobs - b_old_logprobs
            ratios = torch.exp(log_ratio)

            approx_kl = ((ratios - 1) - log_ratio).mean().item()
            epoch_kl += approx_kl
            n_batches += 1

            surr1 = ratios * b_advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * b_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            critic_loss = nn.MSELoss()(values, b_rewards)

            opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt_actor.step()

            opt_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            opt_critic.step()

            last_actor_loss = actor_loss.item()
            last_critic_loss = critic_loss.item()

        # Early stopping
        if n_batches > 0 and (epoch_kl / n_batches) > target_kl:
            break

    return last_actor_loss, last_critic_loss



def train_play_ppo(epochs=4000, episodes_per_epoch=1024, ppo_epochs=8, clip_eps=0.2,
                    num_workers=24, minibatch_size=512, max_grad_norm=0.5, target_kl=0.02):
    print("\n" + "=" * 50)
    print("FAZA 0: Trening PPO")
    print("=" * 50)

    actor = PlayNetwork().to(DEVICE)
    critic = PlayCriticNetwork().to(DEVICE)

    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    buffer = PPOBuffer()

    contract_stats = {i: 0 for i in range(1, 8)}
    contract_stats["Passed_Out"] = 0
    history_tricks = {i: [] for i in range(1, 8)}

    episodes_per_worker = episodes_per_epoch // num_workers

    pbar = tqdm(range(1, epochs + 1), desc="Trening PPO", dynamic_ncols=True)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, initializer=_worker_init
    ) as executor:

        for epoch in pbar:
            actor_sd = {k: v.cpu() for k, v in actor.state_dict().items()}
            critic_sd = {k: v.cpu() for k, v in critic.state_dict().items()}

            epoch_tricks = {i: [] for i in range(1, 8)}

            futures = [
                executor.submit(worker_collect, episodes_per_worker, actor_sd, critic_sd)
                for _ in range(num_workers)
            ]

            for future in concurrent.futures.as_completed(futures):
                local_buf, local_stats, local_tricks = future.result()

                buffer.states.extend(local_buf.states)
                buffer.actions.extend(local_buf.actions)
                buffer.logprobs.extend(local_buf.logprobs)
                buffer.values.extend(local_buf.values)
                buffer.masks.extend(local_buf.masks)
                buffer.rewards.extend(local_buf.rewards)

                for k in contract_stats:
                    contract_stats[k] += local_stats[k]

                for lvl in range(1, 8):
                    epoch_tricks[lvl].extend(local_tricks[lvl])

            for lvl in range(1, 8):
                if epoch_tricks[lvl]:
                    history_tricks[lvl].append(np.mean(epoch_tricks[lvl]))
                else:
                    prev = history_tricks[lvl][-1] if history_tricks[lvl] else 0
                    history_tricks[lvl].append(prev)

            if len(buffer) == 0:
                continue

            actor.train()
            critic.train()

            actor_loss, critic_loss = ppo_update(
                actor, critic, opt_actor, opt_critic, buffer,
                ppo_epochs=ppo_epochs, clip_eps=clip_eps,
                minibatch_size=minibatch_size, max_grad_norm=max_grad_norm,
                target_kl=target_kl,
            )

            if actor_loss is not None:
                pbar.set_postfix({"Act Loss": f"{actor_loss:.4f}", "Crit Loss": f"{critic_loss:.4f}"})
            buffer.clear()

    # RAPORT
    total_played = sum(contract_stats.values())
    print("\n" + "=" * 50)
    print(f"RAPORT Z TRENINGU PPO (Rozegrane rozdania: {total_played})")
    print("=" * 50)
    if total_played > 0:
        po_count = contract_stats["Passed_Out"]
        print(f" 4 Pasy (Passed Out): {po_count} ({po_count/total_played*100:.2f}%)")
        print("-" * 50)
        for level in range(1, 8):
            count = contract_stats[level]
            print(f" Kontrakty na poziomie {level}: {count} ({count/total_played*100:.2f}%)")
    print("=" * 50)

    # WYKRES
    plt.figure(figsize=(12, 8))
    for lvl in range(1, 8):
        plt.plot(history_tricks[lvl], label=f'Poziom {lvl}', alpha=0.8, linewidth=1.5)

    plt.title('Postęp uczenia PPO: Średnia liczba zebranych lew na dany poziom kontraktu', fontsize=14)
    plt.xlabel('Epoka', fontsize=12)
    plt.ylabel('Średnia liczba lew (na 13 możliwych)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    os.makedirs("models", exist_ok=True)
    plot_path = "models/ppo_tricks_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"-> Wygenerowano wykres treningowy: {plot_path}")

    torch.save(actor.state_dict(), "models/play_ppo_fixed_policy.pth")
    print("-> Zapisano wytrenowaną sieć PlayNetwork (PPO) w 'models/play_ppo_fixed_policy.pth'.")


if __name__ == "__main__":
    train_play_ppo()