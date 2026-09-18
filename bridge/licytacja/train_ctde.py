"""Phase 1, actor-critic variant (CTDE): PPO with a centralised critic and heads predicting the partner's hand."""
import argparse
import concurrent.futures
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from bridge.env.bridge_env import AUCTION, NUM_ACTIONS, NUM_BIDS, PASS, AuctionState, BridgeEnv, card_suit
from bridge.env.config import DEVICE
from bridge.licytacja.deal_value import DealValue, auction_reward
from bridge.nets.ctde import N_HIDDEN_CARDS, BidActor, BidCritic
from bridge.ocena.historia import append_row, ctde_row
from bridge.ocena.wzorce import STRAIN_CODES, repertoire
from bridge.ocena.zbior_walidacyjny import load_dd_tables

AUCTIONS_PER_DEAL = 8
MAX_AUCTION_LEN = 60

PPO_EPOCHS = 4
PPO_CLIP = 0.2
GAE_LAMBDA = 0.95
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
BATCH = 4096
GRAD_CLIP = 1.0

W_CARDS = 0.5
W_PARTNER = 0.5
ENTROPY_START = 0.06
ENTROPY_END = 0.010
ENTROPY_DECAY_EPOCHS = 2000

PASS_BIAS = 1.5

CKPT_DIR = "checkpoints/ctde"
BEST_ACTOR_PATH = f"{CKPT_DIR}/best_actor.pth"
BEST_CRITIC_PATH = f"{CKPT_DIR}/best_critic.pth"
HISTORY_PATH = f"{CKPT_DIR}/historia.csv"
CKPT_EVERY = 250

_HCP = {12: 4, 11: 3, 10: 2, 9: 1}
_ACTOR = None


def _hidden_hands(env, player):
    out = np.zeros(N_HIDDEN_CARDS, dtype=np.float32)
    for i, offset in enumerate((1, 2, 3)):
        for c in env.hands[(player + offset) % 4]:
            out[i * 52 + c] = 1.0
    return out


def _suit_lengths(hand):
    lengths = np.zeros(4, dtype=np.float32)
    for c in hand:
        lengths[card_suit(c)] += 1
    return lengths


def _partner_features(env, player):
    hand = env.hands[(player + 2) % 4]
    pc = sum(_HCP.get(c % 13, 0) for c in hand)
    return np.concatenate([[pc / 10.0], _suit_lengths(hand) / 13.0]).astype(np.float32)


def _worker_init():
    global _ACTOR
    torch.set_num_threads(1)
    seed = (os.getpid() * int(time.time() * 1000)) % (2 ** 31 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _ACTOR = BidActor()
    _ACTOR.eval()


def collect_auctions(n_deals, actor_sd):
    _ACTOR.load_state_dict(actor_sd)
    rng = np.random.default_rng()
    auctions = []
    stats = np.zeros(6)

    for _ in range(n_deals):
        env = BridgeEnv()
        env.reset()
        dv = DealValue(env)
        for _ in range(AUCTIONS_PER_DEAL):
            au = AuctionState(env.dealer)
            steps = []
            for _ in range(MAX_AUCTION_LEN):
                if au.is_over():
                    break
                p = au.current_player()
                env.auction = au
                obs = env.observe(p).astype(np.float32)
                mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
                mask[au.legal_calls()] = 1
                with torch.no_grad():
                    lg = _ACTOR.logits(torch.from_numpy(obs), torch.from_numpy(mask.astype(np.float32)))
                    logp = F.log_softmax(lg, dim=-1)
                    pr = logp.exp().numpy().astype(np.float64)
                a = int(rng.choice(NUM_ACTIONS, p=pr / pr.sum()))
                steps.append({
                    "obs": obs,
                    "hidden": _hidden_hands(env, p),
                    "partner": _partner_features(env, p),
                    "mask": mask,
                    "action": a,
                    "logp": float(logp[a]),
                    "side": p % 2,
                })
                au.apply(a)

            if not steps:
                continue
            auctions.append({"steps": steps, "z_ns": float(auction_reward(dv, au, 0))})
            level = 0 if au.last_bid is None else au.last_bid // 5 + 1
            stats[0] += len(au.calls)
            stats[1] += level
            stats[2] += 1
            stats[3] += level >= 6
            stats[4] += au.last_bid is None
            stats[5] += sum(1 for (_, a) in au.calls if a == PASS)

    return auctions, stats


def compute_advantages(auctions, critic, lam=GAE_LAMBDA):
    with torch.no_grad():
        for au in auctions:
            obs = torch.from_numpy(np.stack([k["obs"] for k in au["steps"]]))
            hidden = torch.from_numpy(np.stack([k["hidden"] for k in au["steps"]]))
            v = critic(obs.to(DEVICE), hidden.to(DEVICE)).cpu().numpy()
            sign = np.array([1.0 if k["side"] == 0 else -1.0 for k in au["steps"]])
            v_ns = v * sign
            n = len(v_ns)
            adv_ns = np.zeros(n, dtype=np.float64)
            gae = 0.0
            for t in range(n - 1, -1, -1):
                next_v = au["z_ns"] if t == n - 1 else v_ns[t + 1]
                gae = next_v - v_ns[t] + lam * gae
                adv_ns[t] = gae
            for t, k in enumerate(au["steps"]):
                k["adv"] = float(adv_ns[t] * sign[t])
                k["v_target"] = float((adv_ns[t] + v_ns[t]) * sign[t])
                k["_v"] = float(v[t])

    targets = np.array([k["v_target"] for au in auctions for k in au["steps"]])
    values = np.array([k["_v"] for au in auctions for k in au["steps"]])
    var = targets.var()
    return float(1.0 - ((targets - values).var() / var)) if var > 1e-12 else 0.0


def _to_tensors(steps):
    def stack(key):
        return torch.tensor(np.stack([x[key] for x in steps]), dtype=torch.float32, device=DEVICE)

    def as_list(key, dtype=torch.float32):
        return torch.tensor([x[key] for x in steps], dtype=dtype, device=DEVICE)

    return {
        "obs": stack("obs"),
        "hidden": stack("hidden"),
        "partner": stack("partner"),
        "mask": stack("mask"),
        "action": as_list("action", torch.long),
        "logp": as_list("logp"),
        "adv": as_list("adv"),
        "v_target": as_list("v_target"),
    }


def ppo_update(steps, actor, critic, opt_a, opt_k, beta):
    T = _to_tensors(steps)
    n = T["obs"].shape[0]
    adv = T["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    sums = np.zeros(5)
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            lg, card_logits, partner_out = actor(T["obs"][idx])
            lg = lg.masked_fill(T["mask"][idx] == 0, -1e9)
            logp = F.log_softmax(lg, dim=-1)
            lp = logp.gather(1, T["action"][idx].unsqueeze(1)).squeeze(1)

            ratio = (lp - T["logp"][idx]).exp()
            loss_policy = -torch.min(ratio * adv[idx],
                                     torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv[idx]).mean()
            entropy = -(logp.exp() * logp).sum(1).mean()
            loss_cards = F.binary_cross_entropy_with_logits(card_logits, T["hidden"][idx])
            loss_partner = F.mse_loss(partner_out, T["partner"][idx])

            loss_actor = loss_policy + W_CARDS * loss_cards + W_PARTNER * loss_partner - beta * entropy
            opt_a.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
            opt_a.step()

            loss_value = F.mse_loss(critic(T["obs"][idx], T["hidden"][idx]), T["v_target"][idx])
            opt_k.zero_grad()
            loss_value.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
            opt_k.step()

            sums += [loss_policy.item(), loss_value.item(), loss_cards.item(),
                     loss_partner.item(), entropy.item()]
    return sums / max(PPO_EPOCHS * ((n + BATCH - 1) // BATCH), 1)


def validate(actor, tables, vul, par, seeds, temp=0.0):
    losses, levels, strains, entropies, contracts = [], [], [], [], []
    n_doubled = n_redoubled = n_passed = 0
    hcp_err, shape_err, majors = [], [], []

    for i, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        env = BridgeEnv(seed=sd)
        env.reset()
        k = 0
        while env.phase == AUCTION and not env.terminated and k < MAX_AUCTION_LEN:
            k += 1
            player = env.current_player
            partner = (player + 2) % 4
            obs = env.observe(player).astype(np.float32)
            mk = env.get_action_mask().astype(np.float32)
            with torch.no_grad():
                lg, _, partner_pred = actor(torch.from_numpy(obs))
                lg = lg.masked_fill(torch.from_numpy(mk) == 0, -1e9)
                logp = F.log_softmax(lg, dim=-1)
                entropies.append(float(-(logp.exp() * logp.clamp(min=-30)).sum()))
                pred_hcp = float(partner_pred[0]) * 10.0
                pred_len = partner_pred[1:5].numpy() * 13.0

            own_hcp = sum(_HCP.get(c % 13, 0) for c in env.hands[player])
            true_hcp = sum(_HCP.get(c % 13, 0) for c in env.hands[partner])
            partner_has_bid = any(pl == partner and a < NUM_BIDS for (pl, a) in env.auction.calls)
            hcp_err.append((abs(pred_hcp - true_hcp), abs((40 - own_hcp) / 3.0 - true_hcp),
                            abs(10.0 - true_hcp), partner_has_bid))

            if partner_has_bid:
                partner_len = _suit_lengths(env.hands[partner]).astype(np.float64)
                own_len = _suit_lengths(env.hands[player]).astype(np.float64)
                shape_err.append((np.abs(pred_len - partner_len), np.abs((13 - own_len) / 3.0 - partner_len),
                                  np.abs(3.25 - partner_len)))
                if partner_len[3] != partner_len[2]:
                    majors.append(((pred_len[3] > pred_len[2]) == (partner_len[3] > partner_len[2]),
                                   ((13 - own_len[3]) > (13 - own_len[2])) == (partner_len[3] > partner_len[2])))

            if temp <= 0:
                a = int(torch.argmax(lg).item())
            else:
                q = torch.softmax(lg / temp, dim=-1).numpy().astype(np.float64)
                a = int(rng.choice(NUM_ACTIONS, p=q / q.sum()))
            env.step(a)

        pv = float(par[i])
        if env.terminated or env.phase == AUCTION:
            n_passed += bool(env.terminated)
            losses.append(abs(pv))
            continue
        levels.append(env.level)
        strains.append(env.strain)
        contracts.append((env.level, env.strain))
        n_doubled += bool(env.contract_doubled)
        n_redoubled += bool(env.contract_redoubled)
        dv = DealValue.from_tricks(tables[i], tuple(vul[i]))
        sc_ns = dv.score_ns(env.level, env.strain, env.contract_doubled,
                            env.contract_redoubled, env.declarer)
        losses.append(abs(pv - sc_ns))

    st = np.array(losses)
    nk = max(len(levels), 1)
    A = np.array([(a, b, c) for a, b, c, _ in hcp_err])
    partner_bid_mask = np.array([d for _, _, _, d in hcp_err], dtype=bool)
    return {
        "loss": st.mean(),
        "se": st.std(ddof=1) / np.sqrt(len(st)),
        "pass_loss": float(np.abs(par).mean()),
        "pass": n_passed / len(seeds) * 100,
        "level": float(np.mean(levels)) if levels else 0.0,
        "n_contracts": len(levels),
        "level6plus": sum(1 for l in levels if l >= 6) / nk * 100,
        "level_hist": {l: sum(1 for x in levels if x == l) / nk * 100 for l in range(1, 8)},
        "strain_hist": {STRAIN_CODES[m]: sum(1 for x in strains if x == m) / nk * 100 for m in range(5)},
        "doubled": n_doubled / nk * 100,
        "redoubled": n_redoubled / nk * 100,
        "repertoire": repertoire(contracts),
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "hcp_net": float(A[:, 0].mean()),
        "hcp_hand": float(A[:, 1].mean()),
        "hcp_const": float(A[:, 2].mean()),
        "hcp_net_bid": float(A[partner_bid_mask, 0].mean()) if partner_bid_mask.any() else float("nan"),
        "hcp_hand_bid": float(A[partner_bid_mask, 1].mean()) if partner_bid_mask.any() else float("nan"),
        "n_bid": int(partner_bid_mask.sum()),
        "shape": _shape_stats(shape_err, majors),
    }


def _shape_stats(shape_err, majors):
    if len(shape_err) < 50:
        return None
    out = {"net": np.array([k[0] for k in shape_err]).mean(axis=0),
           "hand": np.array([k[1] for k in shape_err]).mean(axis=0),
           "const": np.array([k[2] for k in shape_err]).mean(axis=0),
           "n": len(shape_err)}
    if majors:
        st = np.array(majors, dtype=bool)
        out["majors_net"] = float(st[:, 0].mean()) * 100
        out["majors_hand"] = float(st[:, 1].mean()) * 100
        out["majors_n"] = int(len(st))
    return out


def train_phase_1_ctde(epochs=3000, deals_per_epoch=64, workers=8, val_every=100, val_deals=1000,
                       val_seed0=7_500_000):
    os.makedirs(CKPT_DIR, exist_ok=True)
    actor = BidActor().to(DEVICE)
    critic = BidCritic().to(DEVICE)
    with torch.no_grad():
        actor.glowa_polityki[-1].bias[PASS] += PASS_BIAS
    opt_a = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_k = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    val_seeds = list(range(val_seed0, val_seed0 + val_deals))
    tables, vul, par = load_dd_tables(val_seeds, workers)

    best_loss = 1e18
    per_worker = max(1, deals_per_epoch // workers)
    pbar = tqdm(range(1, epochs + 1), desc="CTDE")
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                                initializer=_worker_init) as ex:
        for epoch in pbar:
            beta = ENTROPY_END + (ENTROPY_START - ENTROPY_END) * max(0.0, 1.0 - epoch / ENTROPY_DECAY_EPOCHS)
            a_sd = {k: v.cpu() for k, v in actor.state_dict().items()}
            futs = [ex.submit(collect_auctions, per_worker, a_sd) for _ in range(workers)]
            auctions, stats = [], np.zeros(6)
            for f in concurrent.futures.as_completed(futs):
                a, s = f.result()
                auctions += a
                stats += s

            critic_r2 = compute_advantages(auctions, critic)
            steps = [k for au in auctions for k in au["steps"]]
            l_pol, l_val, l_cards, l_partner, ent = ppo_update(steps, actor, critic, opt_a, opt_k, beta)
            pbar.set_postfix(policy=f"{l_pol:.3f}", critic=f"{l_val:.3f}", H=f"{ent:.2f}",
                             level=f"{stats[1] / max(stats[2], 1):.2f}")

            if val_every and epoch % val_every == 0:
                a_cpu = BidActor()
                a_cpu.load_state_dict({k: v.cpu() for k, v in actor.state_dict().items()})
                a_cpu.eval()
                r = validate(a_cpu, tables, vul, par, val_seeds)
                r["critic_r2"] = critic_r2
                if r["loss"] < best_loss:
                    best_loss = r["loss"]
                    torch.save(actor.state_dict(), BEST_ACTOR_PATH)
                    torch.save(critic.state_dict(), BEST_CRITIC_PATH)
                append_row(HISTORY_PATH, ctde_row(epoch, r))
                tqdm.write(f"{epoch}: loss {r['loss']:.1f}, level {r['level']:.2f}, "
                           f"pass {r['pass']:.1f}%, critic R2 {critic_r2:+.2f}")

            if CKPT_EVERY and epoch % CKPT_EVERY == 0:
                torch.save(actor.state_dict(), f"{CKPT_DIR}/aktor_epoka_{epoch}.pth")

    torch.save(actor.state_dict(), f"{CKPT_DIR}/aktor_ostatni.pth")
    torch.save(critic.state_dict(), f"{CKPT_DIR}/krytyk_ostatni.pth")
    return actor, critic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--deals", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count() - 1)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-deals", type=int, default=1000)
    a = ap.parse_args()
    train_phase_1_ctde(epochs=a.epochs, deals_per_epoch=a.deals, workers=a.workers, val_every=a.val_every,
                       val_deals=a.val_deals)


if __name__ == "__main__":
    main()
