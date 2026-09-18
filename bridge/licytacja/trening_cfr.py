"""Phase 1, Deep CFR variant: bidding learned with hierarchical regret matching and solver rewards."""
import argparse
import concurrent.futures
import copy
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from bridge.env.bridge_env import AUCTION, NUM_ACTIONS, NUM_CARDS, PASS, AuctionState, BridgeEnv
from bridge.env.config import DEVICE
from bridge.env.utils import ReservoirBuffer, pack_obs, unpack_obs_batch
from bridge.licytacja.bid_reference import reference_policy
from bridge.licytacja.deal_value import DealValue, auction_reward, auction_reward_truncated
from bridge.nets.hier_rm import hier_masks_torch, hier_regrets, hier_sigma, hier_targets_torch
from bridge.nets.models import HierAdvantageNetwork, PolicyNetwork
from bridge.ocena.historia import append_row, cfr_row
from bridge.ocena.wzorce import repertoire
from bridge.ocena.zbior_walidacyjny import load_dd_tables

REDEALS = 3
TRAVERSALS_PER_DEAL = 2

ANCHOR_LAMBDA0 = 0.5
ANCHOR_DECAY_EPOCHS = 1500

ADV_PRETRAIN_SCALE = 0.5
ADV_PRETRAIN_EPOCHS = 15

MAX_CALLS = 80
TRUNC_P = 0.25
TRUNC_MIN_CALLS = 8
MAX_ROLLOUT_DEPTH = 24

CKPT_DIR = "checkpoints/phase1b"
BEST_POLICY_PATH = f"{CKPT_DIR}/best_policy.pth"
BEST_ADV_PATH = f"{CKPT_DIR}/best_advantage.pth"
HISTORY_PATH = f"{CKPT_DIR}/historia.csv"

_ADV = None


def anchor_lambda(t):
    if ANCHOR_LAMBDA0 <= 0.0 or t >= ANCHOR_DECAY_EPOCHS:
        return 0.0
    return ANCHOR_LAMBDA0 * (1.0 - t / ANCHOR_DECAY_EPOCHS)


def _worker_init():
    global _ADV
    torch.set_num_threads(1)
    seed = (os.getpid() * int(time.time() * 1000)) % (2 ** 31 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _ADV = HierAdvantageNetwork()
    _ADV.eval()


def _redeal(env, keep_seat, rng):
    keep = set(env.hands[keep_seat])
    rest = np.array([c for c in range(NUM_CARDS) if c not in keep])
    rng.shuffle(rest)
    others = [s for s in range(4) if s != keep_seat]
    for i, s in enumerate(others):
        env.hands[s] = set(int(c) for c in rest[i * 13:(i + 1) * 13])


def _sigma(env, au):
    env.auction = au
    obs = env.observe(au.current_player())
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[au.legal_calls()] = 1
    with torch.no_grad():
        d_cat, d_str, d_act = _ADV(torch.tensor(obs, dtype=torch.float32))
    s, parts = hier_sigma(d_cat.numpy(), d_str.numpy(), d_act.numpy(), mask)
    return obs, mask, s, parts


def _rollout(env, au, dv, traverser, rng, t):
    au = copy.deepcopy(au)
    lam = anchor_lambda(t)
    for _ in range(MAX_ROLLOUT_DEPTH):
        if au.is_over():
            return auction_reward(dv, au, traverser)
        if len(au.calls) >= TRUNC_MIN_CALLS and rng.random() < TRUNC_P:
            return auction_reward_truncated(dv, au, traverser)
        _, _, s, _ = _sigma(env, au)
        if lam > 0.0:
            pr = reference_policy(env.hands[au.current_player()], au, au.legal_calls())
            tot = pr.sum()
            if tot > 0.0:
                s = (1.0 - lam) * s + lam * (pr / tot)
                s = s / s.sum()
        au.apply(int(rng.choice(NUM_ACTIONS, p=s)))
    return auction_reward_truncated(dv, au, traverser)


def _traverse(env, dv, traverser, adv_buf, pol_buf, t, stats, rng):
    au = AuctionState(env.dealer)
    env.phase = AUCTION
    depth = 0
    while not au.is_over() and depth < MAX_CALLS:
        depth += 1
        p = au.current_player()
        obs, mask, sigma, parts = _sigma(env, au)

        if p != traverser:
            pol_buf.append((pack_obs(obs), sigma.astype(np.float32), mask, t))
            au.apply(int(rng.choice(NUM_ACTIONS, p=sigma)))
            continue

        v = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in np.flatnonzero(mask):
            ch = copy.deepcopy(au)
            ch.apply(int(a))
            v[a] = (auction_reward(dv, ch, traverser) if ch.is_over()
                    else _rollout(env, ch, dv, traverser, rng, t))

        r_cat, r_str, r_act = hier_regrets(v, parts, mask)
        adv_buf.append((pack_obs(obs), r_cat, r_str, r_act, mask, t))
        pol_buf.append((pack_obs(obs), sigma.astype(np.float32), mask, t))
        stats[6] += float(sigma[PASS])
        stats[7] += 1.0
        au.apply(int(rng.choice(NUM_ACTIONS, p=sigma)))

    lvl = 0 if au.last_bid is None else au.last_bid // 5 + 1
    stats[0] += len(au.calls)
    stats[1] += lvl
    stats[2] += 1
    stats[3] += lvl >= 6
    stats[4] += not au.is_over()
    stats[5] += bool(au.doubled or au.redoubled)


def worker_task(deals, t, adv_sd):
    _ADV.load_state_dict(adv_sd)
    rng = np.random.default_rng()
    local_adv, local_pol = [], []
    stats = [0.0] * 8

    for _ in range(deals):
        env = BridgeEnv()
        env.reset()
        base = [set(h) for h in env.hands]
        for traverser in range(4):
            for k in range(REDEALS):
                env.hands = [set(h) for h in base]
                if k > 0:
                    _redeal(env, traverser, rng)
                dv = DealValue(env)
                for _ in range(TRAVERSALS_PER_DEAL):
                    _traverse(env, dv, traverser, local_adv, local_pol, t, stats, rng)

    return local_adv, local_pol, stats


def _sl_worker(deals, seed0):
    obs_l, tgt_l, mask_l = [], [], []
    for k in range(deals):
        sd = seed0 + k
        rng = np.random.default_rng(sd)
        env = BridgeEnv(seed=sd)
        env.reset()
        au = AuctionState(env.dealer)
        env.phase = AUCTION
        for _ in range(MAX_CALLS):
            if au.is_over():
                break
            p = au.current_player()
            env.auction = au
            legal = au.legal_calls()
            mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
            mask[legal] = 1
            tgt = reference_policy(env.hands[p], au, legal)

            obs_l.append(env.observe(p))
            tgt_l.append(tgt.astype(np.float32))
            mask_l.append(mask)

            if rng.random() < 0.10 and len(legal) > 2:
                action = int(legal[int(rng.integers(len(legal) // 2, len(legal)))])
            else:
                action = int(rng.choice(NUM_ACTIONS, p=tgt))
            au.apply(action)

    return (np.asarray(obs_l, dtype=np.float32), np.asarray(tgt_l, dtype=np.float32),
            np.asarray(mask_l, dtype=np.int8))


def _masked_mse(pred, tgt, m, w=1.0):
    return (w * m * (tgt - pred) ** 2).sum() / (w * m).sum().clamp(min=1e-6)


def pretrain_supervised(pol, adv, deals=500_000, epochs=30, batch=8192, lr=1e-3, num_workers=24, seed0=3_000_000):
    per = max(1, deals // num_workers)
    with concurrent.futures.ProcessPoolExecutor(num_workers, mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(_sl_worker, [per] * num_workers, [seed0 + i * per * 4 for i in range(num_workers)]))
    obs_t = torch.from_numpy(np.concatenate([w[0] for w in results])).to(DEVICE)
    tgt_t = torch.from_numpy(np.concatenate([w[1] for w in results])).to(DEVICE)
    mask_t = torch.from_numpy(np.concatenate([w[2] for w in results])).float().to(DEVICE)
    n = obs_t.shape[0]

    opt = optim.Adam(pol.parameters(), lr=lr)
    pol.train()
    for _ in tqdm(range(epochs), desc="policy pretraining"):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            if i.numel() < 8:
                continue
            lg = pol(obs_t[i]).masked_fill(mask_t[i] == 0, -1e9)
            loss = -(tgt_t[i] * F.log_softmax(lg, dim=-1)).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
    pol.eval()

    opt_a = optim.Adam(adv.parameters(), lr=lr)
    adv.train()
    for _ in tqdm(range(ADV_PRETRAIN_EPOCHS), desc="advantage pretraining"):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            mk = mask_t[idx]
            t_cat, t_str, t_act = hier_targets_torch(tgt_t[idx], mk, ADV_PRETRAIN_SCALE)
            m_cat, m_str, m_act = hier_masks_torch(mk)
            p_cat, p_str, p_act = adv(obs_t[idx])
            loss = (_masked_mse(p_cat, t_cat, m_cat) + _masked_mse(p_str, t_str, m_str)
                    + _masked_mse(p_act, t_act, m_act))
            opt_a.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
            opt_a.step()
    adv.eval()


def _validate(pol_net, seeds, tables, vul, par, temp=1.0):
    scores, levels, lens, contracts, losses, ents = [], [], [], [], [], []
    doubled = passed = 0
    for i, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        env = BridgeEnv(seed=sd)
        env.reset()
        guard = 0
        while env.phase == AUCTION and not env.terminated and guard < MAX_CALLS:
            guard += 1
            obs = env.observe(env.current_player)
            mask = env.get_action_mask()
            with torch.no_grad():
                lg = pol_net(torch.tensor(obs, dtype=torch.float32))
                lg = lg.masked_fill(torch.tensor(mask, dtype=torch.float32) == 0, -1e9)
                q = torch.softmax(lg, dim=-1).numpy().astype(np.float64)
                q = q[q > 1e-12]
                ents.append(float(-(q * np.log(q)).sum()))
                if temp <= 0:
                    env.step(int(torch.argmax(lg).item()))
                    continue
                pr = torch.softmax(lg / temp, dim=-1).numpy().astype(np.float64)
            pr /= pr.sum()
            env.step(int(rng.choice(NUM_ACTIONS, p=pr)))

        lens.append(len(env.auction.calls))
        if env.terminated or env.phase == AUCTION:
            scores.append(0.0)
            losses.append(abs(float(par[i])))
            if env.terminated:
                levels.append(0)
                passed += 1
            else:
                levels.append(0 if env.auction.last_bid is None else env.auction.last_bid // 5 + 1)
            continue
        levels.append(env.level)
        contracts.append((env.level, env.strain))
        doubled += bool(env.contract_doubled or env.contract_redoubled)
        dv = DealValue.from_tricks(tables[i], tuple(vul[i]))
        sc_ns = float(dv.score_ns(env.level, env.strain, env.contract_doubled, env.contract_redoubled, env.declarer))
        scores.append(sc_ns if env.declarer % 2 == 0 else -sc_ns)
        losses.append(abs(float(par[i]) - sc_ns))

    lv = np.array(levels)
    n = len(seeds)
    nc = max(len(contracts), 1)
    return {
        "score": float(np.mean(scores)),
        "loss": float(np.mean(losses)),
        "loss_se": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
        "ent": float(np.mean(ents)) if ents else 0.0,
        "level": float(lv[lv > 0].mean()) if (lv > 0).any() else 0.0,
        "tail6": float((lv >= 6).mean() * 100),
        "nt": sum(1 for _, s in contracts if s == 4) / nc * 100,
        "major4": sum(1 for l, s in contracts if l >= 4 and s in (2, 3)) / nc * 100,
        "dbl": doubled / n * 100,
        "pass": passed / n * 100,
        "len": float(np.mean(lens)),
        "hist": [float((lv == k).mean() * 100) for k in range(1, 8)],
        "repertoire": repertoire(contracts),
    }


def _cpu_copy(pol):
    pc = PolicyNetwork()
    pc.load_state_dict({k: v.cpu() for k, v in pol.state_dict().items()})
    return pc.eval()


def train_phase_1_cfr(epochs=3000, deals_per_epoch=16, batch_size=8192, num_workers=24,
                      grad_steps=10, lr=1e-4, ckpt_every=500, val_every=250, val_deals=400,
                      val_seed0=7_500_000, pretrain=True, pretrain_deals=500_000,
                      pretrain_epochs=30, resume_adv_path=None, resume_pol_path=None):
    os.makedirs(CKPT_DIR, exist_ok=True)
    adv = HierAdvantageNetwork().to(DEVICE)
    pol = PolicyNetwork().to(DEVICE)
    if resume_adv_path:
        adv.load_state_dict(torch.load(resume_adv_path, map_location=DEVICE, weights_only=True))
    if resume_pol_path:
        pol.load_state_dict(torch.load(resume_pol_path, map_location=DEVICE, weights_only=True))
        pretrain = False

    val_seeds = list(range(val_seed0, val_seed0 + val_deals))
    tables, vul, val_par = load_dd_tables(val_seeds, num_workers)
    pass_loss = float(np.abs(val_par).mean())

    if pretrain:
        pretrain_supervised(pol, adv, deals=pretrain_deals, epochs=pretrain_epochs, num_workers=num_workers)
        torch.save(pol.state_dict(), f"{CKPT_DIR}/pretrained_policy.pth")

    opt_a = optim.Adam(adv.parameters(), lr=1e-3)
    opt_p = optim.Adam(pol.parameters(), lr=lr)
    sch_a = CosineAnnealingLR(opt_a, T_max=epochs, eta_min=1e-5)
    sch_p = CosineAnnealingLR(opt_p, T_max=epochs, eta_min=1e-5)

    adv_buf = ReservoirBuffer(3_000_000)
    pol_buf = ReservoirBuffer(3_000_000)
    dpw = max(1, round(deals_per_epoch / num_workers))
    best_loss = 1e18
    pbar = tqdm(range(1, epochs + 1), desc="Phase 1 (Deep CFR)")
    with concurrent.futures.ProcessPoolExecutor(num_workers, mp_context=mp.get_context("spawn"),
                                                initializer=_worker_init) as ex:
        for epoch in pbar:
            a_sd = {k: v.cpu() for k, v in adv.state_dict().items()}
            futs = [ex.submit(worker_task, dpw, epoch, a_sd) for _ in range(num_workers)]
            roll = np.zeros(8)
            for f in concurrent.futures.as_completed(futs):
                la, lp, st_ = f.result()
                roll += st_
                for it in la:
                    adv_buf.add(it)
                for it in lp:
                    pol_buf.add(it)

            la_v = lp_v = 0.0
            if len(adv_buf) > batch_size:
                for _ in range(grad_steps):
                    packed, rc, rs, ra, mk, tps = zip(*adv_buf.sample(batch_size))
                    x = torch.tensor(unpack_obs_batch(packed), device=DEVICE)
                    t_cat = torch.tensor(np.array(rc), dtype=torch.float32, device=DEVICE)
                    t_str = torch.tensor(np.array(rs), dtype=torch.float32, device=DEVICE)
                    t_act = torch.tensor(np.array(ra), dtype=torch.float32, device=DEVICE)
                    mask = torch.tensor(np.array(mk), dtype=torch.float32, device=DEVICE)
                    w = torch.tensor(tps, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                    w = w / (w.max() + 1e-8)
                    m_cat, m_str, m_act = hier_masks_torch(mask)
                    p_cat, p_str, p_act = adv(x)
                    loss = (_masked_mse(p_cat, t_cat, m_cat, w) + _masked_mse(p_str, t_str, m_str, w)
                            + _masked_mse(p_act, t_act, m_act, w))
                    opt_a.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
                    opt_a.step()
                la_v = loss.item()
                sch_a.step()

            if len(pol_buf) > batch_size:
                for _ in range(grad_steps):
                    packed, strats, masks, tps = zip(*pol_buf.sample(batch_size))
                    x = torch.tensor(unpack_obs_batch(packed), device=DEVICE)
                    tg = torch.tensor(np.array(strats), dtype=torch.float32, device=DEVICE)
                    mk = torch.tensor(np.array(masks), dtype=torch.float32, device=DEVICE)
                    w = torch.tensor(tps, dtype=torch.float32, device=DEVICE)
                    w = w / (w.max() + 1e-8)
                    lg = pol(x).masked_fill(mk == 0, -1e9)
                    loss = torch.mean(w * F.cross_entropy(lg, tg, reduction="none"))
                    opt_p.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                    opt_p.step()
                lp_v = loss.item()
                sch_p.step()

            pbar.set_postfix(AdvL=f"{la_v:.4f}", PolL=f"{lp_v:.4f}", level=f"{roll[1] / max(roll[2], 1):.2f}",
                             pass_p=f"{roll[6] / max(roll[7], 1):.2f}")

            if val_every and epoch % val_every == 0:
                pc = _cpu_copy(pol)
                d = _validate(pc, val_seeds, tables, vul, val_par, temp=1.0)
                g = _validate(pc, val_seeds, tables, vul, val_par, temp=0.0)
                if g["loss"] < best_loss:
                    best_loss = g["loss"]
                    torch.save(pol.state_dict(), BEST_POLICY_PATH)
                    torch.save(adv.state_dict(), BEST_ADV_PATH)
                append_row(HISTORY_PATH, cfr_row(epoch, g, d, pass_loss))
                tqdm.write(f"{epoch}: loss {g['loss']:.1f} (T=1: {d['loss']:.1f}), level {g['level']:.2f}, "
                           f"pass {g['pass']:.1f}%")

            if epoch % ckpt_every == 0:
                torch.save(pol.state_dict(), f"{CKPT_DIR}/pol_epoch_{epoch}.pth")
                torch.save(adv.state_dict(), f"{CKPT_DIR}/adv_epoch_{epoch}.pth")
    return adv, pol


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--deals", type=int, default=16)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--resume-adv")
    p.add_argument("--resume-pol")
    a = p.parse_args()
    train_phase_1_cfr(epochs=a.epochs, deals_per_epoch=a.deals, num_workers=a.workers,
                      resume_adv_path=a.resume_adv, resume_pol_path=a.resume_pol)
