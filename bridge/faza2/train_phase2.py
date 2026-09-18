"""Phase 2: card play trained from scratch with Deep Monte Carlo on contracts bid by the phase 1 model."""
import argparse
import concurrent.futures
import multiprocessing as mp
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from bridge.dd.dd_metrics import dd_endgame_eval, format_eval
from bridge.env.bridge_env import AUCTION, BridgeEnv
from bridge.env.config import DEVICE
from bridge.env.play_features import FEATURES_VERSION, legal_card_features
from bridge.nets.models import PlayQNetwork
from bridge.ocena.evaluate_models import load_policy
from bridge.rozgrywka.dmc_fixes import force_auction

EVAL_SEED0 = 9_000_000

_LOGITS = None
_DEC = None
_DEF = None


def _worker_init(policy_path):
    global _LOGITS, _DEC, _DEF
    torch.set_num_threads(1)
    seed = (os.getpid() * int(time.time() * 1000)) % (2 ** 31 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if policy_path:
        _LOGITS = load_policy(policy_path)[0]
    _DEC = PlayQNetwork().eval()
    _DEF = PlayQNetwork().eval()


def _choose(net, obs, env, legal, eps, rng):
    feats = legal_card_features(env, legal)
    if eps > 0 and rng.random() < eps:
        i = int(rng.integers(len(legal)))
    else:
        q = net.q_all(torch.from_numpy(obs), torch.from_numpy(feats))
        i = int(torch.argmax(q).item())
    return int(legal[i]), feats[i]


def _run_auction(env, rng, source):
    if source == "regula":
        force_auction(env, rng)
        return
    while env.phase == AUCTION and not env.terminated:
        env.step(int(torch.argmax(_LOGITS(env)).item()))


def worker_collect(episodes, dec_sd, def_sd, eps, source):
    _DEC.load_state_dict(dec_sd)
    _DEF.load_state_dict(def_sd)
    rng = np.random.default_rng()
    data_dec = {"obs": [], "card": [], "ret": [], "bel": []}
    data_def = {"obs": [], "card": [], "ret": [], "bel": []}
    tricks = []

    for _ in range(episodes):
        env = BridgeEnv()
        env.reset()
        _run_auction(env, rng, source)
        if env.terminated or env.phase == AUCTION:
            continue

        dside = env.declarer % 2
        episode = {True: {"obs": [], "card": [], "bel": [], "t": []},
                   False: {"obs": [], "card": [], "bel": [], "t": []}}
        while not env.terminated:
            p = env.current_player
            obs = env.observe(p).astype(np.float32)
            legal = env.legal_actions()
            is_dec = p % 2 == dside
            a, cf = _choose(_DEC if is_dec else _DEF, obs, env, legal, eps, rng)
            bel = np.zeros(156, dtype=np.float32)
            for k in range(3):
                for c in env.hands[(p + k + 1) % 4]:
                    bel[52 * k + c] = 1.0
            tgt = episode[is_dec]
            tgt["obs"].append(obs)
            tgt["card"].append(cf)
            tgt["bel"].append(bel)
            tgt["t"].append(len(env.completed_tricks))
            env.step(a)
        tricks.append(env.tricks_won[dside])

        win = [t.winner(env.strain) % 2 for t in env.completed_tricks]
        suf = np.zeros((2, len(win) + 1), dtype=np.float32)
        for t in range(len(win) - 1, -1, -1):
            for s in (0, 1):
                suf[s, t] = suf[s, t + 1] + (1.0 if win[t] == s else 0.0)
        for src, dst, side in ((episode[True], data_dec, dside), (episode[False], data_def, 1 - dside)):
            dst["obs"].extend(src["obs"])
            dst["card"].extend(src["card"])
            dst["bel"].extend(src["bel"])
            dst["ret"].extend(float(suf[side, t]) for t in src["t"])

    return data_dec, data_def, tricks


def _fit(net, opt, obs, card, ret, bel, batch, passes, w_bel=0.5):
    n = obs.shape[0]
    tot = k = 0
    for _ in range(passes):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            q, bl = net(obs[i], card[i])
            loss = F.mse_loss(q, ret[i]) + w_bel * F.binary_cross_entropy_with_logits(bl, bel[i])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            k += 1
    return tot / k


def train_phase_2(policy_path, epochs=8000, episodes_per_epoch=2048, num_workers=24, batch=8192, passes=4,
                  lr=1e-4, eps_start=0.30, eps_end=0.02, ckpt_every=1000, eval_every=250, eval_deals=120,
                  source="model", out=None):
    dec = PlayQNetwork().to(DEVICE)
    dfn = PlayQNetwork().to(DEVICE)
    opt_d = optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    opt_f = optim.AdamW(dfn.parameters(), lr=lr, weight_decay=1e-4)
    sch_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-5)
    sch_f = optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=epochs, eta_min=1e-5)

    os.makedirs("checkpoints/phase2", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    eps_w = max(1, episodes_per_epoch // num_workers)
    hist = []
    policy = policy_path if source == "model" else None
    tensor = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=DEVICE)
    pbar = tqdm(range(1, epochs + 1), desc=f"Phase 2 ({source})")

    with concurrent.futures.ProcessPoolExecutor(num_workers, mp_context=mp.get_context("spawn"),
                                                initializer=_worker_init, initargs=(policy,)) as ex:
        for epoch in pbar:
            eps = eps_start + (eps_end - eps_start) * (epoch / epochs)
            d_sd = {k: v.cpu() for k, v in dec.state_dict().items()}
            f_sd = {k: v.cpu() for k, v in dfn.state_dict().items()}
            futs = [ex.submit(worker_collect, eps_w, d_sd, f_sd, eps, source) for _ in range(num_workers)]

            data_dec = {"obs": [], "card": [], "ret": [], "bel": []}
            data_def = {"obs": [], "card": [], "ret": [], "bel": []}
            tr = []
            for fu in concurrent.futures.as_completed(futs):
                d_, f_, t_ = fu.result()
                for k in data_dec:
                    data_dec[k].extend(d_[k])
                    data_def[k].extend(f_[k])
                tr.extend(t_)
            hist.append(float(np.mean(tr)))

            dec.train()
            dfn.train()
            ld = _fit(dec, opt_d, tensor(data_dec["obs"]), tensor(data_dec["card"]),
                      tensor(data_dec["ret"]), tensor(data_dec["bel"]), batch, passes)
            lf = _fit(dfn, opt_f, tensor(data_def["obs"]), tensor(data_def["card"]),
                      tensor(data_def["ret"]), tensor(data_def["bel"]), batch, passes)
            dec.eval()
            dfn.eval()
            sch_d.step()
            sch_f.step()
            pbar.set_postfix(eps=f"{eps:.2f}", Ldec=f"{ld:.3f}", Ldef=f"{lf:.3f}", tricks=f"{hist[-1]:.2f}")

            if eval_every > 0 and (epoch == 1 or epoch % eval_every == 0):
                dec.cpu()
                dfn.cpu()
                r = dd_endgame_eval(dec, dfn, range(EVAL_SEED0, EVAL_SEED0 + eval_deals), k=5)
                dec.to(DEVICE)
                dfn.to(DEVICE)
                tqdm.write(f"{epoch}: {format_eval(r)}")

            if epoch % ckpt_every == 0:
                torch.save({"dec": dec.state_dict(), "def": dfn.state_dict(), "epoch": epoch,
                            "licytacja": source, "wersja_cech": FEATURES_VERSION},
                           f"checkpoints/phase2/dmc_{source}_epoch_{epoch}.pth")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist, lw=.8, alpha=.6)
    if len(hist) > 50:
        window = np.ones(50) / 50
        ax.plot(np.arange(49, len(hist)), np.convolve(hist, window, "valid"), lw=2, c="crimson")
    ax.set_title(f"Faza 2 ({source}): lewy rozgrywajacego w grze samej ze soba")
    ax.set_xlabel("Epoka")
    ax.grid(alpha=.4)
    fig.tight_layout()
    fig.savefig(f"models/phase2_{source}_learning.png", dpi=180, bbox_inches="tight")

    out = out or f"models/play_dmc_faza2_{source}.pth"
    torch.save({"dec": dec.state_dict(), "def": dfn.state_dict(), "licytacja": source, "wersja_cech": FEATURES_VERSION},
               out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="models/best_policy.pth")
    p.add_argument("--bidding", default="model", choices=["model", "regula"])
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--episodes", type=int, default=2048)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--out")
    a = p.parse_args()
    train_phase_2(a.policy, epochs=a.epochs, episodes_per_epoch=a.episodes, num_workers=a.workers,
                  batch=a.batch, eval_every=a.eval_every, source=a.bidding, out=a.out)
