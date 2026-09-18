"""Phase 0: card play learned with Deep Monte Carlo on contracts from the artificial auction."""
import argparse
import concurrent.futures
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from bridge.env.bridge_env import BridgeEnv
from bridge.env.play_features import FEATURES_VERSION, heuristic_defense, legal_card_features
from bridge.nets.models import PlayQNetwork
from bridge.rozgrywka.dmc_fixes import force_auction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_SEEDS = list(range(4_000_000, 4_000_200))

_DEC, _DEF = None, None


def _eval_pair(dec_net, def_net, seeds, use_heur_def, use_heur_dec):
    results = []
    for sd in seeds:
        env = BridgeEnv(seed=sd)
        env.reset()
        force_auction(env, np.random.default_rng(sd))
        if env.terminated:
            continue
        side = env.declarer % 2
        while not env.terminated:
            p = env.current_player
            legal = env.legal_actions()
            is_dec = p % 2 == side
            if (is_dec and use_heur_dec) or (not is_dec and use_heur_def):
                a = heuristic_defense(env, legal)
            else:
                net = dec_net if is_dec else def_net
                feats = legal_card_features(env, legal)
                q = net.q_all(torch.from_numpy(env.observe(p)), torch.from_numpy(feats))
                a = int(legal[int(torch.argmax(q).item())])
            env.step(a)
        results.append(env.tricks_won[side])
    return float(np.mean(results))


def _worker_init():
    global _DEC, _DEF
    torch.set_num_threads(1)
    seed = (os.getpid() * int(time.time() * 1000)) % (2 ** 31 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _DEC, _DEF = PlayQNetwork(), PlayQNetwork()
    _DEC.eval()
    _DEF.eval()


def _choose(net, obs, env, legal, epsilon, rng):
    feats = legal_card_features(env, legal)
    if epsilon > 0 and rng.random() < epsilon:
        i = int(rng.integers(len(legal)))
    else:
        q = net.q_all(torch.from_numpy(obs), torch.from_numpy(feats))
        i = int(torch.argmax(q).item())
    return int(legal[i]), feats[i]


def _rollout(env, dec_net, def_net, eps_dec, eps_def, rng):
    dside = env.declarer % 2
    data = {True: {"obs": [], "card": [], "trick": [], "belief": []},
            False: {"obs": [], "card": [], "trick": [], "belief": []}}

    while not env.terminated:
        p = env.current_player
        obs, legal = env.observe(p), env.legal_actions()
        is_dec = p % 2 == dside
        a, cf = _choose(dec_net if is_dec else def_net, obs, env, legal,
                        eps_dec if is_dec else eps_def, rng)

        tgt = data[is_dec]
        tgt["obs"].append(obs)
        tgt["card"].append(cf)
        tgt["trick"].append(len(env.completed_tricks))
        bel = np.zeros(156, dtype=np.uint8)
        for k in range(3):
            for c in env.hands[(p + k + 1) % 4]:
                bel[52 * k + c] = 1
        tgt["belief"].append(np.packbits(bel))
        env.step(a)

    win = [t.winner(env.strain) % 2 for t in env.completed_tricks]
    suf = np.zeros((2, len(win) + 1), dtype=np.float32)
    for t in range(len(win) - 1, -1, -1):
        for s in (0, 1):
            suf[s, t] = suf[s, t + 1] + (1.0 if win[t] == s else 0.0)

    data[True]["ret"] = [float(suf[dside, t]) for t in data[True]["trick"]]
    data[False]["ret"] = [float(suf[1 - dside, t]) for t in data[False]["trick"]]
    return data[True], data[False], env.tricks_won[dside]


def worker_collect(episodes, dec_sd, def_sd, eps_dec, eps_def):
    _DEC.load_state_dict(dec_sd)
    _DEF.load_state_dict(def_sd)
    rng = np.random.default_rng()
    keys = ("obs", "card", "ret", "belief")
    dec = {k: [] for k in keys}
    dfn = {k: [] for k in keys}
    tricks = []
    for _ in range(episodes):
        env = BridgeEnv()
        env.reset()
        force_auction(env, rng)
        if env.terminated:
            continue
        d, f, t = _rollout(env, _DEC, _DEF, eps_dec, eps_def, rng)
        for k in keys:
            dec[k].extend(d[k])
            dfn[k].extend(f[k])
        tricks.append(t)
    return dec, dfn, tricks


def _fit(net, opt, obs, card, ret, bel_packed, batch, passes, dev):
    bel = torch.tensor(np.unpackbits(bel_packed, axis=1)[:, :156].astype(np.float32), device=dev)
    n = obs.shape[0]
    tot_q, tot_b, k = 0.0, 0.0, 0
    for _ in range(passes):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            q_vals, belief_logits = net(obs[i], card[i])
            loss_q = F.mse_loss(q_vals, ret[i])
            loss_b = F.binary_cross_entropy_with_logits(belief_logits, bel[i])
            loss = loss_q + 0.5 * loss_b
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot_q += loss_q.item()
            tot_b += loss_b.item()
            k += 1
    return tot_q / k, tot_b / k


def train_dmc(epochs=8000, episodes_per_epoch=2048, num_workers=24, batch=8192,
              passes=4, lr=1e-4, eps_start=0.30, eps_end=0.02,
              eval_every=100, ckpt_every=1000, init_from=None,
              out_path="models/play_dmc.pth"):
    dec, def_net = PlayQNetwork().to(DEVICE), PlayQNetwork().to(DEVICE)
    if init_from:
        ck = torch.load(init_from, map_location=DEVICE, weights_only=True)
        dec.load_state_dict(ck["dec"])
        def_net.load_state_dict(ck["def"])

    opt_d = optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    opt_f = optim.AdamW(def_net.parameters(), lr=lr, weight_decay=1e-4)
    sch_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-5)
    sch_f = optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=epochs, eta_min=1e-5)

    hist_tricks, hist_ref = [], []
    eps_w = max(1, episodes_per_epoch // num_workers)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints/dmc", exist_ok=True)
    pbar = tqdm(range(1, epochs + 1), desc="Phase 0")

    with concurrent.futures.ProcessPoolExecutor(num_workers, initializer=_worker_init) as ex:
        for epoch in pbar:
            eps = eps_start + (eps_end - eps_start) * (epoch / epochs)
            d_sd = {k: v.cpu() for k, v in dec.state_dict().items()}
            f_sd = {k: v.cpu() for k, v in def_net.state_dict().items()}
            futs = [ex.submit(worker_collect, eps_w, d_sd, f_sd, eps, eps) for _ in range(num_workers)]

            keys = ("obs", "card", "ret", "belief")
            data_dec = {k: [] for k in keys}
            data_def = {k: [] for k in keys}
            tr = []
            for fu in concurrent.futures.as_completed(futs):
                d_, f_, t_ = fu.result()
                for k in keys:
                    data_dec[k].extend(d_[k])
                    data_def[k].extend(f_[k])
                tr.extend(t_)
            hist_tricks.append(float(np.mean(tr)))
            tensor = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=DEVICE)

            dec.train()
            def_net.train()
            lq_d, _ = _fit(dec, opt_d, tensor(data_dec["obs"]), tensor(data_dec["card"]),
                           tensor(data_dec["ret"]), np.array(data_dec["belief"]), batch, passes, DEVICE)
            _fit(def_net, opt_f, tensor(data_def["obs"]), tensor(data_def["card"]),
                 tensor(data_def["ret"]), np.array(data_def["belief"]), batch, passes, DEVICE)
            dec.eval()
            def_net.eval()
            sch_d.step()
            sch_f.step()

            post = {"eps": f"{eps:.2f}", "MSE": f"{lq_d:.3f}", "tricks": f"{hist_tricks[-1]:.2f}"}
            if epoch % eval_every == 0 or epoch == epochs:
                dec.cpu()
                def_net.cpu()
                m_dec = _eval_pair(dec, def_net, EVAL_SEEDS, True, False)
                m_def = _eval_pair(dec, def_net, EVAL_SEEDS, False, True)
                dec.to(DEVICE)
                def_net.to(DEVICE)
                hist_ref.append((epoch, m_dec, m_def))
                post.update(declarer=f"{m_dec:.2f}", defence=f"{m_def:.2f}")
            pbar.set_postfix(post)

            if epoch % ckpt_every == 0:
                torch.save({"dec": dec.state_dict(), "def": def_net.state_dict(),
                            "epoch": epoch, "wersja_cech": FEATURES_VERSION},
                           f"checkpoints/dmc/dmc_epoch_{epoch}.pth")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(hist_tricks, lw=.8, alpha=.6)
    if len(hist_tricks) > 50:
        window = np.ones(50) / 50
        ax[0].plot(np.arange(49, len(hist_tricks)), np.convolve(hist_tricks, window, "valid"),
                   lw=2, c="crimson")
    ax[0].set_title("Lewy rozgrywajacego w grze sieci samej ze soba")
    ax[0].set_xlabel("Epoka")
    ax[0].grid(alpha=.4)
    if hist_ref:
        xs = [e for e, _, _ in hist_ref]
        ax[1].plot(xs, [rd for _, rd, _ in hist_ref], "o-", c="#4C72B0", label="lewy wziete jako rozgrywajacy")
        ax[1].plot(xs, [rf for _, _, rf in hist_ref], "o-", c="#C44E52", label="lewy oddane w obronie")
        ax[1].set_title("Wynik na tle prostej reguly (stale rozdania)")
        ax[1].set_xlabel("Epoka")
        ax[1].legend()
        ax[1].grid(alpha=.4)
    plt.tight_layout()
    plt.savefig("models/dmc_learning.png", dpi=180, bbox_inches="tight")
    torch.save({"dec": dec.state_dict(), "def": def_net.state_dict(), "wersja_cech": FEATURES_VERSION}, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--episodes", type=int, default=2048)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--init-from")
    p.add_argument("--out", default="models/play_dmc.pth")
    p.add_argument("--eps-start", type=float, default=0.30)
    a = p.parse_args()
    train_dmc(epochs=a.epochs, episodes_per_epoch=a.episodes, num_workers=a.workers,
              eval_every=a.eval_every, init_from=a.init_from, eps_start=a.eps_start, out_path=a.out)
