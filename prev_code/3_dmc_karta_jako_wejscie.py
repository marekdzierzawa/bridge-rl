import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures

from bridge_env import BridgeEnv, OBS_SIZE, CARD_OFFSET
from play_features import legal_card_features, CARD_FEAT_DIM
from dmc_fixes import force_auction, get_hcp
from play_features import heuristic_defense


EVAL_SEEDS = list(range(4_000_000, 4_000_600))


def _eval_pair(dec_net, def_net, seeds, use_heur_def, use_heur_dec):
    X, Y = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        env = BridgeEnv(seed=sd)
        env.reset()
        hands = [set(h) for h in env.hands]
        force_auction(env, rng)
        if env.terminated:
            continue
        side = env.declarer % 2
        hcp = sum(get_hcp(c) for c in hands[side]) + sum(get_hcp(c) for c in hands[side + 2])
        while not env.terminated:
            p = env.current_player
            legal = env.legal_actions()
            is_dec = (p % 2 == side)
            if is_dec and use_heur_dec:
                a = heuristic_defense(env, legal)
            elif (not is_dec) and use_heur_def:
                a = heuristic_defense(env, legal)
            else:
                net = dec_net if is_dec else def_net
                feats = legal_card_features(env, legal)
                q = net.q_all(torch.from_numpy(env.observe(p)), torch.from_numpy(feats))
                a = int(legal[int(torch.argmax(q).item())])
            env.step(a)
        X.append(hcp); Y.append(env.tricks_won[side])
    if len(X) < 30:
        return None, None
    return float(np.polyfit(X, Y, 1)[0]), float(np.mean(Y))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# SIEC Q(stan, karta)
# ============================================================================
class PlayQNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_SIZE, card_dim=CARD_FEAT_DIM, h=512):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(obs_dim, 1024), nn.ReLU(),
            nn.Linear(1024, h), nn.ReLU(),
        )
        self.card_enc = nn.Sequential(
            nn.Linear(card_dim, 128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(h + 128, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, card):
        return self.head(torch.cat([self.state_enc(obs), self.card_enc(card)], -1)).squeeze(-1)

    @torch.no_grad()
    def q_all(self, obs_1d, card_mat):
        se = self.state_enc(obs_1d.unsqueeze(0)).expand(card_mat.shape[0], -1)
        return self.head(torch.cat([se, self.card_enc(card_mat)], -1)).squeeze(-1)

# ============================================================================
# WORKERY
# ============================================================================
_DEC = None      
_DEF = None      

def _worker_init():
    global _DEC, _DEF
    torch.set_num_threads(1)
    seed = (os.getpid() * int(time.time() * 1000)) % (2 ** 31 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _DEC, _DEF = PlayQNetwork(), PlayQNetwork()
    _DEC.eval(); _DEF.eval()

def _choose(net, obs, env, legal, epsilon, rng):
    feats = legal_card_features(env, legal)
    if epsilon > 0 and rng.random() < epsilon:
        i = int(rng.integers(len(legal)))
    else:
        q = net.q_all(torch.from_numpy(obs), torch.from_numpy(feats))
        i = int(torch.argmax(q).item())
    return int(legal[i]), feats[i]

def _rollout(env, dec_net, def_net, eps_dec, eps_def, rng, collect=True):
    dside = env.declarer % 2
    D = {"obs": [], "card": [], "trick": []}
    F = {"obs": [], "card": [], "trick": []}

    while not env.terminated:
        p = env.current_player
        obs = env.observe(p)
        legal = env.legal_actions()
        is_dec = (p % 2 == dside)
        net = dec_net if is_dec else def_net
        eps = eps_dec if is_dec else eps_def
        a, cf = _choose(net, obs, env, legal, eps, rng)
        
        if collect:
            tgt = D if is_dec else F
            tgt["obs"].append(obs); tgt["card"].append(cf)
            tgt["trick"].append(len(env.completed_tricks))
        env.step(a)

    win = [t.winner(env.strain) % 2 for t in env.completed_tricks]
    suf = np.zeros((2, len(win) + 1), dtype=np.float32)
    for t in range(len(win) - 1, -1, -1):
        for s in (0, 1):
            suf[s, t] = suf[s, t + 1] + (1.0 if win[t] == s else 0.0)

    D["ret"] = [float(suf[dside, t]) for t in D["trick"]]
    F["ret"] = [float(suf[1 - dside, t]) for t in F["trick"]]
    return D, F, env.tricks_won[dside]

def worker_collect(episodes, dec_sd, def_sd, eps_dec, eps_def):
    _DEC.load_state_dict(dec_sd)
    _DEF.load_state_dict(def_sd)
    rng = np.random.default_rng()

    dec = {"obs": [], "card": [], "ret": []}
    dfn = {"obs": [], "card": [], "ret": []}
    tricks, hcp_tricks = [], []

    for _ in range(episodes):
        env = BridgeEnv()
        env.reset()
        force_auction(env, rng)
        if env.terminated:
            continue
        
        side = env.declarer % 2
        hcp = sum(get_hcp(c) for c in env.hands[side]) + \
              sum(get_hcp(c) for c in env.hands[side + 2])

        D, F, t = _rollout(env, _DEC, _DEF, eps_dec, eps_def, rng)
        for k in ("obs", "card", "ret"):
            dec[k].extend(D[k]); dfn[k].extend(F[k])
        tricks.append(t); hcp_tricks.append((hcp, t))

    return dec, dfn, tricks, hcp_tricks

# ============================================================================
# TRENING I WEWNETRZNA EWALUACJA
# ============================================================================
def _fit(net, opt, obs, card, ret, batch, passes, dev):
    n = obs.shape[0]
    tot, k = 0.0, 0
    for _ in range(passes):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            loss = nn.functional.mse_loss(net(obs[i], card[i]), ret[i])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item(); k += 1
    return tot / max(k, 1)

def train_dmc(epochs=4000, episodes_per_epoch=1024, num_workers=24, batch=4096,
              passes=4, lr=1e-4, eps_start=0.30, eps_end=0.05,
              eval_every=250, ckpt_every=500, resume=None):
    
    print("\n" + "=" * 64)
    print("FAZA 0: Deep Monte-Carlo, osobne sieci rozgrywajacego i obrony")
    print(f"  wejscie: stan {OBS_SIZE} + cechy karty {CARD_FEAT_DIM} | {DEVICE.type.upper()}")
    print("=" * 64)

    dec = PlayQNetwork().to(DEVICE)
    dfn = PlayQNetwork().to(DEVICE)
    if resume:
        ck = torch.load(resume, map_location=DEVICE, weights_only=True)
        dec.load_state_dict(ck["dec"]); dfn.load_state_dict(ck["def"])
        print(f"-> wznowiono z {resume}")
        
    opt_d = optim.Adam(dec.parameters(), lr=lr)
    opt_f = optim.Adam(dfn.parameters(), lr=lr)

    hist_tricks, hist_eval, hist_ref = [], [], []
    from collections import deque
    HCP_WINDOW = 100
    hcp_win = deque(maxlen=HCP_WINDOW)
    eps_w = max(1, episodes_per_epoch // num_workers)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints/dmc", exist_ok=True)
    
    pbar = tqdm(range(1, epochs + 1), desc="Faza 0 (DMC)", dynamic_ncols=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers,
                                                initializer=_worker_init) as ex:
        for epoch in pbar:
            frac = epoch / epochs
            eps = eps_start + (eps_end - eps_start) * frac

            d_sd = {k: v.cpu() for k, v in dec.state_dict().items()}
            f_sd = {k: v.cpu() for k, v in dfn.state_dict().items()}
            futs = [ex.submit(worker_collect, eps_w, d_sd, f_sd, eps, eps)
                    for _ in range(num_workers)]

            D = {"obs": [], "card": [], "ret": []}
            F = {"obs": [], "card": [], "ret": []}
            tr, ht = [], []
            
            for fu in concurrent.futures.as_completed(futs):
                d_, f_, t_, h_ = fu.result()
                for k in D:
                    D[k].extend(d_[k]); F[k].extend(f_[k])
                tr.extend(t_); ht.extend(h_)
            
            if not D["obs"]:
                continue

            hist_tricks.append(float(np.mean(tr)))
            
            hcp_win.append(ht)

            dec.train(); dfn.train()
            ld = _fit(dec, opt_d,
                      torch.tensor(np.array(D["obs"]), dtype=torch.float32, device=DEVICE),
                      torch.tensor(np.array(D["card"]), dtype=torch.float32, device=DEVICE),
                      torch.tensor(np.array(D["ret"]), dtype=torch.float32, device=DEVICE),
                      batch, passes, DEVICE)
            lf = _fit(dfn, opt_f,
                      torch.tensor(np.array(F["obs"]), dtype=torch.float32, device=DEVICE),
                      torch.tensor(np.array(F["card"]), dtype=torch.float32, device=DEVICE),
                      torch.tensor(np.array(F["ret"]), dtype=torch.float32, device=DEVICE),
                      batch, passes, DEVICE)
            dec.eval(); dfn.eval()

            post = {"eps": f"{eps:.2f}", "Ldec": f"{ld:.3f}", "Ldef": f"{lf:.3f}",
                    "lewy": f"{hist_tricks[-1]:.2f}"}

            if epoch % eval_every == 0 or epoch == epochs:
                X, Y = [], []
                for chunk in hcp_win:
                    for h_val, t_val in chunk:
                        X.append(h_val); Y.append(t_val)
                if len(X) > 100:
                    agent_slope, agent_intercept = np.polyfit(X, Y, 1)
                    theoretical_dd_slope = 0.509
                    slope_pct = (agent_slope / theoretical_dd_slope) * 100
                    
                    hist_eval.append((epoch, agent_slope, slope_pct))
                    post["nachyl%"] = f"{slope_pct:.0f}"

                    dec.cpu(); dfn.cpu()
                    s_dec, m_dec = _eval_pair(dec, dfn, EVAL_SEEDS, True, False)   # agent gra, heurystyka broni
                    s_def, m_def = _eval_pair(dec, dfn, EVAL_SEEDS, False, True)   # heurystyka gra, agent broni
                    dec.to(DEVICE); dfn.to(DEVICE)
                    hist_ref.append((epoch, s_dec, m_dec, s_def, m_def))
                    tqdm.write(
                        f"-> epoka {epoch}: [self-play] nachyl {agent_slope:.3f} ({slope_pct:.0f}% DD)  |  "
                        f"[vs heurystyka] rozgrywka {s_dec:.3f} ({s_dec/0.509*100:.0f}%, {m_dec:.2f} lew), "
                        f"obrona oddaje {m_def:.2f} lew (mniej = lepiej)")
            
            pbar.set_postfix(post)

            if epoch % ckpt_every == 0:
                torch.save({"dec": dec.state_dict(), "def": dfn.state_dict(), "epoch": epoch},
                           f"checkpoints/dmc/dmc_epoch_{epoch}.pth")

    # ---------------- RAPORT I WYKRESY ----------------
    print("\n" + "=" * 64)
    if hist_eval:
        e0, sa0, pct0 = hist_eval[0]
        e1, sa1, pct1 = hist_eval[-1]
        print(f"POSTEP WZGLEDEM TEORETYCZNEGO DOUBLE-DUMMY (0.509 lewy/PC)")
        print(f"  epoka {e0:<6} | nachylenie {sa0:.3f} ({pct0:.0f}%)")
        print(f"  epoka {e1:<6} | nachylenie {sa1:.3f} ({pct1:.0f}%)")
        print("\n  nachylenie >85% = wyrocznia oddaje zwiazek sily reki z lewami")
        print("                    i nadaje sie jako wejscie do Fazy 1")
    if hist_ref:
        e0 = hist_ref[0]; e1 = hist_ref[-1]
        print("\nNA STALYM PRZECIWNIKU (heurystyka, stale rozdania):")
        print(f"  rozgrywka: epoka {e0[0]:<6} nachyl {e0[1]:.3f} ({e0[1]/0.509*100:.0f}% DD), {e0[2]:.2f} lew")
        print(f"             epoka {e1[0]:<6} nachyl {e1[1]:.3f} ({e1[1]/0.509*100:.0f}% DD), {e1[2]:.2f} lew")
        print(f"  obrona   : oddaje {e0[4]:.2f} -> {e1[4]:.2f} lew (mniej = lepiej)")
        print("\n  SKALA: losowy 0.233 (46%) | heurystyka 0.174 (34%) | DD 0.509 (100%)")
    print("=" * 64)

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].plot(hist_tricks, lw=.8, alpha=.6)
    if len(hist_tricks) > 50:
        k = np.ones(50) / 50
        ax[0].plot(np.arange(49, len(hist_tricks)), np.convolve(hist_tricks, k, "valid"),
                   lw=2, c="crimson")
    ax[0].set_title("Lewy rozgrywajacego (obie sieci sie ucza)")
    ax[0].set_xlabel("Epoka"); ax[0].grid(alpha=.4)

    if hist_eval:
        xs = [e for e, _, _ in hist_eval]
        ax[1].plot(xs, [sa for _, sa, _ in hist_eval], "o-", c="#4C72B0")
        ax[1].axhline(0.509, ls="--", c="gray", label="Teoria DD (0.509)")
        ax[1].set_title("Wsp. kierunkowy lewy/PC")
        ax[1].set_xlabel("Epoka"); ax[1].legend(); ax[1].grid(alpha=.4)
        
        ax[2].plot(xs, [pct for _, _, pct in hist_eval], "o-", c="seagreen")
        ax[2].axhline(100, ls="--", c="gray")
        ax[2].set_title("Zachowane nachylenie lewy/PC [%]\n(KLUCZOWE dla Fazy 1)")
        ax[2].set_xlabel("Epoka"); ax[2].grid(alpha=.4)
        
    plt.tight_layout(); plt.savefig("models/dmc_learning.png", dpi=180, bbox_inches="tight")

    plt.figure(figsize=(11, 6))
    hcp_acc = {}
    for chunk in hcp_win:
        for h, t in chunk:
            hcp_acc.setdefault(h, []).append(t)
    hs = sorted(hcp_acc)
    plt.plot(hs, [np.mean(hcp_acc[h]) for h in hs], "o-", ms=3, c="#2c3e50", label="Agent DMC")
    
    plt.plot([0, 40], [-4.11, 16.25], "--", c="#e74c3c", alpha=.7,
             label="Teoria DD (0.509*PC - 4.11)")
    
    plt.title("Sila reki vs liczba lew")
    plt.xlabel("PC strony rozgrywajacej")
    plt.ylabel("Srednia liczba lew")
    plt.legend()
    plt.grid(alpha=.4, ls=":")
    
    plt.savefig("models/dmc_hcp.png", dpi=180, bbox_inches="tight")

    torch.save({"dec": dec.state_dict(), "def": dfn.state_dict()}, "models/play_dmc.pth")
    print("-> models/play_dmc.pth, models/dmc_learning.png, models/dmc_hcp.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--resume", type=str, default=None)
    a = p.parse_args()
    train_dmc(epochs=a.epochs, episodes_per_epoch=a.episodes,
              num_workers=a.workers, eval_every=a.eval_every, resume=a.resume)