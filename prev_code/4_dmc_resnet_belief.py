import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures
from collections import deque

from bridge_env import BridgeEnv, OBS_SIZE, CARD_OFFSET
from play_features import legal_card_features, CARD_FEAT_DIM, heuristic_defense
from dmc_fixes import force_auction, get_hcp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_SEEDS = list(range(4_000_000, 4_000_600))

# ============================================================================
# ZAAWANSOWANA ARCHITEKTURA SIECI (ResNet + Opponent Modeling)
# ============================================================================
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return F.relu(x + self.net(x))

class PlayQNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_SIZE, card_dim=CARD_FEAT_DIM, h=512):
        super().__init__()
        
        self.proj = nn.Linear(obs_dim, h)
        self.res1 = ResBlock(h)
        self.res2 = ResBlock(h)
        
        # Auxiliary Head
        self.belief_head = nn.Sequential(
            nn.Linear(h, 256), nn.ReLU(),
            nn.Linear(256, 156)
        )
        
        self.card_enc = nn.Sequential(
            nn.Linear(card_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        
        self.q_head = nn.Sequential(
            nn.Linear(h + 128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, card):
        state_repr = self.res2(self.res1(F.relu(self.proj(obs))))
        belief_logits = self.belief_head(state_repr)
        card_repr = self.card_enc(card)
        q_val = self.q_head(torch.cat([state_repr, card_repr], dim=-1)).squeeze(-1)
        return q_val, belief_logits

    @torch.no_grad()
    def q_all(self, obs_1d, card_mat):
        state_repr = self.res2(self.res1(F.relu(self.proj(obs_1d.unsqueeze(0)))))
        state_repr = state_repr.expand(card_mat.shape[0], -1)
        card_repr = self.card_enc(card_mat)
        return self.q_head(torch.cat([state_repr, card_repr], dim=-1)).squeeze(-1)


def _eval_pair(dec_net, def_net, seeds, use_heur_def, use_heur_dec):
    X, Y = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        env = BridgeEnv(seed=sd)
        env.reset()
        hands = [set(h) for h in env.hands]
        force_auction(env, rng)
        if env.terminated: continue
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
    if len(X) < 30: return None, None
    return float(np.polyfit(X, Y, 1)[0]), float(np.mean(Y))


# ============================================================================
# WORKERY
# ============================================================================
_DEC, _DEF = None, None

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
    D = {"obs": [], "card": [], "trick": [], "belief": []}
    F = {"obs": [], "card": [], "trick": [], "belief": []}

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
            
            bel = np.zeros(156, dtype=np.uint8)
            for c in env.hands[(p + 1) % 4]: bel[c] = 1
            for c in env.hands[(p + 2) % 4]: bel[52 + c] = 1
            for c in env.hands[(p + 3) % 4]: bel[104 + c] = 1
            tgt["belief"].append(np.packbits(bel))
            
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

    dec = {"obs": [], "card": [], "ret": [], "belief": []}
    dfn = {"obs": [], "card": [], "ret": [], "belief": []}
    tricks, hcp_tricks = [], []

    for _ in range(episodes):
        env = BridgeEnv()
        env.reset()
        force_auction(env, rng)
        if env.terminated: continue
        
        side = env.declarer % 2
        hcp = sum(get_hcp(c) for c in env.hands[side]) + sum(get_hcp(c) for c in env.hands[side + 2])

        D, F, t = _rollout(env, _DEC, _DEF, eps_dec, eps_def, rng)
        for k in ("obs", "card", "ret", "belief"):
            dec[k].extend(D[k]); dfn[k].extend(F[k])
        tricks.append(t); hcp_tricks.append((hcp, t))

    return dec, dfn, tricks, hcp_tricks

# ============================================================================
# TRENING I WEWNETRZNA EWALUACJA
# ============================================================================
def _fit(net, opt, obs, card, ret, bel_packed, batch, passes, dev):
    bel_unpacked = np.unpackbits(bel_packed, axis=1)[:, :156].astype(np.float32)
    bel_tensor = torch.tensor(bel_unpacked, dtype=torch.float32, device=dev)
    
    n = obs.shape[0]
    tot_q, tot_b, k = 0.0, 0.0, 0
    
    for _ in range(passes):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            
            q_vals, belief_logits = net(obs[i], card[i])
            
            loss_q = F.mse_loss(q_vals, ret[i])
            loss_b = F.binary_cross_entropy_with_logits(belief_logits, bel_tensor[i])
            
            loss = loss_q + 0.5 * loss_b
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            
            tot_q += loss_q.item()
            tot_b += loss_b.item()
            k += 1
            
    return tot_q / max(k, 1), tot_b / max(k, 1)

def train_dmc(epochs=8000, episodes_per_epoch=2048, num_workers=24, batch=8192,
              passes=4, lr=1e-4, eps_start=0.30, eps_end=0.02,
              eval_every=100, ckpt_every=1000, resume=None):
    
    print("\n" + "=" * 68)
    print("FAZA 0: Advanced Deep Monte-Carlo (ResNet + Opponent Modeling)")
    print(f"  Wejscie: Stan {OBS_SIZE} + Cechy Karty {CARD_FEAT_DIM} | Device: {DEVICE.type.upper()}")
    print("=" * 68)

    dec = PlayQNetwork().to(DEVICE)
    dfn = PlayQNetwork().to(DEVICE)
    if resume:
        ck = torch.load(resume, map_location=DEVICE, weights_only=True)
        dec.load_state_dict(ck["dec"]); dfn.load_state_dict(ck["def"])
        print(f"-> Wznowiono z {resume}")
        
    opt_d = optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    opt_f = optim.AdamW(dfn.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-5)
    scheduler_f = optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=epochs, eta_min=1e-5)
    hist_tricks, hist_eval, hist_ref = [], [], []
    HCP_WINDOW = 100
    hcp_win = deque(maxlen=HCP_WINDOW)
    eps_w = max(1, episodes_per_epoch // num_workers)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints/dmc", exist_ok=True)
    
    pbar = tqdm(range(1, epochs + 1), desc="Faza 0 (Adv-DMC)", dynamic_ncols=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init) as ex:
        for epoch in pbar:
            frac = epoch / epochs
            eps = eps_start + (eps_end - eps_start) * frac

            d_sd = {k: v.cpu() for k, v in dec.state_dict().items()}
            f_sd = {k: v.cpu() for k, v in dfn.state_dict().items()}
            futs = [ex.submit(worker_collect, eps_w, d_sd, f_sd, eps, eps) for _ in range(num_workers)]

            D = {"obs": [], "card": [], "ret": [], "belief": []}
            F = {"obs": [], "card": [], "ret": [], "belief": []}
            tr, ht = [], []
            
            for fu in concurrent.futures.as_completed(futs):
                d_, f_, t_, h_ = fu.result()
                for k in D:
                    D[k].extend(d_[k]); F[k].extend(f_[k])
                tr.extend(t_); ht.extend(h_)
            
            if not D["obs"]: continue

            hist_tricks.append(float(np.mean(tr)))
            hcp_win.append(ht)

            dec.train(); dfn.train()
            lq_d, lb_d = _fit(dec, opt_d,
                              torch.tensor(np.array(D["obs"]), dtype=torch.float32, device=DEVICE),
                              torch.tensor(np.array(D["card"]), dtype=torch.float32, device=DEVICE),
                              torch.tensor(np.array(D["ret"]), dtype=torch.float32, device=DEVICE),
                              np.array(D["belief"]), batch, passes, DEVICE)
                              
            lq_f, lb_f = _fit(dfn, opt_f,
                              torch.tensor(np.array(F["obs"]), dtype=torch.float32, device=DEVICE),
                              torch.tensor(np.array(F["card"]), dtype=torch.float32, device=DEVICE),
                              torch.tensor(np.array(F["ret"]), dtype=torch.float32, device=DEVICE),
                              np.array(F["belief"]), batch, passes, DEVICE)
            dec.eval(); dfn.eval()
            
            scheduler_d.step()
            scheduler_f.step()

            current_lr = opt_d.param_groups[0]['lr']
            post = {"eps": f"{eps:.2f}", "lr": f"{current_lr:.1e}", "MSE": f"{lq_d:.3f}", "Belief": f"{lb_d:.3f}", "lewy": f"{hist_tricks[-1]:.2f}"}

            if epoch % eval_every == 0 or epoch == epochs:
                X, Y = [], []
                for chunk in hcp_win:
                    for h_val, t_val in chunk:
                        X.append(h_val); Y.append(t_val)
                if len(X) > 100:
                    agent_slope, _ = np.polyfit(X, Y, 1)
                    theoretical_dd_slope = 0.509
                    slope_pct = (agent_slope / theoretical_dd_slope) * 100
                    
                    hist_eval.append((epoch, agent_slope, slope_pct))
                    post["nachyl%"] = f"{slope_pct:.0f}"

                    dec.cpu(); dfn.cpu()
                    s_dec, m_dec = _eval_pair(dec, dfn, EVAL_SEEDS, True, False)
                    s_def, m_def = _eval_pair(dec, dfn, EVAL_SEEDS, False, True)
                    dec.to(DEVICE); dfn.to(DEVICE)
                    hist_ref.append((epoch, s_dec, m_dec, s_def, m_def))
                    tqdm.write(
                        f"-> {epoch}: [Self-Play] nachyl {agent_slope:.3f} ({slope_pct:.0f}%) | "
                        f"[Heurystyka] rozg {s_dec:.3f} ({s_dec/0.509*100:.0f}%, {m_dec:.2f}L), obrona oddaje {m_def:.2f}L")
            
            pbar.set_postfix(post)

            if epoch % ckpt_every == 0:
                torch.save({"dec": dec.state_dict(), "def": dfn.state_dict(), "epoch": epoch},
                           f"checkpoints/dmc/dmc_epoch_{epoch}.pth")

    # ---------------- RAPORT I WYKRESY ----------------
    print("\n" + "=" * 68)
    if hist_eval:
        e0, sa0, pct0 = hist_eval[0]
        e1, sa1, pct1 = hist_eval[-1]
        print(f"POSTEP WZGLEDEM TEORETYCZNEGO DOUBLE-DUMMY (0.509 lewy/PC)")
        print(f"  epoka {e0:<6} | nachylenie {sa0:.3f} ({pct0:.0f}%)")
        print(f"  epoka {e1:<6} | nachylenie {sa1:.3f} ({pct1:.0f}%)")
    if hist_ref:
        e0 = hist_ref[0]; e1 = hist_ref[-1]
        print("\nNA STALYM PRZECIWNIKU (heurystyka, stale rozdania):")
        print(f"  rozgrywka: epoka {e0[0]:<6} nachyl {e0[1]:.3f} ({e0[1]/0.509*100:.0f}% DD), {e0[2]:.2f} lew")
        print(f"             epoka {e1[0]:<6} nachyl {e1[1]:.3f} ({e1[1]/0.509*100:.0f}% DD), {e1[2]:.2f} lew")
        print(f"  obrona   : oddaje {e0[4]:.2f} -> {e1[4]:.2f} lew (mniej = lepiej)")
    print("=" * 68)

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].plot(hist_tricks, lw=.8, alpha=.6)
    if len(hist_tricks) > 50:
        k = np.ones(50) / 50
        ax[0].plot(np.arange(49, len(hist_tricks)), np.convolve(hist_tricks, k, "valid"), lw=2, c="crimson")
    ax[0].set_title("Lewy rozgrywajacego (obie sieci sie ucza)"); ax[0].set_xlabel("Epoka"); ax[0].grid(alpha=.4)

    if hist_eval:
        xs = [e for e, _, _ in hist_eval]
        ax[1].plot(xs, [sa for _, sa, _ in hist_eval], "o-", c="#4C72B0")
        ax[1].axhline(0.509, ls="--", c="gray", label="Teoria DD (0.509)")
        ax[1].set_title("Wsp. kierunkowy lewy/PC"); ax[1].set_xlabel("Epoka"); ax[1].legend(); ax[1].grid(alpha=.4)
        
        ax[2].plot(xs, [pct for _, _, pct in hist_eval], "o-", c="seagreen")
        ax[2].axhline(100, ls="--", c="gray")
        ax[2].set_title("Zachowane nachylenie lewy/PC [%]"); ax[2].set_xlabel("Epoka"); ax[2].grid(alpha=.4)
        
    plt.tight_layout(); plt.savefig("models/dmc_learning.png", dpi=180, bbox_inches="tight")

    plt.figure(figsize=(11, 6))
    hcp_acc = {}
    for chunk in hcp_win:
        for h, t in chunk:
            hcp_acc.setdefault(h, []).append(t)
    hs = sorted(hcp_acc)
    plt.plot(hs, [np.mean(hcp_acc[h]) for h in hs], "o-", ms=3, c="#2c3e50", label="Agent DMC (ResNet+Belief)")
    plt.plot([0, 40], [-4.11, 16.25], "--", c="#e74c3c", alpha=.7, label="Teoria DD (0.509*PC - 4.11)")
    plt.title("Sila reki vs liczba lew"); plt.xlabel("PC strony rozgrywajacej"); plt.ylabel("Srednia liczba lew")
    plt.legend(); plt.grid(alpha=.4, ls=":")
    plt.savefig("models/dmc_hcp.png", dpi=180, bbox_inches="tight")

    torch.save({"dec": dec.state_dict(), "def": dfn.state_dict()}, "models/play_dmc.pth")
    print("-> Zapisano: models/play_dmc.pth, models/dmc_learning.png, models/dmc_hcp.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--episodes", type=int, default=2048)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--resume", type=str, default=None)
    a = p.parse_args()
    train_dmc(epochs=a.epochs, episodes_per_epoch=a.episodes,
              num_workers=a.workers, eval_every=a.eval_every, resume=a.resume)