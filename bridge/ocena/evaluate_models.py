"""Bidding evaluation: loss against the par score and contract distributions compared with par and experts."""
import argparse
import os

import numpy as np
import torch

from bridge.env.bridge_env import AUCTION, NUM_ACTIONS, PASS, BridgeEnv
from bridge.licytacja.bid_reference import reference_policy
from bridge.licytacja.deal_value import DealValue, par_score_ns
from bridge.licytacja.train_ctde import MAX_AUCTION_LEN
from bridge.nets.ctde import BidActor
from bridge.nets.models import PolicyNetwork
from bridge.ocena.wzorce import EXPERT_DOUBLE, EXPERT_LEVEL, EXPERT_STRAIN, STRAIN_CODES
from bridge.ocena.zbior_walidacyjny import load_dd_tables


def score_bidding(bidder, seeds, tables, vul, par):
    losses, levels, strains, pairs, overtricks = [], [], [], [], []
    n_doubled = n_redoubled = n_passed = n_made = 0

    for i, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        env = BridgeEnv(seed=sd)
        env.reset()
        k = 0
        while env.phase == AUCTION and not env.terminated and k < MAX_AUCTION_LEN:
            k += 1
            env.step(bidder(env, rng))

        pv = float(par[i])
        if env.terminated or env.phase == AUCTION:
            n_passed += bool(env.terminated)
            losses.append(abs(pv))
            continue

        levels.append(env.level)
        strains.append(env.strain)
        pairs.append((env.strain, env.level))
        n_doubled += bool(env.contract_doubled)
        n_redoubled += bool(env.contract_redoubled)

        dv = DealValue.from_tricks(tables[i], tuple(vul[i]))
        taken = int(dv.tricks[env.strain, env.declarer])
        overtricks.append(taken - (6 + env.level))
        n_made += taken >= 6 + env.level
        sc_ns = dv.score_ns(env.level, env.strain, env.contract_doubled, env.contract_redoubled, env.declarer)
        losses.append(abs(pv - sc_ns))

    n, nk = len(seeds), max(len(levels), 1)
    st = np.array(losses)
    return {
        "loss": st.mean(),
        "se": st.std(ddof=1) / np.sqrt(n),
        "losses": st,
        "pairs": pairs,
        "pass": n_passed / n * 100,
        "n_contracts": len(levels),
        "level": {l: sum(1 for x in levels if x == l) / nk * 100 for l in range(1, 8)},
        "strain": {STRAIN_CODES[s]: sum(1 for x in strains if x == s) / nk * 100 for s in range(5)},
        "doubled": n_doubled / nk * 100,
        "redoubled": n_redoubled / nk * 100,
        "made": n_made / nk * 100,
        "overtricks": float(np.mean(overtricks)) if overtricks else 0.0,
        "mean_level": float(np.mean(levels)) if levels else 0.0,
    }


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    actor = any(k.startswith("glowa_polityki.") for k in ck)
    net = BidActor() if actor else PolicyNetwork()
    net.load_state_dict(ck)
    net.eval()

    def logits(env):
        obs = torch.tensor(env.observe(env.current_player), dtype=torch.float32)
        mk = torch.tensor(env.get_action_mask(), dtype=torch.float32)
        with torch.no_grad():
            if actor:
                return net.logits(obs, mk)
            return net(obs).masked_fill(mk == 0, -1e9)

    return logits, "CTDE actor" if actor else "Deep CFR policy"


def bidders(policy_path=None):
    def heur(env):
        pr = reference_policy(env.hands[env.current_player], env.auction, env.auction.legal_calls())
        s = pr.sum()
        return pr / s if s > 0 else None

    out = {}
    if policy_path:
        logits, _ = load_policy(policy_path)
        out["model, argmax"] = lambda e, r: int(torch.argmax(logits(e)).item())

        def sample_call(e, r):
            p = torch.softmax(logits(e), dim=-1).numpy().astype(np.float64)
            return int(r.choice(NUM_ACTIONS, p=p / p.sum()))
        out["model, sampling"] = sample_call

    out["heuristic, argmax"] = lambda e, r: PASS if heur(e) is None else int(np.argmax(heur(e)))
    out["always pass"] = lambda e, r: PASS
    out["uniform random"] = lambda e, r: int(r.choice(np.flatnonzero(e.get_action_mask())))
    return out


def par_distribution(tables, vul):
    level_counts = {l: 0 for l in range(1, 8)}
    strain_counts = {m: 0 for m in STRAIN_CODES}
    pairs = []
    for i in range(len(tables)):
        b = par_score_ns(tables[i], tuple(vul[i]))[1]
        if b < 0:
            continue
        level_counts[b // 5 + 1] += 1
        strain_counts[STRAIN_CODES[b % 5]] += 1
        pairs.append((b % 5, b // 5 + 1))
    n = max(len(pairs), 1)
    return {l: v / n * 100 for l, v in level_counts.items()}, {m: v / n * 100 for m, v in strain_counts.items()}, pairs


def report(results, reference, par_level, par_strain, par_pairs):
    w = results[reference]
    print(f"{'':26s}{'loss':>9s}{'+-':>6s}{'pass%':>7s}{'level':>8s}")
    for name, r in results.items():
        print(f"{name:26s}{r['loss']:9.1f}{r['se']:6.0f}{r['pass']:7.1f}{r['mean_level']:8.2f}")
    for name, r in results.items():
        if name != reference:
            d = w["losses"] - r["losses"]
            print(f"{reference} - {name}: {d.mean():+.1f} +- {d.std(ddof=1) / np.sqrt(len(d)):.1f}")

    print("\nlevel   model    par  experts")
    for l in range(1, 8):
        print(f"{l:<6d}{w['level'][l]:7.1f}{par_level[l]:7.1f}{EXPERT_LEVEL[l]:9.1f}")
    print("\nstrain  model    par  experts")
    for m in STRAIN_CODES:
        print(f"{m:<6s}{w['strain'][m]:7.1f}{par_strain[m]:7.1f}{EXPERT_STRAIN[m]:9.1f}")
    print(f"\ndoubled {w['doubled']:.1f}% (experts {EXPERT_DOUBLE}%), redoubled {w['redoubled']:.1f}%, "
          f"made {w['made']:.1f}%, overtricks {w['overtricks']:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="models/best_policy.pth")
    ap.add_argument("--compare")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=7_500_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no-refs", action="store_true")
    a = ap.parse_args()

    seeds = list(range(a.seed0, a.seed0 + a.deals))
    tables, vul, par = load_dd_tables(seeds, a.workers)
    fn = bidders(a.policy)
    if a.no_refs:
        fn = {k: v for k, v in fn.items() if k.startswith("model")}
    if a.compare:
        fn["compare, argmax"] = bidders(a.compare)["model, argmax"]
    results = {name: score_bidding(f, seeds, tables, vul, par) for name, f in fn.items()}
    report(results, "model, argmax", *par_distribution(tables, vul))


if __name__ == "__main__":
    main()
