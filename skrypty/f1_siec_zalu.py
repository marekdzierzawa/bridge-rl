"""Phase 1: behaviour of Deep CFR before the regret network is trained."""
import collections

import numpy as np
import torch

from wspolne import Report, bid_heur_best
from bridge.env.bridge_env import AUCTION, DOUBLE, PASS, REDOUBLE, BridgeEnv
from bridge.nets.hier_rm import _rm, hier_sigma
from bridge.nets.models import HierAdvantageNetwork

N_NETS = 12
N_DEALS = 200
N_SAMPLES = 50


def call_name(a):
    if a == PASS:
        return "pas"
    if a == DOUBLE:
        return "kontra"
    if a == REDOUBLE:
        return "rekontra"
    return f"{a // 5 + 1}{['trefl', 'karo', 'kier', 'pik', 'BA'][a % 5]}"


def main():
    P = Report("f1_siec_zalu")
    torch.set_num_threads(4)

    openings = []
    for sd in range(9_000_000, 9_000_000 + N_DEALS):
        env = BridgeEnv(seed=sd)
        env.reset()
        openings.append((env.observe(env.current_player).astype(np.float32), env.get_action_mask().astype(np.int8)))
    P(f"1. losowa siec zalu: {N_NETS} inicjalizacji, otwarcie na {N_DEALS} rozdaniach")
    P("   %-6s %-14s %22s %16s" % ("siec", "najczestsza", "rozdan z ta odzywka", "srednia masa"))
    top_counts = collections.Counter()
    n_single = 0
    for k in range(N_NETS):
        torch.manual_seed(k)
        net = HierAdvantageNetwork().eval()
        top, top_prob = [], []
        with torch.no_grad():
            for obs, mk in openings:
                c, s, a = net(torch.from_numpy(obs))
                sig = hier_sigma(c.numpy(), s.numpy(), a.numpy(), mk)[0]
                top.append(int(np.argmax(sig)))
                top_prob.append(float(sig.max()))
        top_call, count = collections.Counter(top).most_common(1)[0]
        top_counts[call_name(top_call)] += 1
        n_single += count == N_DEALS
        P("   %-6d %-14s %15d / %d %16.3f" % (k, call_name(top_call), count, N_DEALS, np.mean(top_prob)))
    P(f"   jedna odzywka na wszystkich rozdaniach: {n_single} z {N_NETS}")
    P("   " + ", ".join(f"{a} ({n})" for a, n in top_counts.most_common()))

    positions = []
    for sd in range(9_100_000, 9_100_000 + 400):
        rng = np.random.default_rng(sd)
        env = BridgeEnv(seed=sd)
        env.reset()
        k = 0
        while env.phase == AUCTION and not env.terminated and k < 40:
            mk = env.get_action_mask().astype(np.int8)
            if mk[DOUBLE]:
                positions.append(mk)
            env.step(bid_heur_best(env, rng))
            k += 1
    rng = np.random.default_rng(0)
    flat, hierarchical = [], []
    for mk in positions:
        idx = np.flatnonzero(mk)
        for _ in range(N_SAMPLES):
            flat.append(_rm(rng.standard_normal(len(mk)), idx, 0.0)[DOUBLE])
            sig = hier_sigma(rng.standard_normal(4), rng.standard_normal(5), rng.standard_normal(len(mk)), mk)[0]
            hierarchical.append(sig[DOUBLE])
    p1, p2 = 100 * np.mean(flat), 100 * np.mean(hierarchical)
    P(f"\n2. kontra przy losowych zalach ({len(positions)} pozycji, po {N_SAMPLES} losowan)")
    P(f"   legalnych akcji srednio: {np.mean([mk.sum() for mk in positions]):.1f}")
    P(f"   kontra: plasko {p1:.1f}%, pietrowo {p2:.1f}% ({p2 / p1:.1f}x)")
    P.close()


if __name__ == "__main__":
    main()
