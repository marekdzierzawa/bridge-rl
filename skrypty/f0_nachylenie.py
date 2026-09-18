"""Phase 0: how many tricks one high card point is worth under double dummy play."""
import numpy as np

from wspolne import Report
from bridge.env.bridge_env import BridgeEnv
from bridge.ocena.zbior_walidacyjny import load_dd_tables
from bridge.rozgrywka.dmc_fixes import force_auction, get_hcp

SEED_RANGES = [(7_500_000, 1000), (8_000_000, 1000)]
HCP_THRESHOLD = 20
N_RESAMPLES = 2000


def slope_ci(x, y, groups):
    rng = np.random.default_rng(0)
    n_groups = groups.max() + 1
    members = [np.flatnonzero(groups == g) for g in range(n_groups)]
    out = []
    for _ in range(N_RESAMPLES):
        idx = np.concatenate([members[w] for w in rng.integers(0, n_groups, n_groups)])
        out.append(np.polyfit(x[idx], y[idx], 1)[0])
    return np.percentile(out, [2.5, 97.5])


def describe_fit(P, name, x, y, groups):
    a, b = np.polyfit(x, y, 1)
    r2 = 1.0 - (y - (a * x + b)).var() / y.var()
    lo, hi = slope_ci(x, y, groups)
    P(f"   {name}: nachylenie {a:.3f} ({lo:.3f}-{hi:.3f}), wyraz wolny {b:+.2f}, R2 {r2:.2f}, "
      f"punktow {len(x)}, PC {int(x.min())}-{int(x.max())}")
    return a, b


def main():
    P = Report("f0_nachylenie")
    side_hcp, best_tricks, mean_tricks, nt_tricks, deal_idx = [], [], [], [], []
    contract_hcp, contract_tricks, contract_deal_idx = [], [], []
    deal_no = 0
    for seed0, n in SEED_RANGES:
        seeds = list(range(seed0, seed0 + n))
        tab = load_dd_tables(seeds)[0]
        for i, sd in enumerate(seeds):
            env = BridgeEnv(seed=sd)
            env.reset()
            pc = [sum(get_hcp(c) for c in env.hands[s] | env.hands[s + 2]) for s in (0, 1)]
            for s in (0, 1):
                tricks = np.maximum(tab[i, :, s], tab[i, :, s + 2])
                side_hcp.append(pc[s])
                best_tricks.append(float(tricks.max()))
                mean_tricks.append(float(tricks.mean()))
                nt_tricks.append(float(tricks[4]))
                deal_idx.append(deal_no)
            force_auction(env, np.random.default_rng(sd))
            if env.declarer is not None and not env.terminated:
                contract_hcp.append(pc[env.declarer % 2])
                contract_tricks.append(float(tab[i, env.strain, env.declarer]))
                contract_deal_idx.append(deal_no)
            deal_no += 1

    side_hcp = np.array(side_hcp, dtype=float)
    best_tricks, mean_tricks, nt_tricks = np.array(best_tricks), np.array(mean_tricks), np.array(nt_tricks)
    groups = np.array(deal_idx)
    P(f"gra idealna, {deal_no} rozdan")

    P("\n1. wszystkie rece")
    a1, b1 = describe_fit(P, "najlepsze miano", side_hcp, best_tricks, groups)
    a2, b2 = describe_fit(P, "srednia po mianach", side_hcp, mean_tricks, groups)
    a6, b6 = describe_fit(P, "bez atu", side_hcp, nt_tricks, groups)

    P(f"\n2. rece od {HCP_THRESHOLD} PC")
    m = side_hcp >= HCP_THRESHOLD
    group_idx = np.unique(groups[m], return_inverse=True)[1]
    a3, b3 = describe_fit(P, "najlepsze miano", side_hcp[m], best_tricks[m], group_idx)
    describe_fit(P, "srednia po mianach", side_hcp[m], mean_tricks[m], group_idx)
    describe_fit(P, "bez atu", side_hcp[m], nt_tricks[m], group_idx)

    P("\n3. kontrakt z reguly fazy 0")
    group_idx_k = np.unique(np.array(contract_deal_idx), return_inverse=True)[1]
    a5, b5 = describe_fit(P, "lewy rozgrywajacego", np.array(contract_hcp, dtype=float), np.array(contract_tricks),
                          group_idx_k)

    P("\n4. przewidywania prostych")
    P("   %-30s %7s %7s %7s %7s %7s" % ("prosta", "0 PC", "20 PC", "25 PC", "33 PC", "40 PC"))
    for name, a, b in [("0.509 * PC - 4.11", 0.509, -4.11), ("najlepsze miano", a1, b1),
                       ("srednia po mianach", a2, b2), ("bez atu", a6, b6),
                       ("najlepsze miano, od 20 PC", a3, b3), ("kontrakt z reguly", a5, b5)]:
        P("   %-30s %7.1f %7.1f %7.1f %7.1f %7.1f" % (name, b, 20 * a + b, 25 * a + b, 33 * a + b, 40 * a + b))
    P.close()


if __name__ == "__main__":
    main()
