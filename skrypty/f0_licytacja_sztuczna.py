"""Phase 0: contract distribution of the artificial auction used in training."""
import numpy as np

from wspolne import Report
from bridge.env.bridge_env import AUCTION, NUM_BIDS, BridgeEnv
from bridge.rozgrywka.dmc_fixes import force_auction, get_hcp

STRAIN_NAMES = ["trefl", "karo", "kier", "pik", "bez atu"]
N = 3000


def main():
    P = Report("f0_licytacja_sztuczna")
    levels = np.zeros(8, dtype=int)
    strains = np.zeros(5, dtype=int)
    lengths, entries, n_doubled, passed_out, taken_over = [], 0, 0, 0, 0

    for sd in range(N):
        env = BridgeEnv(seed=sd)
        env.reset()
        ns = sum(get_hcp(c) for c in env.hands[0] | env.hands[2])
        intended_side = 0 if ns >= 20 else 1
        force_auction(env, np.random.default_rng(sd))
        if env.phase == AUCTION or env.terminated:
            passed_out += 1
            continue
        calls = env.auction.calls
        lengths.append(len(calls))
        entries += any(p % 2 != intended_side and a < NUM_BIDS for p, a in calls)
        taken_over += env.declarer % 2 != intended_side
        n_doubled += bool(env.contract_doubled)
        levels[env.level] += 1
        strains[env.strain] += 1

    n = len(lengths)
    P(f"sztuczna licytacja, {N} rozdan")
    P(f"kontraktow: {n}, spasowanych: {passed_out}")
    P(f"srednia dlugosc licytacji: {np.mean(lengths):.1f}")
    P(f"odzywka slabszej strony: {100 * entries / n:.1f}%")
    P(f"kontrakt przejety przez slabsza strone: {100 * taken_over / n:.1f}%")
    P(f"kontry: {100 * n_doubled / n:.1f}%")
    P("\npoziom")
    for p in range(1, 8):
        P(f"   {p}  {100 * levels[p] / n:5.1f}%")
    P(f"sredni poziom: {sum(p * levels[p] for p in range(1, 8)) / n:.2f}")
    P("\nmiano")
    for i, name in enumerate(STRAIN_NAMES):
        P(f"   {name:9s} {100 * strains[i] / n:5.1f}%")
    P.close()


if __name__ == "__main__":
    main()
