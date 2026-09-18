"""Phase 1: why the first version of the reward pushed the bidding to the seventh level."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wspolne import Report, save_figure
from bridge.env.bridge_env import duplicate_score

DIAMONDS = 1


def main():
    P = Report("f1_nagroda")
    P(f"3BA z 9 lewami: {duplicate_score(3, 4, False, False, False, 9)} przed partia, "
      f"{duplicate_score(3, 4, False, False, True, 9)} po partii")
    opp_score = duplicate_score(4, 3, False, False, True, 10)
    P(f"4 piki N/S po partii: {opp_score}, dla E/W {-opp_score}")

    P("\n1. stare odniesienie (-620 dla E/W), wpadki bez kontry")
    P("   %-12s %10s %10s" % ("wpadka", "zapis E/W", "nagroda"))
    old_rewards = []
    for x in range(1, 12):
        z = duplicate_score(7, DIAMONDS, False, False, False, 13 - x)
        r = np.tanh((z + opp_score) / 1000)
        old_rewards.append((x, r))
        P("   bez %-8d %10d %+10.2f" % (x, z, r))

    P("\n2. odniesienie: zapis optymalny (E/W 8 lew w karach)")
    results = [("pas (N/S graja 4 piki)", -opp_score)]
    for level in (5, 6, 7):
        results.append((f"{level} karo z kontra, bez {6 + level - 8}",
                        duplicate_score(level, DIAMONDS, True, False, False, 8)))
    optimal = max(z for _, z in results)
    P(f"   zapis optymalny dla E/W: {optimal}")
    P("   %-28s %10s %14s %14s" % ("decyzja E/W", "zapis", "nagroda nowa", "nagroda stara"))
    for desc, z in results:
        P("   %-28s %10d %+14.2f %+14.2f" % (desc, z, np.tanh((z - optimal) / 1000),
                                            np.tanh((z + opp_score) / 1000)))

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot([x for x, _ in old_rewards], [r for _, r in old_rewards], "o-", color="#c44e52",
            label="stare odniesienie, bez kontry")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("głębokość wpadki (liczba brakujących lew)")
    ax.set_ylabel("nagroda")
    ax.set_xticks(range(1, 12))
    ax.set_ylim(-0.1, 0.6)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "f1_nagroda")
    P.close()


if __name__ == "__main__":
    main()
