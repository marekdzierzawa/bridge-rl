"""Phase 1: how the simple system imitated at the start of Deep CFR bids."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wspolne import SEED_VALIDATION, STRAIN_NAMES, Report, bid_heur_best, load_deals, run_auctions, save_figure
from bridge.licytacja.deal_value import par_score_ns

N = 1000


def main():
    P = Report("f1_heurystyka")
    seeds, tab, vul, opt = load_deals(SEED_VALIDATION, N)
    r = run_auctions(bid_heur_best, seeds, tab, vul)
    losses = np.abs(opt - np.array([x["score_ns"] for x in r]))
    P(f"prosty system, {N} rozdan od {seeds[0]}")
    P(f"strata: {losses.mean():.1f} +- {losses.std(ddof=1) / np.sqrt(N):.1f} (zawsze pas: {np.abs(opt).mean():.1f})\n")

    H = np.zeros((5, 8))
    O = np.zeros((5, 8))
    for x in r:
        if not x["passed"]:
            H[x["strain"], x["level"]] += 1
    for t, z in zip(tab, vul):
        b = par_score_ns(t, tuple(z))[1]
        if b >= 0:
            O[b % 5, b // 5 + 1] += 1
    H = 100 * H / H.sum()
    O = 100 * O / O.sum()

    P("%-10s %12s %12s" % ("miano", "system", "optimum"))
    for m in range(5):
        P("%-10s %11.1f%% %11.1f%%" % (STRAIN_NAMES[m], H[m].sum(), O[m].sum()))
    P("%-10s %11.1f%% %11.1f%%" % ("mlodsze", H[:2].sum(), O[:2].sum()))
    P("%-10s %11.1f%% %11.1f%%\n" % ("starsze", H[2:4].sum(), O[2:4].sum()))
    P("%-10s %12s %12s" % ("poziom", "system", "optimum"))
    for l in range(1, 8):
        P("%-10d %11.1f%% %11.1f%%" % (l, H[:, l].sum(), O[:, l].sum()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    vmax = max(H.max(), O.max())
    for ax, M, title in [(axes[0], H, "prosty system"), (axes[1], O, "licytacja optymalna")]:
        im = ax.imshow(M[:, 1:], cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for m in range(5):
            for l in range(7):
                ax.text(l, m, "%.0f" % M[m, l + 1], ha="center", va="center", fontsize=8,
                        color="white" if M[m, l + 1] > vmax * 0.6 else "black")
        ax.set_xticks(range(7))
        ax.set_xticklabels(range(1, 8))
        ax.set_yticks(range(5))
        ax.set_yticklabels(STRAIN_NAMES)
        ax.set_xlabel("poziom kontraktu")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label="% kontraktów")
    save_figure(fig, "f1_heurystyka")
    P.close()


if __name__ == "__main__":
    main()
