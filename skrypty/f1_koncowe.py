"""Phase 1: results of the final bidding models."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wspolne import SEED_TEST, SEED_VALIDATION, Report, bootstrap_ci, load_deals, save_figure
from bridge.env.bridge_env import NUM_ACTIONS
from bridge.ocena.evaluate_models import bidders, load_policy, par_distribution, score_bidding
from bridge.ocena.wzorce import EXPERT_DOUBLE, EXPERT_LEVEL, EXPERT_MEAN_LEVEL, EXPERT_PASS, EXPERT_STRAIN

ACTOR_PATH = "models/best_actor_final.pth"
CFR_PATH = "models/best_policy.pth"
N = 1000
STRAIN_KEYS = ["C", "D", "H", "S", "NT"]
STRAIN_NAMES = ["trefl", "karo", "kier", "pik", "bez atu"]


def greedy_policy(path):
    lg = load_policy(path)[0]
    return lambda e, r: int(torch.argmax(lg(e)).item())


def sampled_policy(path):
    lg = load_policy(path)[0]

    def f(e, r):
        p = torch.softmax(lg(e), dim=-1).numpy().astype(np.float64)
        return int(r.choice(NUM_ACTIONS, p=p / p.sum()))
    return f


def main():
    ref = bidders(None)
    strategies = [
        ("aktor-krytyk", greedy_policy(ACTOR_PATH)),
        ("Deep CFR", greedy_policy(CFR_PATH)),
        ("prosty system", ref["heuristic, argmax"]),
        ("zawsze pas", ref["always pass"]),
        ("aktor-krytyk, losowanie", sampled_policy(ACTOR_PATH)),
        ("Deep CFR, losowanie", sampled_policy(CFR_PATH)),
    ]

    P = Report("f1_koncowe")
    seeds, tab, vul, opt = load_deals(SEED_TEST, N)
    mean_abs_opt = float(np.abs(opt).mean())
    P(f"aktor-krytyk: {ACTOR_PATH}, Deep CFR: {CFR_PATH}")
    P(f"zbior wydzielony: {N} rozdan, {seeds[0]}..{seeds[-1]}, sredni |zapis optymalny| {mean_abs_opt:.1f}\n")
    W = {name: score_bidding(f, seeds, tab, vul, opt) for name, f in strategies}

    P("1. strata wobec zapisu optymalnego")
    P("   %-26s %8s %22s %8s" % ("strategia", "strata", "przedzial 95%", "ulamek"))
    for name, _ in strategies:
        s = W[name]["losses"]
        lo, hi, _ = bootstrap_ci(lambda i: s[i].mean(), len(s))
        P("   %-26s %8.1f %10.1f .. %-9.1f %8.3f" % (name, s.mean(), lo, hi, s.mean() / mean_abs_opt))

    P("\n2. roznice sparowane (A - B)")
    for a, b in [("aktor-krytyk", "Deep CFR"), ("aktor-krytyk", "prosty system"), ("Deep CFR", "prosty system"),
                 ("aktor-krytyk, losowanie", "aktor-krytyk"), ("Deep CFR, losowanie", "Deep CFR")]:
        d = W[a]["losses"] - W[b]["losses"]
        lo, hi, _ = bootstrap_ci(lambda i: d[i].mean(), len(d))
        P("   %-24s - %-24s %+7.1f   [%+7.1f, %+7.1f]" % (a, b, d.mean(), lo, hi))

    par_level, par_strain, _ = par_distribution(tab, vul)
    P("\n3. profil kontraktow")
    P("   %-24s %8s %8s %9s %8s %10s" % ("strategia", "pasow", "poziom", "poz. 6-7", "kontr", "wykonanych"))
    for name in ("aktor-krytyk", "Deep CFR", "prosty system"):
        w = W[name]
        P("   %-24s %7.1f%% %8.2f %8.1f%% %7.1f%% %9.1f%%" % (name, w["pass"], w["mean_level"],
                                                          w["level"][6] + w["level"][7], w["doubled"], w["made"]))
    P("   %-24s %8s %8.2f %8.1f%%" % ("licytacja optymalna", "-", sum(l * v for l, v in par_level.items()) / 100,
                                     par_level[6] + par_level[7]))
    P("   %-24s %7.1f%% %8.2f %8.1f%% %7.1f%%" % ("eksperci", EXPERT_PASS, EXPERT_MEAN_LEVEL,
                                               EXPERT_LEVEL[6] + EXPERT_LEVEL[7], EXPERT_DOUBLE))

    series = [("aktor-krytyk", W["aktor-krytyk"]["level"], W["aktor-krytyk"]["strain"], "#4c72b0"),
              ("Deep CFR", W["Deep CFR"]["level"], W["Deep CFR"]["strain"], "#dd8452"),
              ("licytacja optymalna", par_level, par_strain, "#55a868"),
              ("eksperci", EXPERT_LEVEL, EXPERT_STRAIN, "#8c8c8c")]
    P("\n4. poziomy (% kontraktow)")
    P("   %-22s %s" % ("", "  ".join("%6d" % l for l in range(1, 8))))
    for name, level_dist, _, _ in series:
        P("   %-22s %s" % (name, "  ".join("%5.1f%%" % level_dist[l] for l in range(1, 8))))
    P("\n5. miana (% kontraktow)")
    P("   %-22s %s" % ("", "  ".join("%7s" % m for m in STRAIN_NAMES)))
    for name, _, strain_dist, _ in series:
        P("   %-22s %s" % (name, "  ".join("%6.1f%%" % strain_dist[k] for k in STRAIN_KEYS)))

    for fname, keys, labels, xlabel, field_idx in [
            ("wyniki_poziomy", list(range(1, 8)), [str(l) for l in range(1, 8)], "poziom kontraktu", 1),
            ("wyniki_miana", STRAIN_KEYS, STRAIN_NAMES, "miano", 2)]:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x = np.arange(len(keys))
        width = 0.2
        for j, s in enumerate(series):
            ax.bar(x + (j - 1.5) * width, [s[field_idx][k] for k in keys], width, label=s[0], color=s[3])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("% kontraktów")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        save_figure(fig, fname)

    val_seeds, val_tables, val_vul, val_opt = load_deals(SEED_VALIDATION, N)
    mean_abs_val = float(np.abs(val_opt).mean())
    P(f"\n6. zbior walidacyjny (sredni |zapis optymalny| {mean_abs_val:.1f})")
    for name, f in strategies[:2]:
        s = score_bidding(f, val_seeds, val_tables, val_vul, val_opt)["losses"]
        P("   %-24s walidacyjny %6.1f (ulamek %.3f) | wydzielony %6.1f (ulamek %.3f)"
          % (name, s.mean(), s.mean() / mean_abs_val, W[name]["losses"].mean(),
             W[name]["losses"].mean() / mean_abs_opt))
    P.close()


if __name__ == "__main__":
    main()
