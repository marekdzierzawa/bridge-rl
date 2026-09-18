"""Phase 0: quality of the network's card play compared with double dummy play."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wspolne import MODEL_F0, SEED_F0, Report, bootstrap_ci, play_deals, save_figure

N = 1200
DEFENCE_ROLES = ("wist", "obrona")
DECLARER_ROLES = ("rozgrywajacy", "dziadek")


def error_tables(recs, roles, zero_lead=False):
    n = len(recs)
    st = np.array([r["strain"] for r in recs])
    err_sum = np.zeros((n, 12))
    counts = np.zeros((n, 12))
    for i, r in enumerate(recs):
        for role, trick, err, _ in r["errors"]:
            if role not in roles or trick >= 12:
                continue
            if zero_lead and role == "wist":
                err = 0
            err_sum[i, trick] += err
            counts[i, trick] += 1
    return st, err_sum, counts


def nt_excess(st, err_sum, counts, idx=None, per_trick=False):
    if idx is not None:
        st, err_sum, counts = st[idx], err_sum[idx], counts[idx]
    rows, s = [], 0.0
    for lw in range(12):
        avgs = []
        for m in range(5):
            w = st == m
            c = counts[w, lw].sum()
            avgs.append(err_sum[w, lw].sum() / c if c > 3 else np.nan)
        suit_avgs = [x for x in avgs[:4] if np.isfinite(x)]
        if not np.isfinite(avgs[4]) or not suit_avgs:
            continue
        k = float(np.mean(suit_avgs))
        rows.append((lw + 1, avgs[4], k))
        s += avgs[4] - k
    return rows if per_trick else s


def accuracy(recs, roles, strains=None):
    t = [b == 0 for r in recs if strains is None or r["strain"] in strains
         for role, _, b, significant in r["errors"] if role in roles and significant]
    return 100.0 * np.mean(t) if t else float("nan"), len(t)


def main():
    P = Report("f0_audyt")
    seeds = list(range(SEED_F0, SEED_F0 + N))
    recs = play_deals(seeds, "net", "net", True, MODEL_F0)
    P(f"{MODEL_F0}, rozdania {seeds[0]}..{seeds[-1]}, rozegrano {len(recs)}")

    significant = [i for r in recs for _, _, _, i in r["errors"]]
    P("\n1. decyzje istotne")
    P(f"   decyzji: {len(significant)}, istotnych: {100 * np.mean(significant):.1f}%")

    P("\n2. trafnosc (istotne decyzje z najlepsza karta)")
    P("   %-16s %10s %10s %10s" % ("rola", "kolory", "bez atu", "razem"))
    for name, roles in [("wist", ("wist",)), ("obrona bez wistu", ("obrona",)), ("rozgrywajacy", DECLARER_ROLES)]:
        k, _ = accuracy(recs, roles, strains=(0, 1, 2, 3))
        b, _ = accuracy(recs, roles, strains=(4,))
        c, n = accuracy(recs, roles)
        P("   %-16s %9.1f%% %9.1f%% %9.1f%%   (%d decyzji)" % (name, k, b, c, n))

    lead_errors = [b for r in recs for role, _, b, _ in r["errors"] if role == "wist"]
    def_errors = [b for r in recs for role, _, b, _ in r["errors"] if role == "obrona"]
    P("\n3. wist na tle reszty obrony")
    P(f"   blad wistu {np.mean(lead_errors):.3f}, reszty obrony {np.mean(def_errors):.3f} lewy na decyzje")
    P(f"   wist: {100 * len(lead_errors) / (len(lead_errors) + len(def_errors)):.1f}% decyzji obrony, "
      f"{100 * sum(lead_errors) / (sum(lead_errors) + sum(def_errors)):.1f}% straty")

    recs_random = play_deals(seeds, "net", "net", True, MODEL_F0, random_lead=True)
    P("\n4. wist losowy")
    P(f"   trafnosc wistu: siec {accuracy(recs, ('wist',))[0]:.1f}%, losowo {accuracy(recs_random, ('wist',))[0]:.1f}%")

    P("\n5. nadwyzka straty w bez atu (przedzial 95%)")
    results = {}
    for name, roles in [("obrona", DEFENCE_ROLES), ("rozgrywajacy", DECLARER_ROLES)]:
        T = error_tables(recs, roles)
        w = nt_excess(*T)
        lo, hi, _ = bootstrap_ci(lambda i: nt_excess(*T, idx=i), len(recs))
        results[name] = (T, w)
        P("   %-13s %+.3f   [%+.3f, %+.3f]" % (name, w, lo, hi))
    T0 = error_tables(recs, DEFENCE_ROLES, zero_lead=True)
    w0 = nt_excess(*T0)
    lo, hi, _ = bootstrap_ci(lambda i: nt_excess(*T0, idx=i), len(recs))
    P("   obrona bez bledow wistu %+.3f   [%+.3f, %+.3f]" % (w0, lo, hi))
    P("   udzial wistu: %.1f%%" % (100 * (results["obrona"][1] - w0) / results["obrona"][1]))

    P("\n6. sredni blad na decyzje w lewie")
    P("   %-5s %10s %10s | %10s %10s" % ("lewa", "obr. BA", "obr. kol.", "rozg. BA", "rozg. kol."))
    def_by_trick = nt_excess(*results["obrona"][0], per_trick=True)
    dec_by_trick = nt_excess(*results["rozgrywajacy"][0], per_trick=True)
    dr = {l: (b, k) for l, b, k in dec_by_trick}
    for l, b, k in def_by_trick:
        rb, rk = dr.get(l, (np.nan, np.nan))
        P("   %-5d %10.3f %10.3f | %10.3f %10.3f" % (l, b, k, rb, rk))

    y_max = max(max(b, k) for _, b, k in def_by_trick + dec_by_trick) * 1.12
    for data, title, name in [(def_by_trick, "obrona", "f0_nadwyzka_obrona"),
                              (dec_by_trick, "rozgrywający", "f0_nadwyzka_rozgrywajacy")]:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.array([l for l, _, _ in data])
        ax.bar(x - 0.2, [b for _, b, _ in data], 0.4, label="bez atu", color="#c44e52")
        ax.bar(x + 0.2, [k for _, _, k in data], 0.4, label="kolory (średnia)", color="#4c72b0")
        ax.set_title(title)
        ax.set_xlabel("numer lewy")
        ax.set_ylabel("średnia strata na decyzję [lewy]")
        ax.set_xticks(x)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, name)
    P.close()


if __name__ == "__main__":
    main()
