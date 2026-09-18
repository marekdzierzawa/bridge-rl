"""Phase 1: three measures of bidding quality compared."""
import numpy as np

from wspolne import (SEED_VALIDATION, Report, bid_heur_best, bid_heur_sample, bid_pass, bid_random, bootstrap_ci,
                     load_deals, run_auctions)

N = 1000
POLICIES = [("zawsze pas", bid_pass),
            ("losowanie jednostajne", bid_random),
            ("prosty system, najlepsza odzywka", bid_heur_best),
            ("prosty system, losowanie", bid_heur_sample)]


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def main():
    P = Report("f1_miary")
    seeds, tab, vul, opt = load_deals(SEED_VALIDATION, N)
    mean_abs_opt = np.abs(opt).mean()
    P(f"{N} rozdan od {seeds[0]}, sredni |zapis optymalny| {mean_abs_opt:.1f}\n")

    W = {}
    for name, fn in POLICIES:
        r = run_auctions(fn, seeds, tab, vul)
        ns = np.array([x["score_ns"] for x in r])
        played = [x for x in r if not x["passed"]]
        W[name] = {"m1": np.array([x["score_dec"] for x in r]),
                   "m2": np.where(opt >= 0, opt - ns, ns - opt),
                   "m3": np.abs(opt - ns),
                   "passed": 100 * np.mean([x["passed"] for x in r]),
                   "level": np.mean([x["level"] for x in played]) if played else 0.0,
                   "doubled": 100 * np.mean([x["doubled"] or x["redoubled"] for x in played]) if played else 0.0}

    P("1. trzy miary")
    P("   %-34s %14s %14s %14s" % ("polityka", "M1 (wiecej+)", "M2 (mniej+)", "M3 (mniej+)"))
    for name, _ in POLICIES:
        w = W[name]
        P("   %-34s %8.1f+-%-4.0f %8.1f+-%-4.0f %8.1f+-%-4.0f"
          % (name, w["m1"].mean(), se(w["m1"]), w["m2"].mean(), se(w["m2"]), w["m3"].mean(), se(w["m3"])))

    P("\n2. losowanie jednostajne minus prosty system")
    for m in ("m2", "m3"):
        d = W["losowanie jednostajne"][m] - W["prosty system, najlepsza odzywka"][m]
        lo, hi, _ = bootstrap_ci(lambda i: d[i].mean(), len(d))
        P("   %s: %+.1f +- %.1f (t = %+.2f, przedzial %+.0f..%+.0f)"
          % (m.upper(), d.mean(), se(d), d.mean() / se(d), lo, hi))
    P("   M3 / sredni |zapis optymalny|:")
    for name, _ in POLICIES:
        P("     %-34s %.2f" % (name, W[name]["m3"].mean() / mean_abs_opt))

    P("\n3. profil polityk")
    P("   %-34s %8s %8s %8s" % ("polityka", "pasow", "poziom", "kontr"))
    for name, _ in POLICIES:
        w = W[name]
        P("   %-34s %7.1f%% %8.2f %7.1f%%" % (name, w["passed"], w["level"], w["doubled"]))

    a = W["prosty system, najlepsza odzywka"]["m3"]
    b = W["prosty system, losowanie"]["m3"]
    d = b - a
    lo, hi, _ = bootstrap_ci(lambda i: d[i].mean(), len(d))
    P("\n4. prosty system, M3: najlepsza odzywka %.1f, losowanie %.1f, roznica %+.1f +- %.1f (t = %+.2f, "
      "przedzial %+.0f..%+.0f)" % (a.mean(), b.mean(), d.mean(), se(d), d.mean() / se(d), lo, hi))
    P.close()


if __name__ == "__main__":
    main()
