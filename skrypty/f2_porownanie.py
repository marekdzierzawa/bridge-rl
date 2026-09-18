"""Phase 2: the phase 0 play network against the phase 2 network on the same deals."""
import numpy as np

from wspolne import MODEL_F0, SEED_TEST, Report, bootstrap_ci, play_deals

MODEL_F2 = "models/play_dmc_faza2_model.pth"
N = 1000
CONTRACT_SOURCES = [("sztuczna licytacja", None),
                    ("Deep CFR", ("model", "models/best_policy.pth")),
                    ("aktor-krytyk", ("model", "models/best_actor_final.pth"))]
DECLARER_ROLES = ("rozgrywajacy", "dziadek")
ACCURACY_GROUPS = [("wist", ("wist",)), ("obrona bez wistu", ("obrona",)), ("strona rozgrywajaca", DECLARER_ROLES)]


def metrics(recs):
    out = {}
    for r in recs:
        e = r["errors"]
        m = {"nt": r["strain"] == 4,
             "dec": sum(b for role, _, b, _ in e if role in DECLARER_ROLES),
             "def": sum(b for role, _, b, _ in e if role in ("wist", "obrona")),
             "lead": sum(b for role, _, b, _ in e if role == "wist")}
        for name, roles in ACCURACY_GROUPS:
            significant = [b for role, _, b, i in e if role in roles and i]
            m["t_" + name] = (sum(1 for b in significant if b == 0), len(significant))
        out[r["seed"]] = m
    return out


def mean_ci(x):
    lo, hi, _ = bootstrap_ci(lambda i: x[i].mean(), len(x))
    return x.mean(), lo, hi


def accuracy(t, idx=None):
    t = t if idx is None else t[idx]
    return 100.0 * t[:, 0].sum() / max(1, t[:, 1].sum())


def main():
    P = Report("f2_porownanie")
    seeds = list(range(SEED_TEST, SEED_TEST + N))
    P(f"faza 0: {MODEL_F0}, faza 2: {MODEL_F2}, {N} rozdan {seeds[0]}..{seeds[-1]}")
    P("roznica = faza 2 minus faza 0\n")

    summary = []
    for num, (source, bidding) in enumerate(CONTRACT_SOURCES, 1):
        m0 = metrics(play_deals(seeds, "net", "net", True, MODEL_F0, bidding=bidding))
        m2 = metrics(play_deals(seeds, "net", "net", True, MODEL_F2, bidding=bidding))
        common = sorted(set(m0) & set(m2))
        n = len(common)
        A = [m0[s] for s in common]
        B = [m2[s] for s in common]
        is_nt = np.array([a["nt"] for a in A])

        P(f"{num}. kontrakty: {source}")
        P(f"   z kontraktem: {n} (spasowanych {N - n}), bez atu {100 * is_nt.mean():.1f}%")
        P("   %-34s %22s %22s %24s" % ("miara [lewy na rozdanie]", "faza 0", "faza 2", "roznica"))
        rows = {}
        for key, desc in [("dec", "strata rozgrywajacego"), ("def", "strata obrony"), ("lead", "strata wistu")]:
            for mask, suffix in [(None, ""), (is_nt, ", bez atu"), (~is_nt, ", kolory")]:
                if key == "lead" and mask is not None:
                    continue
                x0 = np.array([a[key] for a in A], dtype=float)
                x2 = np.array([b[key] for b in B], dtype=float)
                if mask is not None:
                    x0, x2 = x0[mask], x2[mask]
                s0, s2, d = mean_ci(x0), mean_ci(x2), mean_ci(x2 - x0)
                P("   %-34s %6.3f (%5.3f-%5.3f) %6.3f (%5.3f-%5.3f) %+7.3f (%+6.3f;%+6.3f)"
                  % (desc + suffix, *s0, *s2, *d))
                rows[key + suffix] = d

        P("   %-34s %22s %22s %24s" % ("trafnosc [% decyzji istotnych]", "faza 0", "faza 2", "roznica [pkt proc.]"))
        for name, _ in ACCURACY_GROUPS:
            t0 = np.array([a["t_" + name] for a in A], dtype=float)
            t2 = np.array([b["t_" + name] for b in B], dtype=float)
            lo0, hi0, _ = bootstrap_ci(lambda i: accuracy(t0, i), n)
            lo2, hi2, _ = bootstrap_ci(lambda i: accuracy(t2, i), n)
            lo_d, hi_d, _ = bootstrap_ci(lambda i: accuracy(t2, i) - accuracy(t0, i), n)
            d = accuracy(t2) - accuracy(t0)
            P("   %-34s %5.1f%% (%4.1f-%4.1f) %5.1f%% (%4.1f-%4.1f) %+6.1f (%+5.1f;%+5.1f)"
              % (name, accuracy(t0), lo0, hi0, accuracy(t2), lo2, hi2, d, lo_d, hi_d))
            rows["t_" + name] = (d, lo_d, hi_d)
        P("")
        summary.append((source, rows))

    P("podsumowanie (faza 2 minus faza 0)")
    P("   %-24s %22s %22s %22s" % ("zrodlo kontraktow", "strata rozgrywajacego", "strata obrony", "trafnosc wistu"))
    for source, w in summary:
        P("   %-24s %+6.3f (%+6.3f;%+6.3f) %+6.3f (%+6.3f;%+6.3f) %+5.1f (%+5.1f;%+5.1f)"
          % (source, *w["dec"], *w["def"], *w["t_wist"]))
    P.close()


if __name__ == "__main__":
    main()
