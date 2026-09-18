"""Phase 0: networks with the old and the new card description compared."""
import numpy as np

from wspolne import MODEL_F0, MODEL_F0_OLD, SEED_F0, Report, play_deals
from f0_audyt import DECLARER_ROLES, DEFENCE_ROLES, error_tables, nt_excess

N = 1200


def hits(recs, roles):
    t = np.zeros(len(recs))
    n = np.zeros(len(recs))
    for i, r in enumerate(recs):
        for role, _, b, significant in r["errors"]:
            if role in roles and significant:
                n[i] += 1
                t[i] += b == 0
    return t, n


def main():
    P = Report("f0_stare_nowe_cechy")
    seeds = list(range(SEED_F0, SEED_F0 + N))
    new_recs = play_deals(seeds, "net", "net", True, MODEL_F0)
    old_recs = play_deals(seeds, "net", "net", True, MODEL_F0_OLD)
    common = set(r["seed"] for r in new_recs) & set(r["seed"] for r in old_recs)
    new_recs = [r for r in new_recs if r["seed"] in common]
    old_recs = [r for r in old_recs if r["seed"] in common]
    n = len(new_recs)
    P(f"stara: {MODEL_F0_OLD}, nowa: {MODEL_F0}, rozdania {seeds[0]}+, wspolnych: {n}\n")

    rng = np.random.default_rng(0)
    resamples = [rng.integers(0, n, n) for _ in range(2000)]
    P("%-26s %9s %9s %10s %22s" % ("miara", "stara", "nowa", "roznica", "przedzial 95%"))
    for name, roles in [("trafnosc: wist", ("wist",)), ("trafnosc: reszta obrony", ("obrona",)),
                        ("trafnosc: rozgrywajacy", DECLARER_ROLES)]:
        ts, ns = hits(old_recs, roles)
        tn, nn = hits(new_recs, roles)
        s = 100 * ts.sum() / ns.sum()
        w = 100 * tn.sum() / nn.sum()
        d = [100 * (tn[i].sum() / nn[i].sum() - ts[i].sum() / ns[i].sum()) for i in resamples]
        lo, hi = np.percentile(d, [2.5, 97.5])
        P("%-26s %8.1f%% %8.1f%% %+9.1f pp   [%+6.1f, %+6.1f] pp" % (name, s, w, w - s, lo, hi))

    for name, roles in [("nadwyzka BA: obrona", DEFENCE_ROLES), ("nadwyzka BA: rozgrywajacy", DECLARER_ROLES)]:
        Ts, Tn = error_tables(old_recs, roles), error_tables(new_recs, roles)
        s, w = nt_excess(*Ts), nt_excess(*Tn)
        d = [nt_excess(*Tn, idx=i) - nt_excess(*Ts, idx=i) for i in resamples]
        lo, hi = np.percentile(d, [2.5, 97.5])
        P("%-26s %9.3f %9.3f %+10.3f   [%+6.3f, %+6.3f] lewy" % (name, s, w, w - s, lo, hi))
    P.close()


if __name__ == "__main__":
    main()
