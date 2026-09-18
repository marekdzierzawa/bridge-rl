"""Phase 1: numbers for the actor-critic (CTDE) section."""
import numpy as np
import torch

from wspolne import SEED_TEST, Report, bid_heur_best, load_deals, run_auctions
from bridge.env.bridge_env import PASS, BridgeEnv, duplicate_score
from bridge.licytacja.deal_value import par_score_ns
from bridge.licytacja.train_ctde import PASS_BIAS, _partner_features, validate
from bridge.nets.ctde import BidActor
from bridge.ocena.wzorce import repertoire


def restricted_loss(tab, vul, opt, allowed):
    out = []
    for t, z, o in zip(tab, vul, opt):
        best = abs(o)
        for level, strain in allowed:
            for declarer in range(4):
                s = duplicate_score(level, strain, False, False, bool(z[declarer % 2]), int(t[strain, declarer]))
                best = min(best, abs(o - (s if declarer % 2 == 0 else -s)))
        out.append(best)
    a = np.array(out)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def main():
    P = Report("f1_ctde")
    seeds, tab, vul, opt = load_deals(SEED_TEST, 1000)

    P("1. swiezy aktor, 300 rozdan")
    P("   %-30s %10s %8s %20s %12s" % ("wariant", "entropia", "pasow", "najczestszy kontrakt", "na 80% rozd."))
    for name, bias in [("bez przechylenia", 0.0), (f"przechylenie w strone pasa {PASS_BIAS}", PASS_BIAS)]:
        torch.manual_seed(0)
        a = BidActor().eval()
        with torch.no_grad():
            a.glowa_polityki[-1].bias[PASS] += bias
        r = validate(a, tab[:300], vul[:300], opt[:300], seeds[:300], temp=0.0)
        rep = r["repertoire"] or {}
        P("   %-30s %9.2f %7.1f%% %19s %12s" % (name, r["entropy"], r["pass"],
                                               "%.0f%%" % rep["top"] if rep else "-", rep.get("n80", "-")))
    P("   entropia rozkladu jednostajnego: %.2f" % np.log(38))

    P(f"\n2. najmniejsza strata przy ograniczonym zestawie kontraktow ({len(seeds)} rozdan od {seeds[0]}, "
      f"sredni |optymalny| {np.abs(opt).mean():.1f})")
    for desc, allowed in [("3 BA, 4 kier, 1 pik", [(3, 4), (4, 2), (1, 3)]),
                          ("wszystkie 35 kontraktow", [(l, m) for l in range(1, 8) for m in range(5)])]:
        P("   %-34s %6.1f +- %.1f" % (desc, *restricted_loss(tab, vul, opt, allowed)))
    st = np.abs(opt - np.array([x["score_ns"] for x in run_auctions(bid_heur_best, seeds, tab, vul)]))
    P("   %-34s %6.1f +- %.1f" % ("prosty system", st.mean(), st.std(ddof=1) / np.sqrt(len(st))))
    P("   %-34s %6.1f" % ("zawsze pas", np.abs(opt).mean()))
    par_bids = [b for b in (par_score_ns(t, tuple(z))[1] for t, z in zip(tab, vul)) if b >= 0]
    P(f"   optymalny kontrakt: 4 piki {100 * np.mean([b == 18 for b in par_bids]):.1f}%, "
      f"4 kiery {100 * np.mean([b == 17 for b in par_bids]):.1f}%")
    rep = repertoire([(b // 5 + 1, b % 5) for b in par_bids])
    P(f"   repertuar optymalny: najczestszy {rep['top']:.1f}%, na 80% {rep['n80']} kontraktow, "
      f"roznych {rep['distinct']}")

    X = []
    for sd in range(3_000_000, 3_004_000):
        e = BridgeEnv(seed=sd)
        e.reset()
        X.append(_partner_features(e, 0))
    X = np.array(X)
    w = X.var(axis=0)
    P("\n3. wariancja celow glowicy partnera (4000 rozdan)")
    for name, v in zip(["punkty/10", "trefle/13", "kara/13", "kiery/13", "piki/13"], w):
        P("   %-10s %.4f  %5.1f%%" % (name, v, 100 * v / w.sum()))
    P(f"   sila {100 * w[0] / w.sum():.1f}%, ksztalt {100 * w[1:].sum() / w.sum():.1f}%, "
      f"starsze {100 * w[3:].sum() / w.sum():.1f}%")
    P(f"   odchylenie: punkty {X[:, 0].std() * 10:.2f}, dlugosc koloru {X[:, 1:].std() * 13:.2f}")
    P.close()


if __name__ == "__main__":
    main()
