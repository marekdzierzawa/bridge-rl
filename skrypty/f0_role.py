"""Phase 0: which side of the table loses tricks - the defence or the declarer."""
import numpy as np

from wspolne import MODEL_F0, SEED_F0, Report, play_deals

N = 600


def main():
    P = Report("f0_role")
    seeds = list(range(SEED_F0, SEED_F0 + N))
    matchups = {k: play_deals(seeds, d, o, False, MODEL_F0)
                for k, (d, o) in {"ss": ("net", "net"), "si": ("net", "dd"),
                                  "is": ("dd", "net"), "ii": ("dd", "dd")}.items()}
    common = set.intersection(*[set(r["seed"] for r in v) for v in matchups.values()])
    tricks = {k: np.array([r["tricks"] for r in v if r["seed"] in common], float) for k, v in matchups.items()}
    strain = np.array([r["strain"] for r in matchups["ii"] if r["seed"] in common])
    dd = np.array([r["dd_declarer"] for r in matchups["ii"] if r["seed"] in common], float)
    P(f"{MODEL_F0}, rozdania {seeds[0]}+, wspolnych: {len(common)}")
    P(f"idealny/idealny zgodny z tablica DD: {100 * np.mean(tricks['ii'] == dd):.1f}%\n")

    rng = np.random.default_rng(0)
    P("%-24s %24s %24s" % ("", "bez atu", "kolory"))
    for desc, a in [("blad obrony", "is"), ("blad rozgrywajacego", "si")]:
        row = []
        for mask in (strain == 4, strain != 4):
            d = (tricks[a] - tricks["ii"])[mask]
            lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])
            row.append("%+.2f [%+.2f, %+.2f]" % (d.mean(), lo, hi))
        P("%-24s %24s %24s" % (desc, row[0], row[1]))
    P(f"\nrozdan bez atu: {(strain == 4).sum()}, w kolorze: {(strain != 4).sum()}")
    P.close()


if __name__ == "__main__":
    main()
