"""Phase 1: the phase 0 network as a trick oracle compared with the solver."""
import concurrent.futures
import multiprocessing as mp
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wspolne import (MODEL_F0, MODEL_F0_OLD, SEED_VALIDATION, STRAIN_NAMES, Report, bootstrap_ci, load_f0_net,
                     save_figure)
from bridge.dd.dd_table import dd_tricks_tables
from bridge.env.bridge_env import PLAY, BridgeEnv, Trick
from bridge.licytacja.deal_value import par_score_ns
from bridge.ocena.zbior_walidacyjny import load_dd_tables

N = 150
LEVEL = 4
CALIBRATION = np.array([0.518, 0.360, 0.400, 0.470, 1.072])
_NETS = {}


def _init(path):
    torch.set_num_threads(1)
    _NETS["dec"], _NETS["dfn"], _NETS["feats"] = load_f0_net(path)


def _play_out(env):
    side = env.declarer % 2
    while not env.terminated:
        p = env.current_player
        legal = env.legal_actions()
        net = _NETS["dec"] if p % 2 == side else _NETS["dfn"]
        with torch.no_grad():
            q = net.q_all(torch.from_numpy(env.observe(p)), torch.from_numpy(_NETS["feats"](env, legal)))
        env.step(int(legal[int(torch.argmax(q).item())]))


def oracle_table(sd):
    base = BridgeEnv(seed=sd)
    base.reset()
    t = np.zeros((5, 4), dtype=np.int8)
    for strain in range(5):
        for declarer in range(4):
            e = BridgeEnv()
            e.reset(dealer=base.dealer, vulnerable=base.vulnerable, hands=[set(h) for h in base.hands])
            e.phase = PLAY
            e.level, e.strain, e.declarer, e.dummy = LEVEL, strain, declarer, (declarer + 2) % 4
            e.current_trick = Trick(leader=(declarer + 1) % 4)
            _play_out(e)
            t[strain, declarer] = e.tricks_won[declarer % 2]
    return t


def _oracle_batch(seeds):
    return [oracle_table(sd) for sd in seeds]


def compute_oracle(path, seeds, workers=8):
    chunks = [seeds[i::workers] for i in range(workers)]
    tab = np.zeros((len(seeds), 5, 4), dtype=np.int8)
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                                initargs=(path,)) as ex:
        for i, w in enumerate(ex.map(_oracle_batch, chunks)):
            tab[i::workers] = np.array(w)
    return tab


def optimal_strain(tab, vul):
    b = np.array([par_score_ns(np.asarray(t), tuple(z))[1] for t, z in zip(tab, vul)])
    return np.where(b < 0, -1, b % 5)


def calibrate(tab, seed=0):
    rng = np.random.default_rng(seed)
    x = tab.astype(float) - CALIBRATION[None, :, None]
    floor_x = np.floor(x)
    return np.clip(floor_x + (rng.random(x.shape) < (x - floor_x)), 0, 13).astype(np.int8)


def main():
    P = Report("f1_wyrocznia")
    seeds = list(range(SEED_VALIDATION, SEED_VALIDATION + N))
    tab_dd, vul, _ = load_dd_tables(list(range(SEED_VALIDATION, SEED_VALIDATION + 1000)))
    tab_dd, vul = tab_dd[:N], vul[:N]
    P(f"wyrocznia fazy 0 wobec gry idealnej: {N} rozdan od {seeds[0]}, poziom {LEVEL}\n")

    tables = {"stara": compute_oracle(MODEL_F0_OLD, seeds), "nowa": compute_oracle(MODEL_F0, seeds)}
    tables["stara, skalibrowana"] = calibrate(tables["stara"])

    P("1. srednia liczba lew (roznica wobec idealnej i przedzial 95%)")
    P("   %-9s %8s | %-24s | %-24s" % ("miano", "idealna", "wyrocznia stara", "wyrocznia nowa"))
    bias = {"stara": [], "nowa": []}
    for m in range(5):
        dd = tab_dd[:, m, :].mean(axis=1)
        cells = []
        for k in ("stara", "nowa"):
            w = tables[k][:, m, :].mean(axis=1)
            d = w - dd
            lo, hi, _ = bootstrap_ci(lambda i: d[i].mean(), len(d))
            bias[k].append((d.mean(), lo, hi))
            cells.append("%5.2f  (%+.2f [%+.2f, %+.2f])" % (w.mean(), d.mean(), lo, hi))
        P("   %-9s %8.2f | %-24s | %-24s" % (STRAIN_NAMES[m], dd.mean(), cells[0], cells[1]))
    for k in ("stara", "nowa"):
        b = bias[k][4][0]
        suits_mean = float(np.mean([x[0] for x in bias[k][:4]]))
        P(f"   {k}: bez atu {b:+.2f}, kolory {suits_mean:+.2f}, roznica {b - suits_mean:+.2f}")

    P("\n2. miano licytacji optymalnej")
    opt_dd = optimal_strain(tab_dd, vul)
    P("   %-22s %s" % ("tabela", "  ".join("%7s" % STRAIN_NAMES[m] for m in range(5))))
    for name, t in [("gra idealna", tab_dd), ("wyrocznia stara", tables["stara"]),
                    ("stara skalibrowana", tables["stara, skalibrowana"]), ("wyrocznia nowa", tables["nowa"])]:
        o = optimal_strain(t, vul)
        n = max((o >= 0).sum(), 1)
        P("   %-22s %s" % (name, "  ".join("%6.1f%%" % (100 * (o == m).sum() / n) for m in range(5))))

    P("\n3. trafnosc na rozdaniu")
    P("   %-22s %16s %16s %18s" % ("", "komorek dokladnie", "sredni blad", "miano optymalne"))
    for name in ("stara", "stara, skalibrowana", "nowa"):
        t = tables[name]
        P("   %-22s %15.1f%% %16.2f %17.1f%%" % (name, 100 * np.mean(t == tab_dd),
                                              np.mean(np.abs(t.astype(int) - tab_dd.astype(int))),
                                              100 * np.mean(optimal_strain(t, vul) == opt_dd)))
    most_common_strain = np.bincount(opt_dd[opt_dd >= 0], minlength=5).argmax()
    P("   zawsze %s: %.1f%%" % (STRAIN_NAMES[most_common_strain], 100 * np.mean(opt_dd == most_common_strain)))

    hands = []
    for sd in seeds[:20]:
        e = BridgeEnv(seed=sd)
        e.reset()
        hands.append([set(h) for h in e.hands])
    t0 = time.perf_counter()
    dd_tricks_tables(hands)
    t_dd = (time.perf_counter() - t0) / 20
    _init(MODEL_F0)
    t0 = time.perf_counter()
    for sd in seeds[:20]:
        oracle_table(sd)
    t_w = (time.perf_counter() - t0) / 20
    P(f"\n4. czas tabeli 5x4: solver {t_dd:.2f} s, wyrocznia {t_w:.2f} s ({t_w / t_dd:.1f}x)")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(5)
    for off, k, color, label in [(-0.18, "stara", "#c44e52", "sieć z 40 cechami (wyrocznia fazy 1)"),
                                 (0.18, "nowa", "#dd8452", "sieć z 41 cechami (obecna)")]:
        m, lo, hi = (np.array(v) for v in zip(*bias[k]))
        ax.bar(x + off, m, 0.36, yerr=[m - lo, hi - m], capsize=3, color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(STRAIN_NAMES)
    ax.set_ylabel("lewy ponad grę idealną")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "f1_wyrocznia")
    P.close()


if __name__ == "__main__":
    main()
