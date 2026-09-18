"""Phase 1 training curves plotted from the validation history file."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bridge.ocena.historia import column, load_history
from bridge.ocena.wzorce import EXPERT_MEAN_LEVEL, EXPERT_STRAIN, HEURISTIC_LOSS, PAR_REPERTOIRE_80

DEFAULT_FILES = {"ctde": "checkpoints/ctde/historia.csv",
                 "cfr": "checkpoints/phase1b/historia.csv"}
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]


def _style_axes(ax, title, ylab):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("epoka", fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)


def _panel_loss(ax, histories, labels):
    for (h, label, c) in zip(histories, labels, COLORS):
        e, v = column(h, "strata")
        if not e:
            continue
        e, v = np.array(e), np.array(v)
        _, se = column(h, "strata_se")
        ax.plot(e, v, color=c, lw=1.6, label=label)
        if se:
            se = np.array(se)
            ax.fill_between(e, v - se, v + se, color=c, alpha=0.18, lw=0)
        _, pass_loss = column(h, "strata_pas")
        if pass_loss:
            ax.axhline(float(np.mean(pass_loss)), color="#888", ls=":", lw=1,
                       label="zawsze pas" if label == labels[0] else None)
    ax.axhline(HEURISTIC_LOSS, color="#555", ls="--", lw=1,
               label="prosty system")
    peak, clipped = 0.0, False
    for h in histories:
        _, v = column(h, "strata")
        _, pass_loss = column(h, "strata_pas")
        threshold = 3.0 * (float(np.mean(pass_loss)) if pass_loss else HEURISTIC_LOSS)
        w = [x for x in v if x <= threshold]
        if w:
            peak = max(peak, max(w))
        if v and max(v) > threshold:
            clipped = True
    if peak > 0:
        ax.set_ylim(0, peak * 1.1)
    _style_axes(ax, "1. Strata wobec optymalnego wyniku (mniej = lepiej)"
              + ("  [os obcieta u gory]" if clipped else ""),
                "punkty na rozdanie")
    ax.legend(fontsize=7, framealpha=0.9)


def _panel_level(ax, histories, labels):
    for (h, label, c) in zip(histories, labels, COLORS):
        e, v = column(h, "poziom")
        if e:
            ax.plot(e, v, color=c, lw=1.6, label=label)
    ax.axhline(EXPERT_MEAN_LEVEL, color="#555", ls="--", lw=1,
               label="eksperci %.2f" % EXPERT_MEAN_LEVEL)
    _style_axes(ax, "2. Sredni poziom kontraktu", "poziom")
    ax.legend(fontsize=7)


def _panel_level_dist(ax, h, label):
    e = [w["epoka"] for w in h if "poziom1" in w]
    if not e:
        _style_axes(ax, "3. Rozklad poziomow - brak danych", "%")
        return
    layers = np.array([[w.get(f"poziom{l}", 0.0) for w in h if "poziom1" in w]
                       for l in range(1, 8)])
    ax.stackplot(e, layers, labels=[f"{l}" for l in range(1, 8)],
                 colors=plt.cm.viridis(np.linspace(0.1, 0.95, 7)), alpha=0.9)
    ax.set_ylim(0, 100)
    _style_axes(ax, f"3. Rozklad poziomow ({label})", "% kontraktow")
    ax.legend(fontsize=6, ncol=7, loc="upper center", framealpha=0.85,
              columnspacing=0.8, handlelength=1.0)


def _panel_strain(ax, histories, labels):
    for (h, label, c) in zip(histories, labels, COLORS):
        e, v = column(h, "ba")
        if e:
            ax.plot(e, v, color=c, lw=1.6, label=f"bez atu ({label})")
        e2, v2 = column(h, "pas")
        if e2:
            ax.plot(e2, v2, color=c, lw=1.0, ls=":", label=f"pasow ({label})")
    ax.axhline(EXPERT_STRAIN["NT"], color="#555", ls="--", lw=1,
               label="bez atu u ekspertow %.1f%%" % EXPERT_STRAIN["NT"])
    _style_axes(ax, "4. Udzial bez atu i rozdan spasowanych", "%")
    ax.legend(fontsize=7)


def _panel_collapse(ax, histories, labels):
    ax2 = ax.twinx()
    for (h, label, c) in zip(histories, labels, COLORS):
        e, v = column(h, "entropia")
        if e:
            ax.plot(e, v, color=c, lw=1.6, label=f"entropia ({label})")
        e2, v2 = column(h, "rep_na80")
        if e2:
            ax2.plot(e2, v2, color=c, lw=1.2, ls="--",
                     label=f"kontraktow na 80% ({label})")
    ax2.axhline(PAR_REPERTOIRE_80, color="#555", ls=":", lw=1)
    ax2.set_ylabel("kontraktow na 80%% rozdan (optymalna: %d)" % PAR_REPERTOIRE_80,
                   fontsize=8)
    ax2.tick_params(labelsize=8)
    _style_axes(ax, "5. Zapasc: entropia (lewa) i szerokosc repertuaru (prawa)",
                "entropia [nat]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)


def _panel_method(ax, histories, labels):
    plotted = False
    for (h, label, c) in zip(histories, labels, COLORS):
        e, v = column(h, "krytyk_r2")
        if e:
            ax.plot(e, v, color=c, lw=1.6, label=f"krytyk R2 ({label})")
            plotted = True
        e2, v2 = column(h, "konwencja")
        if e2:
            ax.plot(e2, np.array(v2) / 4.0, color=c, lw=1.2, ls="--",
                    label=f"zysk z licytacji /4 [pkt] ({label})")
            plotted = True
        e3, v3 = column(h, "strata_t1")
        e4, v4 = column(h, "strata")
        if e3 and e4 and len(e3) == len(e4):
            ax.plot(e3, (np.array(v3) - np.array(v4)) / 1000.0, color=c,
                    lw=1.2, ls="-.",
                    label=f"rozjazd T=1 minus argmax /1000 ({label})")
            plotted = True
    ax.axhline(0, color="#555", lw=0.8)
    _style_axes(ax, "6. Zdrowie metody (skale znormalizowane)", "")
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "brak metryk specyficznych dla metody",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)


def plot_history(paths, labels, out, panels=(1, 2, 3, 4, 5, 6), title=True):
    histories = [load_history(p) for p in paths]
    panel_fns = {
        1: lambda ax: _panel_loss(ax, histories, labels),
        2: lambda ax: _panel_level(ax, histories, labels),
        3: lambda ax: _panel_level_dist(ax, histories[0], labels[0]),
        4: lambda ax: _panel_strain(ax, histories, labels),
        5: lambda ax: _panel_collapse(ax, histories, labels),
        6: lambda ax: _panel_method(ax, histories, labels),
    }
    n_cols = 1 if len(panels) == 1 else 2
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.0 * n_rows), squeeze=False)
    if title:
        fig.suptitle("Faza 1: przebieg treningu  (%s)" % " | ".join(labels), fontsize=12)
    for i, n in enumerate(panels):
        panel_fns[n](axes[i // n_cols][i % n_cols])
    for i in range(len(panels), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    fig.savefig(out, dpi=130)
    if out.endswith(".png"):
        fig.savefig(out[:-4] + ".pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ctde", "cfr"])
    ap.add_argument("--history")
    ap.add_argument("--compare")
    ap.add_argument("--labels")
    ap.add_argument("--out")
    ap.add_argument("--panels", default="1,2,3,4,5,6")
    ap.add_argument("--no-title", action="store_true")
    a = ap.parse_args()

    paths = [p for p in (a.history, DEFAULT_FILES.get(a.method), a.compare) if p]
    paths = paths or [p for p in DEFAULT_FILES.values() if os.path.exists(p)]
    labels = a.labels.split(",") if a.labels else [os.path.basename(os.path.dirname(p)) for p in paths]
    out = a.out or os.path.splitext(paths[0])[0] + ".png"
    plot_history(paths, labels, out, tuple(int(x) for x in a.panels.split(",")), not a.no_title)


if __name__ == "__main__":
    main()
