"""Phase 1: contract distributions of experts and of optimal bidding."""
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wspolne import ROOT, SEED_TEST, SEED_VALIDATION, STRAINS_SHORT, Report, load_deals, save_figure
from bridge.env.bridge_env import duplicate_score
from bridge.licytacja.deal_value import par_score_ns

DATASET_CANDIDATES = [
    os.path.join(ROOT, "praca", "dane", "master_dataset.jsonl"),
    os.path.join(os.path.dirname(ROOT), "MAGISTERKA_DANE", "SUpervised_learning", "master_dataset.jsonl"),
    os.path.join(os.path.expanduser("~"), "Downloads", "master_dataset.jsonl"),
]
SUIT_SYMBOLS = {"♣": 0, "♦": 1, "♥": 2, "♠": 3, "NT": 4}
SEAT_INDEX = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}
RANKS = "23456789TJQKA"
STRAIN_ORDER = (4, 3, 2, 1, 0)
SEAT_ORDER = (0, 2, 1, 3)


def parse_call(t):
    t = t.replace("!*", "").strip()
    if t in ("P", "X", "XX"):
        return (t,)
    return ("bid", int(t[0]), SUIT_SYMBOLS[t[1:]])


def parse_contract(rec):
    d0 = SEAT_INDEX[rec["dealer"]]
    calls = [parse_call(t) for t in rec["auction"]]
    last_bid_idx = None
    for i, c in enumerate(calls):
        if c[0] == "bid":
            last_bid_idx = i
    if last_bid_idx is None:
        return None
    _, level, strain = calls[last_bid_idx]
    side = (d0 + last_bid_idx) % 2
    declarer = next((d0 + i) % 4 for i, c in enumerate(calls)
                    if c[0] == "bid" and c[2] == strain and (d0 + i) % 2 == side)
    after = [c[0] for c in calls[last_bid_idx + 1:]]
    return level, strain, declarer, "X" in after, "XX" in after


def parse_vul(v):
    return {"NONE": (False, False), "NS": (True, False), "EW": (False, True),
            "BOTH": (True, True), "ALL": (True, True)}[v.upper()]


def parse_score(s):
    side, w = s.split()
    return int(w) if side == "NS" else -int(w)


def decode_dd_table(hexs, seats=SEAT_ORDER):
    v = [int(c, 16) for c in hexs]
    t = np.zeros((5, 4), dtype=np.int8)
    for a, m in enumerate(STRAIN_ORDER):
        for b, s in enumerate(seats):
            t[m, s] = v[b * 5 + a]
    return t


def parse_card(t):
    return "♣♦♥♠".index(t[0]) * 13 + RANKS.index(t[1:].replace("10", "T"))


def parse_hand(s):
    return {suit * 13 + RANKS.index(r) for suit, ranks in zip((3, 2, 1, 0), s.split(".")) for r in ranks}


def card_seats(rec):
    k = parse_contract(rec)
    if k is None:
        return {}
    strain, declarer = k[1], k[2]
    leader, out = (declarer + 1) % 4, {}
    z = [parse_card(t) for t in rec["play"]]
    for t in range(0, len(z), 4):
        trick = [((leader + i) % 4, c) for i, c in enumerate(z[t:t + 4])]
        out.update({c: s for s, c in trick})
        if len(trick) < 4:
            break
        winner = trick[0]
        for s, c in trick[1:]:
            if ((c // 13 == winner[1] // 13 and c % 13 > winner[1] % 13)
                    or (c // 13 != winner[1] // 13 and c // 13 == strain)):
                winner = (s, c)
        leader = winner[0]
    return out


def group_deals(recs):
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["dealer"], r["vulnerability"], r["initial_hand"], r["double_dummy_tricks"])].append(r)
    out = []
    for key, rs in groups.items():
        card_seat = {}
        for r in rs:
            card_seat.update(card_seats(r))
        initial_hand = parse_hand(key[2])
        ms = {card_seat[c] for c in initial_hand if c in card_seat}
        player = ms.pop() if len(ms) == 1 else None
        hands = None
        if player is not None:
            for c in initial_hand:
                card_seat[c] = player
            rr = [set() for _ in range(4)]
            for c, s in card_seat.items():
                rr[s].add(c)
            missing = set(range(52)) - set(card_seat)
            incomplete = [s for s in range(4) if len(rr[s]) < 13]
            if missing and len(incomplete) == 1 and len(rr[incomplete[0]]) + len(missing) == 13:
                rr[incomplete[0]] |= missing
                missing = set()
            if not missing and all(len(h) == 13 for h in rr):
                hands = rr
        out.append({"key": key, "auctions": rs, "player": player, "hands": hands})
    return out


def seat_order_from(player):
    return player, (player + 2) % 4, (player + 1) % 4, (player + 3) % 4


def vul_at_table(v, player):
    z = parse_vul(v)
    return z[(0 - player) % 2], z[(1 - player) % 2]


def load_expert_records():
    fname = next(k for k in DATASET_CANDIDATES if os.path.exists(k))
    with open(fname, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    P = Report("f1_rozklady")
    recs = load_expert_records()
    contracts = [k for k in map(parse_contract, recs) if k is not None]
    n = len(contracts)
    expert_level = {l: 100 * sum(k[0] == l for k in contracts) / n for l in range(1, 8)}
    expert_strain = {m: 100 * sum(k[1] == m for k in contracts) / n for m in range(5)}
    n_deals = len({(r["dealer"], r["vulnerability"], r["double_dummy_tricks"]) for r in recs})
    P(f"1. eksperci: {len(recs)} licytacji, {n_deals} rozdan")
    P(f"   spasowanych {100 * (len(recs) - n) / len(recs):.1f}%, "
      f"kontr {100 * sum(k[3] and not k[4] for k in contracts) / n:.1f}%, "
      f"rekontr {100 * sum(k[4] for k in contracts) / n:.1f}%, sredni poziom {np.mean([k[0] for k in contracts]):.2f}")

    opt_dist = {}
    for name, z0 in [("validation", SEED_VALIDATION), ("test", SEED_TEST)]:
        _, tab, vul, _ = load_deals(z0, 1000)
        b = np.array([par_score_ns(t, tuple(z))[1] for t, z in zip(tab, vul)])
        b = b[b >= 0]
        level, strain = b // 5 + 1, b % 5
        opt_dist[name] = ({l: 100 * np.mean(level == l) for l in range(1, 8)},
                          {m: 100 * np.mean(strain == m) for m in range(5)}, level.mean())

    P("\n2. rozklad kontraktow (%)")
    P("   %-8s %10s %14s %14s" % ("poziom", "eksperci", "opt. walid.", "opt. wydziel."))
    for l in range(1, 8):
        P("   %-8d %9.1f%% %13.1f%% %13.1f%%"
          % (l, expert_level[l], opt_dist["validation"][0][l], opt_dist["test"][0][l]))
    P("   %-8s %10.2f %14.2f %14.2f" % ("srednio", np.mean([k[0] for k in contracts]), opt_dist["validation"][2],
                                      opt_dist["test"][2]))
    P("   %-8s %10s %14s %14s" % ("miano", "eksperci", "opt. walid.", "opt. wydziel."))
    for m in range(5):
        P("   %-8s %9.1f%% %13.1f%% %13.1f%%" % (STRAINS_SHORT[m], expert_strain[m], opt_dist["validation"][1][m],
                                             opt_dist["test"][1][m]))

    unique_deals = list({(r["double_dummy_tricks"], r["vulnerability"]): r for r in reversed(recs)}.values())
    matching = sum(par_score_ns(decode_dd_table(r["double_dummy_tricks"]), parse_vul(r["vulnerability"]))[0]
                   == parse_score(r["optimum_score"]) for r in unique_deals)
    losses, used, skipped = [], [], 0
    for rd in group_deals(recs):
        if rd["player"] is None:
            skipped += len(rd["auctions"])
            continue
        for r in rd["auctions"]:
            t = decode_dd_table(r["double_dummy_tricks"])
            vul = parse_vul(r["vulnerability"])
            k = parse_contract(r)
            ns = 0
            if k is not None:
                level, strain, declarer, x, xx = k
                rel_seat = (declarer - rd["player"]) % 4
                s = duplicate_score(level, strain, x and not xx, xx, vul[rel_seat % 2], int(t[strain, rel_seat]))
                ns = s if rel_seat % 2 == 0 else -s
            losses.append(abs(parse_score(r["optimum_score"]) - ns))
            used.append(r)
    losses = np.array(losses, float)
    mean_abs_opt = np.mean([abs(parse_score(r["optimum_score"])) for r in used])
    P("\n3. strata ekspertow |zapis optymalny - zapis|")
    P(f"   zapis optymalny zgodny z plikiem: {matching} z {len(unique_deals)} rozdan")
    P(f"   licytacji pominietych: {skipped}")
    P(f"   strata {losses.mean():.1f} +- {losses.std(ddof=1) / np.sqrt(len(losses)):.1f} ({len(losses)} licytacji), "
      f"trafienie w optimum {100 * np.mean(losses == 0):.1f}%")
    P(f"   sredni |zapis optymalny| {mean_abs_opt:.1f}, ulamek {losses.mean() / mean_abs_opt:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(1, 8)
    axes[0].bar(x - 0.2, [expert_level[l] for l in x], 0.4, label="eksperci", color="#4c72b0")
    axes[0].bar(x + 0.2, [opt_dist["test"][0][l] for l in x], 0.4, label="licytacja optymalna", color="#dd8452")
    axes[0].set_xlabel("poziom kontraktu")
    axes[0].set_ylabel("% kontraktów")
    axes[0].set_xticks(x)
    axes[0].legend(frameon=False)
    x = np.arange(5)
    axes[1].bar(x - 0.2, [expert_strain[m] for m in x], 0.4, color="#4c72b0")
    axes[1].bar(x + 0.2, [opt_dist["test"][1][m] for m in x], 0.4, color="#dd8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["trefl", "karo", "kier", "pik", "bez atu"])
    axes[1].set_xlabel("miano")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "f1_rozklady")
    P.close()


if __name__ == "__main__":
    main()
