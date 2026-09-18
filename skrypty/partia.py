"""A single deal played by the final models (.lin and .pbn records and a LaTeX table)."""
import argparse
import os
import urllib.parse

import numpy as np
import torch
from endplay.types import Deal

from wspolne import RESULTS_DIR, ROOT, SEED_TEST, Report, load_deals
from bridge.env.bridge_env import AUCTION, CARD_OFFSET, DOUBLE, PASS, REDOUBLE, BridgeEnv, action_str, duplicate_score
from bridge.env.play_features import legal_card_features
from bridge.licytacja.deal_value import DealValue
from bridge.ocena.eval_dd import DENOM, PLAYER, dd_values, hands_to_pbn, load_nets, to_endplay_card
from bridge.ocena.evaluate_models import load_policy

ACTOR_PATH = "models/best_actor_final.pth"
CFR_PATH = "models/best_policy.pth"
SEATS = "NESW"
RANKS = "23456789TJQKA"
SUITS = "CDHS"
STRAINS_TXT = ["♣", "♦", "♥", "♠", "BA"]
LIN_DEALER = {2: 1, 3: 2, 0: 3, 1: 4}
SUITS_TEX = ["\\Trefl{}", "\\Karo{}", "\\Kier{}", "\\Pik{}"]
STRAINS_TEX = SUITS_TEX + ["BA"]
VUL_TEX = {(False, False): "nikt", (True, False): "NS", (False, True): "EW", (True, True): "obie strony"}


def suit_ranks(cards, s):
    return sorted((c % 13 for c in cards if c // 13 == s), reverse=True)


def call_text(a, pass_s, dbl_s, rdbl_s, strains):
    return {PASS: pass_s, DOUBLE: dbl_s, REDOUBLE: rdbl_s}.get(a) or f"{a // 5 + 1}{strains[a % 5]}"


def card_lin(c):
    return SUITS[c // 13] + RANKS[c % 13]


def run_deal(sd, logits, dec, dfn):
    env = BridgeEnv(seed=sd)
    env.reset()
    hands0 = [sorted(h) for h in env.hands]
    dealer, vul = env.dealer, tuple(env.vulnerable)
    calls = []
    while env.phase == AUCTION and not env.terminated and len(calls) < 60:
        calls.append(int(torch.argmax(logits(env)).item()))
        env.step(calls[-1])

    plays = []
    if not env.terminated and env.phase != AUCTION:
        side = env.declarer % 2
        deal = Deal(hands_to_pbn([set(h) for h in hands0]))
        deal.trump = DENOM[env.strain]
        deal.first = PLAYER[(env.declarer + 1) % 4]
        while not env.terminated:
            seat = env._seat_to_play()
            legal = env.legal_actions()
            net = dec if seat % 2 == side else dfn
            with torch.no_grad():
                q = net.q_all(torch.from_numpy(env.observe(env.current_player)).float(),
                              torch.from_numpy(legal_card_features(env, legal)).float())
            a = int(legal[int(torch.argmax(q).item())])
            cost = 0
            if len(legal) > 1:
                vals = dd_values(deal)
                best = max(vals.get(x - CARD_OFFSET, -1) for x in legal)
                cost = best - vals.get(a - CARD_OFFSET, best)
            plays.append((seat, a - CARD_OFFSET, cost))
            deal.play(to_endplay_card(a - CARD_OFFSET))
            env.step(a)
    return {"env": env, "hands0": hands0, "dealer": dealer, "vul": vul, "calls": calls, "plays": plays}


def lin(p, title, player):
    vul = {(False, False): "o", (True, False): "n", (False, True): "e", (True, True): "b"}
    hand_str = lambda cards: "".join("SHDC"[3 - s] + "".join(RANKS[r] for r in suit_ranks(cards, s))
                                     for s in (3, 2, 1, 0))
    parts = [f"pn|{player},{player},{player},{player}|", "st||",
             "md|%d%s|" % (LIN_DEALER[p["dealer"]], ",".join(hand_str(p["hands0"][m]) for m in (2, 3, 0, 1))),
             f"sv|{vul[p['vul']]}|", f"ah|{title}|"]
    parts += [f"mb|{call_text(a, 'p', 'd', 'r', 'CDHSN')}|" for a in p["calls"]]
    parts += [f"pc|{card_lin(c)}|" for _, c, _ in p["plays"]]
    return "".join(parts)


def pbn(p, board_no, title, optimum):
    env = p["env"]
    vul = {(False, False): "None", (True, False): "NS", (False, True): "EW", (True, True): "All"}
    hand_str = lambda cards: ".".join("".join(RANKS[r] for r in suit_ranks(cards, s)) for s in (3, 2, 1, 0))
    rows = [f'[Event "{title}"]', '[Site "?"]', '[Date "?"]', f'[Board "{board_no}"]',
            '[West "?"]', '[North "?"]', '[East "?"]', '[South "?"]',
            f'[Dealer "{SEATS[p["dealer"]]}"]', f'[Vulnerable "{vul[p["vul"]]}"]',
            '[Deal "N:%s"]' % " ".join(hand_str(p["hands0"][m]) for m in range(4)), '[Scoring "?"]']
    if env.declarer is None or not p["plays"]:
        rows += ['[Declarer ""]', '[Contract "Pass"]', '[Result ""]']
    else:
        dbl_suffix = "XX" if env.contract_redoubled else ("X" if env.contract_doubled else "")
        rows += [f'[Declarer "{SEATS[env.declarer]}"]',
                 f'[Contract "{env.level}{["C", "D", "H", "S", "NT"][env.strain]}{dbl_suffix}"]',
                 f'[Result "{env.tricks_won[env.declarer % 2]}"]']
    rows.append(f'[Auction "{SEATS[p["dealer"]]}"]')
    calls = [call_text(x, "Pass", "X", "XX", ["C", "D", "H", "S", "NT"]) for x in p["calls"]]
    rows += [" ".join(calls[k:k + 4]) for k in range(0, len(calls), 4)]
    if p["plays"]:
        leader = (env.declarer + 1) % 4
        rows.append(f'[Play "{SEATS[leader]}"]')
        z = p["plays"]
        for t in range(0, len(z), 4):
            trick = {m: card_lin(c) for m, c, _ in z[t:t + 4]}
            rows.append(" ".join(trick.get((leader + k) % 4, "-") for k in range(4)))
    rows.append('[OptimumScore "NS %d"]' % round(optimum))
    return "\n".join(rows)


def card_tex(c):
    return SUITS_TEX[c // 13] + RANKS[c % 13].replace("T", "10")


def trick_winner(trick, trump):
    led_suit = trick[0][1] // 13
    rank_key = lambda c: 200 + c % 13 if c // 13 == trump else (100 + c % 13 if c // 13 == led_suit else 0)
    return max(trick, key=lambda x: rank_key(x[1]))[0]


def hand_tex(seat, cards):
    rows = ["\\textbf{%s}" % SEATS[seat]]
    for s in (3, 2, 1, 0):
        rows.append(SUITS_TEX[s] + " " + (" ".join(RANKS[r].replace("T", "10") for r in suit_ranks(cards, s)) or "---"))
    return "\\begin{tabular}[c]{@{}l@{}}" + " \\\\ ".join(rows) + "\\end{tabular}"


def tex(p, first_diff, desc, optimum):
    env, w = p["env"], p["result"]
    r = [hand_tex(m, p["hands0"][m]) for m in range(4)]
    L = ["% wygenerowane przez praca/skrypty/partia.py -- nie edytowac recznie",
         "% " + desc,
         "\\providecommand{\\Pik}{\\ensuremath{\\spadesuit}}",
         "\\providecommand{\\Kier}{\\ensuremath{\\heartsuit}}",
         "\\providecommand{\\Karo}{\\ensuremath{\\diamondsuit}}",
         "\\providecommand{\\Trefl}{\\ensuremath{\\clubsuit}}",
         "\\begin{minipage}[c]{0.5\\textwidth}\\centering\\small",
         "\\begin{tabular}{@{}c@{\\hspace{6pt}}c@{\\hspace{6pt}}c@{}}",
         " & %s & \\\\[4pt]" % r[0],
         "%s & \\fbox{\\begin{tabular}{@{}c@{}}N\\\\ W\\hspace{1.2em}E\\\\ S\\end{tabular}} & %s \\\\[4pt]"
         % (r[3], r[1]),
         " & %s & \\\\" % r[2],
         "\\end{tabular}\\\\[4pt]",
         "rozdaje %s, po partii: %s" % (SEATS[p["dealer"]], VUL_TEX[p["vul"]]),
         "\\end{minipage}\\hfill",
         "\\begin{minipage}[c]{0.46\\textwidth}\\centering\\small",
         "\\begin{tabular}{cccc}", "\\toprule", "W & N & E & S \\\\", "\\midrule"]
    cells = [""] * ((p["dealer"] + 1) % 4) + [call_text(x, "pas", "ktr", "rktr", STRAINS_TEX) for x in p["calls"]]
    cells += [""] * (-len(cells) % 4)
    L += [" & ".join(cells[k:k + 4]) + " \\\\" for k in range(0, len(cells), 4)]
    L += ["\\bottomrule", "\\end{tabular}\\\\[8pt]",
          "\\begin{tabular}{@{}ll@{}}",
          "kontrakt & %d%s, rozgrywa %s \\\\" % (env.level, STRAINS_TEX[env.strain], SEATS[env.declarer]),
          "lewy w~rozgrywce sieci & %d (zapis NS $%+d$) \\\\" % (w["taken"], w["score"]),
          "lewy przy grze idealnej & %d (zapis NS $%+d$) \\\\" % (w["dd"], round(w["score_dd"])),
          "zapis optymalny & NS $%+d$ \\\\" % round(optimum),
          "\\end{tabular}",
          "\\end{minipage}", "", "\\medskip", "{\\small",
          "\\begin{tabular}{@{}r l l l l c c@{}}", "\\toprule",
          "lewa & 1. karta & 2. karta & 3. karta & 4. karta & bierze & NS--EW \\\\", "\\midrule"]
    z, tricks = p["plays"], [0, 0]
    for t in range(0, len(z), 4):
        winner = trick_winner([(m, c) for m, c, _ in z[t:t + 4]], env.strain)
        tricks[winner % 2] += 1
        cards = []
        for m, c, cost in z[t:t + 4]:
            k = "%s %s" % (SEATS[m], card_tex(c))
            cards.append("\\underline{%s\\,$(-%d)$}" % (k, cost) if cost else k)
        trick_no = str(t // 4 + 1) + ("$^{*}$" if first_diff is not None and t // 4 == first_diff // 4 else "")
        L.append("%s & %s & %s & %d--%d \\\\" % (trick_no, " & ".join(cards), SEATS[winner], tricks[0], tricks[1]))
    L += ["\\bottomrule", "\\end{tabular}}"]
    return "\n".join(L)


def role_of(p, k, seat):
    if k == 0:
        return "wist"
    if seat == p["env"].declarer:
        return "rozgrywajacy"
    if seat == p["env"].dummy:
        return "dziadek"
    return "obrona"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--play-model", default="models/play_dmc.pth")
    ap.add_argument("--tag", default="faza0")
    a = ap.parse_args()

    seeds, tab, vul, opt = load_deals(SEED_TEST, 1000)
    i = int(np.random.default_rng(2026).integers(0, 1000)) if a.seed is None else a.seed - SEED_TEST
    sd = seeds[i]
    dec, dfn = load_nets(os.path.join(ROOT, a.play_model))
    P = Report(f"partia_{sd}_{a.tag}")
    P(f"rozdanie {sd} (nr {i}), siec rozgrywki {a.play_model}")
    runs, pbn_deals = [], []
    for name, short_name, path in [("aktor-krytyk", "aktor", ACTOR_PATH), ("Deep CFR", "cfr", CFR_PATH)]:
        p = run_deal(sd, load_policy(os.path.join(ROOT, path))[0], dec, dfn)
        runs.append((name, short_name, p))
        pbn_deals.append(pbn(p, len(runs), f"Rozdanie {sd} - {name} - {a.tag}", opt[i]))
        env = p["env"]
        if len(runs) == 1:
            P(f"rozdaje {SEATS[p['dealer']]}, po partii NS {p['vul'][0]}, EW {p['vul'][1]}")
            for m in range(4):
                P(f"   {SEATS[m]}: " + "   ".join(["♣", "♦", "♥", "♠"][s] + " " + (
                    "".join(RANKS[r] for r in suit_ranks(p["hands0"][m], s)) or "-") for s in (3, 2, 1, 0)))
            P(f"   zapis optymalny NS {opt[i]:+.0f}\n")

        P(name)
        P("   licytacja: " + "  ".join(f"{SEATS[(p['dealer'] + k) % 4]}:{action_str(x)}"
                                       for k, x in enumerate(p["calls"])))
        if env.declarer is None or not p["plays"]:
            P(f"   pas, strata {abs(opt[i]):.0f}\n")
            continue
        side = env.declarer % 2
        taken = env.tricks_won[side]
        dv = DealValue.from_tricks(tab[i], tuple(vul[i]))
        dd = int(dv.tricks[env.strain, env.declarer])
        score = (1 if side == 0 else -1) * duplicate_score(env.level, env.strain, bool(env.contract_doubled),
                                                            bool(env.contract_redoubled), bool(p["vul"][side]), taken)
        score_dd = dv.score_ns(env.level, env.strain, bool(env.contract_doubled), bool(env.contract_redoubled),
                               env.declarer)
        P(f"   kontrakt {env.level}{STRAINS_TXT[env.strain]} {SEATS[env.declarer]}")
        P(f"   siec: {taken} lew, zapis NS {score:+d}; gra idealna: {dd} lew, zapis NS {score_dd:+.0f}")
        P(f"   strata licytacji: {abs(opt[i] - score_dd):.0f}")
        p["result"] = {"taken": taken, "dd": dd, "score": score, "score_dd": score_dd}

        z = p["plays"]
        for t in range(13):
            P("   %4d  %s" % (t + 1, "  ".join("%-10s" % (f"{SEATS[m]} {card_lin(c)}" + (f"(-{k})" if k else ""))
                                               for m, c, k in z[4 * t:4 * t + 4])))
        costly = [(k // 4 + 1, m, c, cost, role_of(p, k, m)) for k, (m, c, cost) in enumerate(z) if cost]
        dec_cost = sum(cost for m, c, cost in z if m % 2 == side)
        def_cost = sum(cost for m, c, cost in z if m % 2 != side)
        P("   koszty: " + (", ".join(f"lewa {t} {SEATS[m]} {card_lin(c)} -{cost} ({r})"
                                     for t, m, c, cost, r in costly) or "brak"))
        P(f"   suma: rozgrywajacy {dec_cost}, obrona {def_cost}; "
          f"{dd} - {dec_cost} + {def_cost} = {dd - dec_cost + def_cost} (wzieto {taken})")

        text = lin(p, f"Rozdanie {sd} - {name} - {a.tag}", short_name)
        with open(os.path.join(RESULTS_DIR, f"partia_{sd}_{short_name}_{a.tag}.lin"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        P("   https://www.bridgebase.com/tools/handviewer.html?lin=" + urllib.parse.quote(text, safe=""))
        P("")

    plays_a, plays_c = runs[0][2]["plays"], runs[1][2]["plays"]
    first_diff = next((k for k in range(min(len(plays_a), len(plays_c))) if plays_a[k][:2] != plays_c[k][:2]), None)
    if plays_a and plays_c and first_diff is not None:
        (ma, ca, cost_a), (_, cc, cost_c) = plays_a[first_diff], plays_c[first_diff]
        P(f"rozgrywki identyczne przez {first_diff} kart; lewa {first_diff // 4 + 1}, "
          f"{role_of(runs[0][2], first_diff, ma)} "
          f"{SEATS[ma]}: {card_lin(ca)} (koszt {cost_a}) wobec {card_lin(cc)} (koszt {cost_c})")
    if not (plays_a and plays_c):
        first_diff = None

    for name, short_name, p in runs:
        if "result" in p:
            with open(os.path.join(ROOT, "praca", "wykresy", f"partia_{sd}_{short_name}_{a.tag}.tex"), "w",
                      encoding="utf-8") as f:
                f.write(tex(p, first_diff, f"rozdanie {sd}, licytacja: {name}, rozgrywka: {a.play_model}",
                            opt[i]) + "\n")
    with open(os.path.join(RESULTS_DIR, f"partia_{sd}_{a.tag}.pbn"), "w", encoding="utf-8") as f:
        f.write('% PBN 2.1\n% EXPORT\n\n' + "\n\n".join(pbn_deals) + "\n")
    P.close()


if __name__ == "__main__":
    main()
