"""Phase 0: check of the suit establishment features in the card description."""
from math import comb

from wspolne import Report
from bridge.env.bridge_env import CARD_OFFSET, PLAY, BridgeEnv, Trick
from bridge.env.play_features import legal_card_features

SPADES = 3
R = {r: i for i, r in enumerate("23456789TJQKA")}

EXAMPLES = [
    ("AK543 / 76", 2, "76", "QJ2", "AK543", "T98", {3: 4, 4: 3, 5: 2, 6: 2}),
    ("AK6543 / 7", 2, "7", "QJ2", "AK6543", "T98", {3: 5, 4: 4, 5: 3, 6: 2}),
    ("AK5432 / 76", 2, "76", "QJ", "AK5432", "T98", {3: 5, 4: 4, 5: 3}),
    ("AK65432 / renons", 2, "", "QJ2", "AK65432", "T98", {3: 6, 4: 5, 5: 4, 6: 3}),
    ("AKQ / 765", 2, "765", "JT92", "AKQ", "843", {4: 3, 5: 3, 6: 3, 7: 3}),
    ("KQ32 + A4 (as u dziadka)", 2, "A4", "JT9", "KQ32", "8765", {4: 3, 5: 3, 6: 3, 7: 3}),
]
DEFENDER_EXAMPLE = ("obronca W: KQJT9, dziadek A2", 3, "A2", "876", "543", "KQJT9", 4)


def suit_features(spades_N, spades_E, spades_S, spades_W, leader):
    hands = [{SPADES * 13 + R[x] for x in f} for f in (spades_N, spades_E, spades_S, spades_W)]
    rest = [c for c in range(52) if c // 13 != SPADES]
    for r in hands:
        while len(r) < 13:
            r.add(rest.pop())
    env = BridgeEnv()
    env.reset(dealer=0, vulnerable=(False, False), hands=hands)
    env.phase = PLAY
    env.level, env.strain, env.declarer, env.dummy = 3, 4, 2, 0
    env.dummy_revealed = True
    env.current_trick = Trick(leader=leader)
    legal = env.legal_actions()
    f = legal_card_features(env, legal)
    w = [i for i, a in enumerate(legal) if (a - CARD_OFFSET) // 13 == SPADES][0]
    return f[w, 36] * 13, f[w, 38] * 6, f[w, 39] * 6, f[w, 40] * 4


def splits(n):
    out = {}
    for k in range(n + 1):
        a = max(k, n - k)
        out[a] = out.get(a, 0.0) + comb(n, k) * comb(26 - n, 13 - k) / comb(26, 13)
    return out


def main():
    P = Report("f0_cechy_koloru")
    P("%-28s %8s %9s %11s %13s %8s" % ("przyklad", "od razu", "okrazen", "lewy: kod", "lewy: recznie", "oddac"))
    matching = 0
    rows = [(o, p, n, e, s, w, sum(pr * lr[a] for a, pr in splits(13 - len(n) - len(s)).items()))
            for o, p, n, e, s, w, lr in EXAMPLES] + [DEFENDER_EXAMPLE]
    for desc, leader, n, e, s, w, manual in rows:
        immediate, rounds, tricks, to_lose = suit_features(n, e, s, w, leader)
        matching += abs(tricks - manual) <= 0.01
        P("%-28s %8.0f %9.2f %11.2f %13.2f %8.2f" % (desc, immediate, rounds, tricks, manual, to_lose))

    P("\npodzialy kart przeciwnikow:")
    for n in (5, 6, 7):
        P(f"   {n} brakujacych: " + ", ".join(f"{a}-{n - a}: {100 * p:.1f}%" for a, p in sorted(splits(n).items())))
    P(f"\nzgodnosc cechy 'lewy' z rachunkiem recznym: {matching} z {len(rows)}")
    P.close()


if __name__ == "__main__":
    main()
