"""Phase 0: effect of the dummy's cards on the defender's features."""
import numpy as np

from wspolne import Report
from bridge.env.bridge_env import CARD_OFFSET, PLAY, BridgeEnv, Trick, card_rank, card_suit
from bridge.env.play_features import _side_visible_cards, legal_card_features, unseen_mask

SPADES = 3
R = {r: i for i, r in enumerate("23456789TJQKA")}
FEATURES = [(25, "mamy najwyzsza karte"), (26, "wyzszych kart w grze"),
            (29, "kart koloru u innych"), (33, "najwyzszych z rzedu"),
            (34, "mamy najwyzsza (kolor)"), (35, "ile do najwyzszej"),
            (36, "lewy od razu"), (37, "wejscia")]
SCALE = {25: 1, 26: 13, 29: 13, 33: 13, 34: 1, 35: 13, 36: 13, 37: 3}
IDX = [feat_idx for feat_idx, _ in FEATURES]


def old_features(env, legal):
    seat = env._seat_to_play()
    viewer = seat if env.current_player is None else env.current_player
    unseen = unseen_mask(env, viewer)
    unseen_in_suit = np.array([unseen[s * 13:(s + 1) * 13].sum() for s in range(4)], dtype=np.float32)
    higher = np.zeros((4, 13), dtype=np.float32)
    for s in range(4):
        blk = unseen[s * 13:(s + 1) * 13].astype(np.float32)
        higher[s] = blk[::-1].cumsum()[::-1] - blk

    ours = np.zeros(52, dtype=bool)
    ours[list(_side_visible_cards(env, seat))] = True
    live = ours | unseen

    run_len = np.zeros(4, dtype=np.float32)
    we_have_top = np.zeros(4, dtype=np.float32)
    immediate = np.zeros(4, dtype=np.float32)
    for s in range(4):
        sd, lv = ours[s * 13:s * 13 + 13], live[s * 13:s * 13 + 13]
        for r in range(12, -1, -1):
            if not lv[r]:
                continue
            if not sd[r]:
                break
            we_have_top[s] = 1.0
            run_len[s] += 1
        immediate[s] = max(0.0, sd.sum() - unseen_in_suit[s])
    entries = float(we_have_top.sum())

    out = np.zeros((len(legal), 41), dtype=np.float32)
    for i, a in enumerate(legal):
        s, r = card_suit(a - CARD_OFFSET), card_rank(a - CARD_OFFSET)
        hu = higher[s, r]
        out[i, 25] = 1.0 if hu == 0 else 0.0
        out[i, 26] = hu / 13.0
        out[i, 29] = unseen_in_suit[s] / 13.0
        out[i, 33] = run_len[s] / 13.0
        out[i, 34] = we_have_top[s]
        out[i, 35] = hu / 13.0 if not we_have_top[s] else 0.0
        out[i, 36] = immediate[s] / 13.0
        out[i, 37] = (entries - we_have_top[s]) / 3.0
    return out


def make_play_env(hands, level, strain, declarer, dummy, leader, revealed):
    env = BridgeEnv()
    env.reset(dealer=0, vulnerable=(False, False), hands=hands)
    env.phase = PLAY
    env.level, env.strain, env.declarer, env.dummy = level, strain, declarer, dummy
    env.dummy_revealed = revealed
    env.current_trick = Trick(leader=leader)
    return env


def main():
    P = Report("f0_dziadek")

    hands = [{SPADES * 13 + R[x] for x in p} for p in ["A2", "876", "543", "KQJT9"]]
    rest = [c for c in range(52) if c // 13 != SPADES]
    for r in hands:
        while len(r) < 13:
            r.add(rest.pop())
    env = make_play_env(hands, 3, 4, 2, 0, 3, True)
    legal = env.legal_actions()
    i = [k for k, a in enumerate(legal) if (a - CARD_OFFSET) // 13 == SPADES][0]
    new_feats = legal_card_features(env, legal)
    st = old_features(env, legal)
    P("1. obronca W: KQJT9 pik, dziadek N: A2, bez atu; cechy krola pik")
    P("   %-24s %10s %10s" % ("cecha", "stare", "po poprawce"))
    for feat_idx, desc in FEATURES:
        P("   %-24s %10.2f %10.2f"
          % (f"{feat_idx} {desc}", st[i, feat_idx] * SCALE[feat_idx], new_feats[i, feat_idx] * SCALE[feat_idx]))

    N_RANDOM_DEALS = 300
    rng = np.random.default_rng(0)
    def_decisions = def_diff = false_top = false_run = dec_decisions = dec_diff = 0
    run_overshoot = []
    for _ in range(N_RANDOM_DEALS):
        cards = list(rng.permutation(52))
        level, strain, declarer = int(rng.integers(1, 8)), int(rng.integers(0, 5)), int(rng.integers(0, 4))
        env = make_play_env([set(cards[k * 13:(k + 1) * 13]) for k in range(4)], level, strain, declarer,
                            (declarer + 2) % 4, (declarer + 1) % 4, False)
        while not env.terminated:
            legal = env.legal_actions()
            viewer = env.current_player
            if env.dummy_revealed and viewer is not None:
                st = old_features(env, legal)
                new_feats = legal_card_features(env, legal)
                differs = not np.allclose(st[:, IDX], new_feats[:, IDX], atol=1e-6)
                if viewer % 2 == env.declarer % 2:
                    dec_decisions += 1
                    dec_diff += differs
                else:
                    def_decisions += 1
                    def_diff += differs
                    false_top += bool(((st[:, 25] > 0.5) & (new_feats[:, 25] < 0.5)).any())
                    fr = st[:, 33] > new_feats[:, 33] + 1e-6
                    if fr.any():
                        false_run += 1
                        run_overshoot.append(float((st[fr, 33] - new_feats[fr, 33]).max() * 13))
            env.step(int(legal[rng.integers(len(legal))]))

    P(f"\n2. losowe rozgrywki ({N_RANDOM_DEALS} rozdan)")
    P(f"   decyzji obroncy: {def_decisions}")
    P(f"   cechy rozne: {def_diff} ({100 * def_diff / def_decisions:.1f}%)")
    P(f"   falszywe 'mamy najwyzsza': {false_top} ({100 * false_top / def_decisions:.1f}%)")
    P(f"   zawyzone 'najwyzszych z rzedu': {false_run} ({100 * false_run / def_decisions:.1f}%), "
      f"srednio o {np.mean(run_overshoot):.2f}, najwiecej o {max(run_overshoot):.0f}")
    P(f"   decyzji rozgrywajacego: {dec_decisions}, rozne: {dec_diff}")
    P.close()


if __name__ == "__main__":
    main()
