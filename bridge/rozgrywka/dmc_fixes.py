"""Artificial auction of phase 0: the contract follows the strength and shape of the hands, no bidding model."""
import numpy as np

from bridge.env.bridge_env import AUCTION, NOTRUMP, NUM_BIDS, PASS, card_suit

SLAM_COVERAGE = 0.15
OVERCALL_PROB = 0.60
OVERCALL_MIN_HCP = 8
OVERCALL_MIN_LEN = 5


def get_hcp(card_id):
    return {12: 4, 11: 3, 10: 2, 9: 1}.get(card_id % 13, 0)


def _rint(rng, lo, hi):
    return rng.randint(lo, hi) if hasattr(rng, "randint") else int(rng.integers(lo, hi))


def side_suit_lengths(env, side):
    lens = np.zeros(4, dtype=int)
    for seat in (side, side + 2):
        for c in env.hands[seat]:
            lens[card_suit(c)] += 1
    return lens


def _hand_stats(hand):
    hcp = sum(get_hcp(c) for c in hand)
    lens = np.zeros(4, dtype=int)
    for c in hand:
        lens[card_suit(c)] += 1
    return hcp, lens


def force_auction(env, rng=None, coverage_prob=0.15, slam_coverage=SLAM_COVERAGE):
    rng = rng or np.random
    ns = sum(get_hcp(c) for c in env.hands[0]) + sum(get_hcp(c) for c in env.hands[2])
    declarer_side = 0 if ns >= 20 else 1
    points = ns if declarer_side == 0 else (40 - ns)

    lens = side_suit_lengths(env, declarer_side)
    best_suit = int(np.argmax(lens))
    best_len = int(lens[best_suit])

    if rng.random() < coverage_prob:
        want_strain = _rint(rng, 0, 5)
    elif best_len >= 8:
        if best_suit in (0, 1) and best_len < 9 and points >= 25 and rng.random() < 0.65:
            want_strain = NOTRUMP
        else:
            want_strain = best_suit
    else:
        want_strain = NOTRUMP

    fit_bonus = 1 if (best_len >= 9 and want_strain == best_suit) else 0
    if rng.random() < slam_coverage:
        target = _rint(rng, 6, 8)
    elif points >= 33:
        target = _rint(rng, 6, 8)
    elif points >= 25:
        target = _rint(rng, 4, 6) + fit_bonus
    elif points >= 20:
        target = _rint(rng, 2, 4) + fit_bonus
    else:
        target = _rint(rng, 1, 3)
    target = min(target, 7)

    overcalled = [False, False]

    while env.phase == AUCTION and not env.terminated:
        legal = np.flatnonzero(env.get_action_mask())
        seat = env.current_player

        if seat % 2 != declarer_side:
            side = seat % 2
            if not overcalled[side]:
                hcp, hl = _hand_stats(env.hands[seat])
                bs = int(np.argmax(hl))
                if (hl[bs] >= OVERCALL_MIN_LEN and hcp >= OVERCALL_MIN_HCP
                        and rng.random() < OVERCALL_PROB):
                    cand = [a for a in legal if a < NUM_BIDS and a % 5 == bs]
                    if cand:
                        overcalled[side] = True
                        env.step(int(cand[0]))
                        continue
            env.step(int(PASS if PASS in legal else legal[0]))
            continue

        last = env.auction.last_bid
        cur_level = (last // 5 + 1) if last is not None else 0
        if cur_level >= target and PASS in legal:
            env.step(int(PASS))
            continue

        bids = [a for a in legal if a < NUM_BIDS]
        if not bids:
            env.step(int(PASS if PASS in legal else legal[0]))
            continue

        pref = [a for a in bids if a % 5 == want_strain]
        env.step(int(pref[0] if pref else bids[0]))
