"""Simple bidding system based on high card points and suit lengths (baseline and Deep CFR anchor)."""
import numpy as np

from bridge.env.bridge_env import DOUBLE, NOTRUMP, NUM_ACTIONS, NUM_BIDS, PASS, card_suit

_HCP = {12: 4, 11: 3, 10: 2, 9: 1}
PARTNER_BONUS = 1
LEVEL_SHARPNESS = 0.45


def hand_hcp(hand):
    return sum(_HCP.get(c % 13, 0) for c in hand)


def suit_lengths(hand):
    lens = np.zeros(4, dtype=int)
    for c in hand:
        lens[card_suit(c)] += 1
    return lens


def _target_level(hcp, longest):
    if hcp < 6:
        return 0
    if hcp < 11:
        return 2 if longest >= 6 else 1
    if hcp < 15:
        return 2 + (1 if longest >= 6 else 0)
    if hcp < 19:
        return 3 + (1 if longest >= 6 else 0)
    return 4 + (1 if longest >= 6 else 0)


def reference_policy(hand, auction, legal_calls):
    p = np.zeros(NUM_ACTIONS, dtype=np.float64)
    legal = list(legal_calls)
    if not legal:
        return p

    hcp = hand_hcp(hand)
    lens = suit_lengths(hand)
    best_suit = int(np.argmax(lens))
    longest = int(lens[best_suit])
    balanced = longest <= 4
    target = _target_level(hcp, longest)

    last = auction.last_bid
    cur_level = 0 if last is None else last // 5 + 1
    seat = auction.current_player()
    partner_bid = auction.last_bidder is not None and auction.last_bidder % 2 == seat % 2

    partner = (seat + 2) % 4
    partner_bids = sum(1 for (pl, a) in auction.calls if pl == partner and a < NUM_BIDS)
    if target >= 1 and partner_bids:
        target = min(target + PARTNER_BONUS * partner_bids, 7)

    bids = [a for a in legal if a < NUM_BIDS]

    if cur_level >= target or not bids:
        if PASS in legal:
            p[PASS] = 1.0
            if DOUBLE in legal and not partner_bid and cur_level >= target + 2 and hcp >= 9:
                p[PASS], p[DOUBLE] = 0.55, 0.45
            return p / p.sum()
        p[legal] = 1.0 / len(legal)
        return p

    want = NOTRUMP if (balanced and hcp >= 12) else best_suit
    pref = [a for a in bids if a % 5 == want]
    if not pref:
        pref = [a for a in bids if a % 5 == best_suit] or bids

    pref.sort()
    lv = np.array([a // 5 + 1 for a in pref], dtype=np.float64)
    w = LEVEL_SHARPNESS ** np.abs(lv - target)
    w = w / w.sum()
    p[pref] = 0.72 * w

    if PASS in legal:
        p[PASS] = 0.28
    else:
        p[pref] += 0.28 * w

    s = p.sum()
    return p / s if s > 0 else np.where(np.isin(np.arange(NUM_ACTIONS), legal), 1.0 / len(legal), 0.0)
