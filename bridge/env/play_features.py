"""Features of a legal card during play (41 values) and a simple card play rule used as a baseline."""
from functools import lru_cache
from itertools import product
from math import comb

import numpy as np

from bridge.env.bridge_env import CARD_OFFSET, NOTRUMP, NUM_CARDS, card_rank, card_suit

CARD_FEAT_DIM = 41
FEATURES_VERSION = 4


def check_features_version(ck, source="checkpoint"):
    feat_dim = ck["def"]["card_enc.0.weight"].shape[1]
    if feat_dim != CARD_FEAT_DIM:
        return f"{source}: {feat_dim} card features, the code expects {CARD_FEAT_DIM}"
    if ck.get("wersja_cech") != FEATURES_VERSION:
        return f"{source}: feature version {ck.get('wersja_cech', 2)}, the code expects version {FEATURES_VERSION}"
    return None


@lru_cache(maxsize=None)
def _split_probs(n, hidden_sizes, is_opp_seat):
    total = sum(hidden_sizes)
    if n == 0 or total == 0:
        return (((0,) * sum(is_opp_seat), 1.0),)
    total_combos = comb(total, n)
    probs = {}
    for k in product(*[range(min(r, n) + 1) for r in hidden_sizes]):
        if sum(k) != n:
            continue
        p = 1.0
        for ki, ri in zip(k, hidden_sizes):
            p *= comb(ri, ki)
        key = tuple(ki for ki, is_opp in zip(k, is_opp_seat) if is_opp)
        probs[key] = probs.get(key, 0.0) + p / total_combos
    return tuple(probs.items())


def _tricks_lost(ours, theirs, lengths):
    a = max(lengths, default=0)
    pool = list(theirs)
    lost = 0
    for r in range(min(a, len(ours))):
        m = sum(1 for d in lengths if d > r)
        c = ours[r]
        if pool and pool[-1] > c:
            i = next(j for j, x in enumerate(pool) if x > c)
            pool.pop(i)
            lost += 1
            m -= 1
        for _ in range(min(m, len(pool))):
            pool.pop(0)
    return lost


@lru_cache(maxsize=200_000)
def _suit_establishment(ours, theirs, n, hidden_sizes, is_opp_seat, their_visible_lens):
    length = len(ours)
    e_a = e_l = e_o = 0.0
    for k, p in _split_probs(n, hidden_sizes, is_opp_seat):
        lengths = list(k) + list(their_visible_lens)
        o = _tricks_lost(ours, theirs, lengths)
        e_a += p * max(lengths, default=0)
        e_l += p * (length - o)
        e_o += p * o
    return e_a, e_l, e_o


def _beats(c, best, trump):
    if card_suit(c) == card_suit(best):
        return card_rank(c) > card_rank(best)
    return card_suit(c) == trump and trump != NOTRUMP


def trick_best(env):
    plays = env.current_trick.plays
    if not plays:
        return None
    w_seat, w_card = plays[0]
    for seat, c in plays[1:]:
        if _beats(c, w_card, env.strain):
            w_seat, w_card = seat, c
    return w_seat, w_card


def unseen_mask(env, viewer):
    seen = np.zeros(NUM_CARDS, dtype=bool)
    for c in env.hands[viewer]:
        seen[c] = True
    if env.dummy_revealed and env.dummy is not None:
        for c in env.hands[env.dummy]:
            seen[c] = True
    for t in env.completed_tricks:
        for _, c in t.plays:
            seen[c] = True
    for _, c in env.current_trick.plays:
        seen[c] = True
    return ~seen


def _side_visible_cards(env, seat):
    cards = set(env.hands[seat])
    if env.declarer is not None and env.dummy_revealed and seat % 2 == env.declarer % 2:
        other = env.dummy if seat == env.declarer else env.declarer
        cards |= set(env.hands[other])
    return cards


def legal_card_features(env, legal_actions, include_dummy=True):
    seat = env._seat_to_play()
    viewer = env.current_player
    if viewer is None:
        viewer = seat
    trump = env.strain
    has_trump = trump != NOTRUMP

    unseen = unseen_mask(env, viewer)
    best = trick_best(env)
    led = env.current_trick.suit_led()
    leading = best is None
    partner_winning = best is not None and best[0] % 2 == seat % 2
    trick_idx = len(env.completed_tricks)
    pos_in_trick = len(env.current_trick.plays)

    my_len = np.zeros(4, dtype=np.float32)
    for c in env.hands[seat]:
        my_len[card_suit(c)] += 1
    unseen_in_suit = np.array([unseen[s * 13:(s + 1) * 13].sum() for s in range(4)],
                              dtype=np.float32)

    foreign = unseen.copy()
    if (include_dummy and env.dummy_revealed and env.dummy is not None
            and env.dummy % 2 != seat % 2):
        for c in env.hands[env.dummy]:
            foreign[c] = True
    foreign_in_suit = np.array([foreign[s * 13:(s + 1) * 13].sum() for s in range(4)],
                               dtype=np.float32)

    higher_foreign = np.zeros((4, 13), dtype=np.float32)
    for s in range(4):
        blk = foreign[s * 13:(s + 1) * 13].astype(np.float32)
        higher_foreign[s] = blk[::-1].cumsum()[::-1] - blk

    side_mask = np.zeros(NUM_CARDS, dtype=bool)
    for c in _side_visible_cards(env, seat):
        side_mask[c] = True
    live = side_mask | foreign

    side_len = np.zeros(4, dtype=np.float32)
    top_run = np.zeros(4, dtype=np.float32)
    has_top = np.zeros(4, dtype=np.float32)
    est_tricks = np.zeros(4, dtype=np.float32)
    for s in range(4):
        sd = side_mask[s * 13:s * 13 + 13]
        lv = live[s * 13:s * 13 + 13]
        side_len[s] = sd.sum()
        run = 0
        started = False
        for r in range(12, -1, -1):
            if not lv[r]:
                continue
            if sd[r]:
                if not started:
                    has_top[s] = 1.0
                    started = True
                run += 1
            else:
                break
        top_run[s] = run
        est_tricks[s] = max(0.0, side_len[s] - foreign_in_suit[s])
    entries = float(has_top.sum())

    visible = {viewer}
    if env.dummy_revealed and env.dummy is not None:
        visible.add(env.dummy)
    our_seats = [p for p in visible if p % 2 == seat % 2]
    their_visible_seats = [p for p in visible if p % 2 != seat % 2]
    hidden_seats = [p for p in range(4) if p not in visible]
    hidden_sizes = tuple(len(env.hands[p]) for p in hidden_seats)
    is_opp_seat = tuple(p % 2 != seat % 2 for p in hidden_seats)

    played = np.zeros(NUM_CARDS, dtype=bool)
    for tr in env.completed_tricks:
        for _, c in tr.plays:
            played[c] = True
    for _, c in env.current_trick.plays:
        played[c] = True
    our_cards = set()
    for p in our_seats:
        our_cards |= set(env.hands[p])

    rounds = np.zeros(4, dtype=np.float32)
    tricks_after = np.zeros(4, dtype=np.float32)
    to_lose = np.zeros(4, dtype=np.float32)
    for s in range(4):
        suit_holdings = [sorted((card_rank(c) for c in env.hands[p] if card_suit(c) == s),
                                reverse=True) for p in our_seats]
        longer = max(suit_holdings, key=len, default=[])
        top_cards = []
        for r in range(12, -1, -1):
            c = s * 13 + r
            if played[c]:
                continue
            if c in our_cards:
                top_cards.append(r)
            else:
                break
        ours = tuple((top_cards + [r for r in longer if r not in top_cards])[:len(longer)])
        theirs = sorted([r for r in range(13) if unseen[s * 13 + r]]
                        + [card_rank(c) for p in their_visible_seats for c in env.hands[p]
                           if card_suit(c) == s])
        their_visible_lens = tuple(sum(1 for c in env.hands[p] if card_suit(c) == s)
                                   for p in their_visible_seats)
        rounds[s], tricks_after[s], to_lose[s] = _suit_establishment(
            ours, tuple(theirs), int(unseen_in_suit[s]), hidden_sizes, is_opp_seat, their_visible_lens)

    out = np.zeros((len(legal_actions), CARD_FEAT_DIM), dtype=np.float32)
    for i, a in enumerate(legal_actions):
        c = a - CARD_OFFSET
        s, r = card_suit(c), card_rank(c)
        f = out[i]
        f[s] = 1.0
        f[4 + r] = 1.0
        f[17] = 1.0 if (has_trump and s == trump) else 0.0
        f[18] = r / 12.0
        f[19] = 1.0 if leading else 0.0
        f[20] = 1.0 if (led is not None and s == led) else 0.0
        f[21] = 1.0 if (best is not None and _beats(c, best[1], trump)) else 0.0
        f[22] = 1.0 if partner_winning else 0.0
        f[23] = 1.0 if (has_trump and s == trump and led is not None
                        and led != trump and my_len[led] == 0) else 0.0
        f[24] = 1.0 if (led is not None and s != led and not (has_trump and s == trump)) else 0.0
        ho = higher_foreign[s, r]
        f[25] = 1.0 if ho == 0 else 0.0
        f[26] = ho / 13.0
        f[27] = my_len[s] / 13.0
        f[28] = 1.0 if my_len[s] == 1 else 0.0
        f[29] = foreign_in_suit[s] / 13.0
        f[30] = trick_idx / 13.0
        f[31] = pos_in_trick / 4.0
        f[32] = side_len[s] / 13.0
        f[33] = top_run[s] / 13.0
        f[34] = has_top[s]
        f[35] = ho / 13.0 if not has_top[s] else 0.0
        f[36] = est_tricks[s] / 13.0
        f[37] = (entries - has_top[s]) / 3.0
        f[38] = rounds[s] / 6.0
        f[39] = tricks_after[s] / 6.0
        f[40] = to_lose[s] / 4.0
    return out


def heuristic_defense(env, legal_actions):
    cards = [a - CARD_OFFSET for a in legal_actions]
    best = trick_best(env)
    seat = env._seat_to_play()

    if best is None:
        lens = {}
        for c in env.hands[seat]:
            lens[card_suit(c)] = lens.get(card_suit(c), 0) + 1
        longest = max(lens, key=lambda s: (lens[s], -s)) if lens else None
        cand = [c for c in cards if card_suit(c) == longest] or cards
        return CARD_OFFSET + min(cand, key=card_rank)

    if best[0] % 2 == seat % 2:
        return CARD_OFFSET + min(cards, key=card_rank)

    winners = [c for c in cards if _beats(c, best[1], env.strain)]
    if winners:
        return CARD_OFFSET + min(winners, key=card_rank)
    return CARD_OFFSET + min(cards, key=card_rank)
