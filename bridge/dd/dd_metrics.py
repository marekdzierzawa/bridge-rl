"""Quick play metric used during training: the last k tricks compared with double dummy play."""
import copy

import numpy as np
import torch
from endplay.dds import solve_board
from endplay.types import Card, Deal, Denom, Player

from bridge.env.bridge_env import CARD_OFFSET, BridgeEnv
from bridge.env.play_features import legal_card_features
from bridge.rozgrywka.dmc_fixes import force_auction

RANKS = "23456789TJQKA"
SUITS = "CDHS"
DENOMS = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
PLAYERS = [Player.north, Player.east, Player.south, Player.west]


def _deal(hands, to_move, trick, trump):
    tmp = [set(h) for h in hands]
    for m, c in trick:
        tmp[m].add(c)
    text = "N:" + " ".join(
        ".".join("".join(RANKS[c % 13] for c in sorted((x for x in tmp[i] if x // 13 == s), key=lambda y: -(y % 13)))
                 for s in (3, 2, 1, 0))
        for i in range(4))
    d = Deal(text)
    d.trump = DENOMS[trump]
    d.first = PLAYERS[trick[0][0] if trick else to_move]
    for _, c in trick:
        d.play(Card(SUITS[c // 13] + RANKS[c % 13]))
    return d


def solve_position(hands, to_move, trick, trump):
    out = {}
    for card, t in solve_board(_deal(hands, to_move, trick, trump)):
        s = "cdhs".index(card.suit.name[0])
        out[s * 13 + RANKS.index(card.rank.name[1])] = t
    return out


def dd_tricks(hands, leader, trump):
    return max(t for _, t in solve_board(_deal(hands, leader, [], trump)))


def _dd_pick(env, legal):
    w = solve_position(env.hands, env._seat_to_play(), list(env.current_trick.plays), env.strain)
    return int(max(legal, key=lambda a: (w.get(a - CARD_OFFSET, -1), -a)))


def _net_pick(net):
    def f(env, legal):
        obs = env.observe(env.current_player)
        feats = legal_card_features(env, legal)
        q = net.q_all(torch.from_numpy(obs).float(), torch.from_numpy(feats).float())
        return int(legal[int(torch.argmax(q).item())])
    return f


def _playout(env, side, pick_dec, pick_def, audit=None):
    while not env.terminated:
        legal = env.legal_actions()
        m = env._seat_to_play()
        is_dec = m % 2 == side
        pick = pick_dec if is_dec else pick_def
        if audit is not None and len(legal) > 1:
            w = solve_position(env.hands, m, list(env.current_trick.plays), env.strain)
            best = max(w.get(a - CARD_OFFSET, -1) for a in legal)
            worst = min(w.get(a - CARD_OFFSET, best) for a in legal)
            a = pick(env, legal)
            if best != worst:
                audit["dec" if is_dec else "def"].append(best - w.get(a - CARD_OFFSET, best))
        else:
            a = pick(env, legal)
        env.step(a)
    return env.tricks_won[side]


def dd_endgame_eval(dec_net, def_net, seeds, k=5, audit=True):
    pick_dec, pick_def = _net_pick(dec_net), _net_pick(def_net)
    r_dec, r_def, r_self = [], [], []
    aud = {"dec": [], "def": []} if audit else None

    for sd in seeds:
        env = BridgeEnv(seed=sd)
        env.reset()
        force_auction(env, np.random.default_rng(sd))
        if env.terminated:
            continue
        side = env.declarer % 2
        while len(env.completed_tricks) < 13 - k and not env.terminated:
            is_dec = env._seat_to_play() % 2 == side
            env.step((pick_dec if is_dec else pick_def)(env, env.legal_actions()))
        if env.terminated or env.current_trick.plays:
            continue

        tricks_before = env.tricks_won[side]
        on_lead = env._seat_to_play()
        dd = dd_tricks(env.hands, on_lead, env.strain)
        dd_dec = dd if on_lead % 2 == side else k - dd
        r_dec.append(_playout(copy.deepcopy(env), side, pick_dec, _dd_pick, aud) - tricks_before - dd_dec)
        r_def.append(_playout(copy.deepcopy(env), side, _dd_pick, pick_def, aud) - tricks_before - dd_dec)
        r_self.append(_playout(env, side, pick_dec, pick_def) - tricks_before - dd_dec)

    if not r_dec:
        return None
    aud = aud or {"dec": [], "def": []}
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    opt = lambda x: float(np.mean(np.array(x) == 0) * 100) if x else float("nan")
    return {"n": len(r_dec), "k": k,
            "resid_dec": mean(r_dec), "resid_def": mean(r_def), "resid_self": mean(r_self),
            "err_dec": mean(aud["dec"]), "err_def": mean(aud["def"]),
            "opt_dec": opt(aud["dec"]), "opt_def": opt(aud["def"])}


def format_eval(r):
    if r is None:
        return "no data"
    return (f"DD endgame ({r['k']} tricks, n={r['n']}): declarer {r['resid_dec']:+.3f} | "
            f"defence {r['resid_def']:+.3f} | self-play {r['resid_self']:+.3f} | "
            f"error/card {r['err_dec']:.3f}/{r['err_def']:.3f} | opt {r['opt_dec']:.0f}%/{r['opt_def']:.0f}%")
