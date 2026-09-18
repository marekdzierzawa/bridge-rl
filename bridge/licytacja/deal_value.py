"""Scores of all contracts from the double dummy table, the par score and the bidding reward."""
import numpy as np

from bridge.dd.dd_table import dd_tricks_table
from bridge.env.bridge_env import NUM_BIDS, duplicate_score

REWARD_SCALE = 1000.0


def _contract_score_ns(tricks, vul, level, strain, declarer):
    side = declarer % 2
    t = int(tricks[strain, declarer])
    made = t >= 6 + level
    s = duplicate_score(level, strain, not made, False, vul[side], t)
    return s if side == 0 else -s


def par_score_ns(tricks, vul):
    score_ns = 0
    last = -1
    par_bid = -1
    for _ in range(NUM_BIDS + 4):
        moved = False
        for side in (0, 1):
            sign = 1 if side == 0 else -1
            cur = sign * score_ns
            cand = None
            for b in range(last + 1, NUM_BIDS):
                level, strain = b // 5 + 1, b % 5
                for declarer in (side, side + 2):
                    s_ns = _contract_score_ns(tricks, vul, level, strain, declarer)
                    if sign * s_ns > cur + 1e-9 and (cand is None or sign * s_ns > sign * cand[0] + 1e-9):
                        cand = (s_ns, b)
            if cand is not None:
                score_ns, last, par_bid = cand[0], cand[1], cand[1]
                moved = True
        if not moved:
            break
    return score_ns, par_bid


class DealValue:
    __slots__ = ("tricks", "vul", "par", "par_bid")

    def __init__(self, env):
        self.vul = env.vulnerable
        self.tricks = dd_tricks_table(env.hands)
        self.par, self.par_bid = par_score_ns(self.tricks, self.vul)

    @classmethod
    def from_tricks(cls, tricks, vul):
        self = cls.__new__(cls)
        self.vul = tuple(vul)
        self.tricks = np.asarray(tricks, dtype=np.int8)
        self.par, self.par_bid = par_score_ns(self.tricks, self.vul)
        return self

    def score_ns(self, level, strain, doubled, redoubled, declarer):
        side = declarer % 2
        t = int(self.tricks[strain, declarer])
        s = duplicate_score(level, strain, doubled, redoubled, self.vul[side], t)
        return s if side == 0 else -s

    def reward_ns(self, level, strain, doubled, redoubled, declarer):
        d = self.score_ns(level, strain, doubled, redoubled, declarer) - self.par
        return float(np.tanh(d / REWARD_SCALE))

    def reward_for(self, player, level, strain, doubled, redoubled, declarer):
        r = self.reward_ns(level, strain, doubled, redoubled, declarer)
        return r if player % 2 == 0 else -r

    def reward_passed_out(self, player):
        r = float(np.tanh(-self.par / REWARD_SCALE))
        return r if player % 2 == 0 else -r


def auction_reward(dv, au, player):
    if au.passed_out:
        return dv.reward_passed_out(player)
    level, strain, dbl, rdbl, declarer = au.contract()
    return dv.reward_for(player, level, strain, dbl, rdbl, declarer)


def auction_reward_truncated(dv, au, player):
    if au.last_bid is None:
        return dv.reward_passed_out(player)
    level, strain, dbl, rdbl, declarer = au.contract()
    return dv.reward_for(player, level, strain, dbl, rdbl, declarer)
