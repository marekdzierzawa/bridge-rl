"""Double dummy 5x4 trick table (strain x declarer) computed with the DDS solver from endplay."""
import numpy as np
from endplay._dds import SetMaxThreads
from endplay.dds import calc_all_tables
from endplay.types import Deal, Denom, Player

RANK_CH = "23456789TJQKA"
DDS_THREADS = 1
DDS_CHUNK = 32

_DENOMS = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
_PLAYERS = [Player.north, Player.east, Player.south, Player.west]
_threads_set = False


def pbn(hands):
    out = []
    for seat in range(4):
        suits = []
        for suit in (3, 2, 1, 0):
            ranks = sorted((c % 13 for c in hands[seat] if c // 13 == suit), reverse=True)
            suits.append("".join(RANK_CH[r] for r in ranks))
        out.append(".".join(suits))
    return "N:" + " ".join(out)


def _set_threads():
    global _threads_set
    if not _threads_set:
        SetMaxThreads(DDS_THREADS)
        _threads_set = True


def dd_tricks_tables(hands_list):
    _set_threads()
    out = []
    for i in range(0, len(hands_list), DDS_CHUNK):
        deals = [Deal(pbn(h)) for h in hands_list[i:i + DDS_CHUNK]]
        for tab in calc_all_tables(deals):
            t = np.zeros((5, 4), dtype=np.int8)
            for s in range(5):
                for d in range(4):
                    t[s, d] = int(tab[_DENOMS[s], _PLAYERS[d]])
            out.append(t)
    return out


def dd_tricks_table(hands):
    return dd_tricks_tables([hands])[0]
