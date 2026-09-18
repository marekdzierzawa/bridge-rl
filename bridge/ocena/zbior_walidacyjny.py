"""Double dummy tables and par scores for a fixed set of deals, computed once and cached on disk."""
import concurrent.futures
import multiprocessing as mp
import os

import numpy as np

from bridge.dd.dd_table import dd_tricks_tables
from bridge.env.bridge_env import BridgeEnv
from bridge.licytacja.deal_value import par_score_ns

CACHE_DIR = "checkpoints/walidacja"


def _solve_chunk(hands):
    return [np.asarray(t, dtype=np.int8) for t in dd_tricks_tables(hands)]


def compute(seeds, workers=8):
    envs = []
    for sd in seeds:
        e = BridgeEnv(seed=sd)
        e.reset()
        envs.append(e)
    vul = np.array([e.vulnerable for e in envs], dtype=bool)
    hands = [[set(h) for h in e.hands] for e in envs]

    chunks = [hands[i::workers] for i in range(workers)]
    tab = np.zeros((len(seeds), 5, 4), dtype=np.int8)
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(_solve_chunk, chunks))
    for i, w in enumerate(results):
        if w:
            tab[i::workers] = np.array(w, dtype=np.int8)
    return tab, vul


def load_dd_tables(seeds, workers=8):
    seeds = list(seeds)
    p = os.path.join(CACHE_DIR, f"dd_{seeds[0]}_{len(seeds)}.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["tablice"], d["zalozenia"], d["par"]

    tab, vul = compute(seeds, workers)
    par = np.array([par_score_ns(tab[i], tuple(vul[i]))[0] for i in range(len(seeds))], dtype=np.float64)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(p, tablice=tab, zalozenia=vul, par=par)
    return tab, vul, par
