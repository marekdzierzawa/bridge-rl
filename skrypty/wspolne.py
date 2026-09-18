"""Shared helpers of the thesis scripts: reports, bootstrap, loading networks and playing deals."""
import concurrent.futures
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from bridge.env.bridge_env import AUCTION, NUM_ACTIONS, PASS, BridgeEnv
from bridge.env.play_features import legal_card_features
from bridge.licytacja.bid_reference import reference_policy
from bridge.licytacja.deal_value import DealValue
from bridge.nets.models import PlayQNetwork
from bridge.ocena import eval_dd as E
from bridge.ocena.evaluate_models import load_policy
from bridge.ocena.zbior_walidacyjny import load_dd_tables

RESULTS_DIR = os.path.join(ROOT, "praca", "wyniki")
FIGURES_DIR = os.path.join(ROOT, "praca", "wykresy")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_F0 = "models/play_dmc.pth"
MODEL_F0_OLD = "models/play_dmc_40cech.pth"

SEED_F0 = 60_000
SEED_VALIDATION = 7_500_000
SEED_TEST = 8_000_000

STRAIN_NAMES = ["trefl", "karo", "kier", "pik", "bez atu"]
STRAINS_SHORT = ["C", "D", "H", "S", "BA"]


class Report:
    def __init__(self, name):
        self._f = open(os.path.join(RESULTS_DIR, name + ".txt"), "w", encoding="utf-8")

    def __call__(self, *a):
        self._f.write(" ".join(str(x) for x in a) + "\n")

    def close(self):
        self._f.close()


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"), dpi=160, bbox_inches="tight")


def bootstrap_ci(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.std())


def load_deals(seed0, n):
    seeds = list(range(seed0, seed0 + n))
    tab, vul, opt = load_dd_tables(seeds)
    return seeds, tab, vul, np.asarray(opt, dtype=float)


def run_auctions(bidder, seeds, tables, vul, max_calls=60):
    out = []
    for i, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        env = BridgeEnv(seed=sd)
        env.reset()
        k = 0
        while env.phase == AUCTION and not env.terminated and k < max_calls:
            k += 1
            env.step(bidder(env, rng))
        if env.terminated or env.phase == AUCTION:
            out.append({"passed": True, "score_ns": 0.0, "score_dec": 0.0, "level": 0, "strain": -1, "declarer": -1,
                        "doubled": False, "redoubled": False, "length": k})
            continue
        dv = DealValue.from_tricks(tables[i], tuple(vul[i]))
        s = float(dv.score_ns(env.level, env.strain, env.contract_doubled, env.contract_redoubled, env.declarer))
        out.append({"passed": False, "score_ns": s, "score_dec": s if env.declarer % 2 == 0 else -s,
                    "level": env.level, "strain": env.strain, "declarer": env.declarer,
                    "doubled": bool(env.contract_doubled), "redoubled": bool(env.contract_redoubled), "length": k})
    return out


def _heuristic_probs(env):
    pr = reference_policy(env.hands[env.current_player], env.auction, env.auction.legal_calls())
    s = pr.sum()
    return pr / s if s > 0 else None


def bid_heur_best(env, rng):
    p = _heuristic_probs(env)
    return PASS if p is None else int(np.argmax(p))


def bid_heur_sample(env, rng):
    p = _heuristic_probs(env)
    if p is None:
        return PASS
    return int(rng.choice(NUM_ACTIONS, p=p.astype(np.float64) / p.sum()))


def bid_pass(env, rng):
    return PASS


def bid_random(env, rng):
    return int(rng.choice(np.flatnonzero(env.get_action_mask())))


def old_card_features(env, legal):
    f = legal_card_features(env, legal, include_dummy=False)[:, :38]
    own_side = env._seat_to_play() % 2
    decl_side = env.declarer % 2 if env.declarer is not None else 0
    target = (6 + env.level) if own_side == decl_side else (14 - (6 + env.level))
    out = np.zeros((len(legal), 40), dtype=np.float32)
    out[:, :38] = f
    out[:, 38] = max(0, target - env.tricks_won[own_side]) / 13.0
    out[:, 39] = 1.0 if own_side == decl_side else 0.0
    return out


def load_f0_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    feat_dim = int(ck["def"]["card_enc.0.weight"].shape[1])
    feats = old_card_features if feat_dim == 40 else legal_card_features
    dec, dfn = PlayQNetwork(card_dim=feat_dim), PlayQNetwork(card_dim=feat_dim)
    dec.load_state_dict(ck["dec"])
    dfn.load_state_dict(ck["def"])
    return dec.eval(), dfn.eval(), feats


def _init_net(path, random_lead, bidding=None):
    torch.set_num_threads(1)
    E._LEAD_RANDOM = bool(random_lead)
    if bidding is not None:
        E._BID_SOURCE, E._BID_POLICY_PATH = bidding
        E._LOGITS = load_policy(bidding[1])[0]
    if path is not None:
        dec, dfn, feats = load_f0_net(path)
        E.legal_card_features = feats
        E._NETS["dec"], E._NETS["def"] = dec, dfn


def play_deals(seeds, dec_policy, def_policy, audit, path, workers=8, random_lead=False, bidding=None):
    if "net" not in (dec_policy, def_policy):
        path = None
    fingerprint = "brak"
    if path is not None:
        s = os.stat(path)
        fingerprint = f"{os.path.basename(path)}_{s.st_size}_{int(s.st_mtime)}"
    key = f"{fingerprint}_{dec_policy}_{def_policy}_{int(audit)}_{int(random_lead)}_{seeds[0]}_{len(seeds)}"
    if bidding is not None:
        s = os.stat(bidding[1])
        key += f"_licyt-{os.path.splitext(os.path.basename(bidding[1]))[0]}_{s.st_size}_{int(s.st_mtime)}"
    os.makedirs(os.path.join(RESULTS_DIR, "_pamiec"), exist_ok=True)
    cache_file = os.path.join(RESULTS_DIR, "_pamiec", key + ".pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    chunks = [seeds[i::workers] for i in range(workers)]
    recs = []
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_net,
                                                initargs=(path, random_lead, bidding)) as ex:
        for r in ex.map(E._run_chunk, chunks, [dec_policy] * workers, [def_policy] * workers, [audit] * workers):
            recs.extend(r)
    recs.sort(key=lambda r: r["seed"])
    with open(cache_file, "wb") as f:
        pickle.dump(recs, f)
    return recs
