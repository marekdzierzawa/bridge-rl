"""Phase 0: the card play network compared with a simple heuristic."""
import concurrent.futures
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F

from wspolne import MODEL_F0, MODEL_F0_OLD, Report, load_f0_net
from bridge.env.bridge_env import BridgeEnv
from bridge.env.play_features import heuristic_defense
from bridge.rozgrywka.dmc_fixes import force_auction

N_HEUR = 400
N_SIGNAL_DEALS = 300
SEED_START = 4_000_000
FROM_TRICK = 3

_NETS = {}


def _init(path):
    torch.set_num_threads(1)
    _NETS["dec"], _NETS["dfn"], _NETS["feats"] = load_f0_net(path)


def _net_card(net, env, legal):
    feats = _NETS["feats"](env, legal)
    with torch.no_grad():
        q = net.q_all(torch.from_numpy(env.observe(env.current_player)), torch.from_numpy(feats))
    return int(legal[int(torch.argmax(q).item())])


def _new_deal(sd):
    env = BridgeEnv(seed=sd)
    env.reset()
    force_auction(env, np.random.default_rng(sd))
    return env


def _heur_deal(sd, heur_dec, heur_def):
    env = _new_deal(sd)
    if env.terminated:
        return None
    side = env.declarer % 2
    while not env.terminated:
        legal = env.legal_actions()
        is_dec = env.current_player % 2 == side
        if (is_dec and heur_dec) or (not is_dec and heur_def):
            a = heuristic_defense(env, legal)
        else:
            a = _net_card(_NETS["dec"] if is_dec else _NETS["dfn"], env, legal)
        env.step(a)
    return env.tricks_won[side]


def _signal_deal(sd):
    env = _new_deal(sd)
    if env.terminated:
        return None
    side = env.declarer % 2
    partner_err, declarer_err, dummy_err = [], [], []
    while not env.terminated:
        p = env.current_player
        legal = env.legal_actions()
        is_dec = p % 2 == side
        net = _NETS["dec"] if is_dec else _NETS["dfn"]
        if not is_dec and len(env.completed_tricks) >= FROM_TRICK:
            with torch.no_grad():
                h = net.res2(net.res1(F.relu(net.proj(torch.from_numpy(env.observe(p))))))
                b = torch.sigmoid(net.belief_head(h)).numpy()
            for k, offset in enumerate((1, 2, 3)):
                seat = (p + offset) % 4
                truth = np.zeros(52, dtype=np.float32)
                truth[list(env.hands[seat])] = 1.0
                err = float(np.abs(b[k * 52:(k + 1) * 52] - truth).mean())
                if seat == (p + 2) % 4:
                    partner_err.append(err)
                elif seat == env.dummy:
                    dummy_err.append(err)
                else:
                    declarer_err.append(err)
        env.step(_net_card(net, env, legal))
    if not partner_err or not declarer_err:
        return None
    return np.mean(partner_err), np.mean(declarer_err), np.mean(dummy_err) if dummy_err else np.nan


def _heur_batch(seeds):
    out = []
    for sd in seeds:
        w = (_heur_deal(sd, False, True), _heur_deal(sd, True, False), _heur_deal(sd, True, True))
        if None not in w:
            out.append(w)
    return out


def _signal_batch(seeds):
    return [w for w in map(_signal_deal, seeds) if w is not None]


def _run_parallel(path, fn, seeds, workers=8):
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                                initializer=_init, initargs=(path,)) as ex:
        results = ex.map(fn, [seeds[i::workers] for i in range(workers)])
        return np.array([w for chunk in results for w in chunk], dtype=float)


def _ci(x, B=2000):
    rng = np.random.default_rng(0)
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)], [2.5, 97.5])


def main():
    P = Report("f0_heurystyka_sygnaly")
    for name, path in [("stara siec (40 cech)", MODEL_F0_OLD), ("nowa siec (41 cech)", MODEL_F0)]:
        P(f"\n{name}: {path}")
        H = _run_parallel(path, _heur_batch, list(range(SEED_START, SEED_START + N_HEUR)))
        P(f"1. lewy rozgrywajacego ({len(H)} rozdan)")
        for i, desc in [(0, "siec rozgrywa, heurystyka broni"), (1, "heurystyka rozgrywa, siec broni"),
                        (2, "heurystyka po obu stronach")]:
            P("   %-34s %6.2f" % (desc, H[:, i].mean()))
        for i, desc in [(0, "siec jako rozgrywajacy"), (1, "siec jako obrona")]:
            d = H[:, i] - H[:, 2]
            lo, hi = _ci(d)
            P("   %-24s %+.2f  [%+.2f, %+.2f]" % (desc, d.mean(), lo, hi))

        S = _run_parallel(path, _signal_batch, list(range(SEED_START, SEED_START + N_SIGNAL_DEALS)))
        P(f"2. blad zgadywania kart ({len(S)} rozdan, od {FROM_TRICK + 1}. lewy)")
        P("   partner %.4f, rozgrywajacy %.4f, dziadek %.4f" % (S[:, 0].mean(), S[:, 1].mean(), np.nanmean(S[:, 2])))
        gap = S[:, 1] - S[:, 0]
        lo, hi = _ci(gap)
        P("   partner lepiej o %+.4f  [%+.4f, %+.4f], %+.1f%% bledu przy rozgrywajacym"
          % (gap.mean(), lo, hi, 100 * gap.mean() / S[:, 1].mean()))
    P.close()


if __name__ == "__main__":
    main()
