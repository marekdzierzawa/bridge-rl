"""Phase 2: phase 0 and phase 2 play networks on expert deals and auctions."""
import concurrent.futures
import multiprocessing as mp
import os
import pickle

import numpy as np
import torch
from endplay.types import Deal

from wspolne import MODEL_F0, RESULTS_DIR, ROOT, Report, bootstrap_ci
from f1_rozklady import (DATASET_CANDIDATES, decode_dd_table, group_deals, load_expert_records, parse_call, parse_card,
                         parse_contract, seat_order_from, vul_at_table)
from bridge.dd.dd_table import dd_tricks_table
from bridge.env.bridge_env import AUCTION, CARD_OFFSET, BridgeEnv
from bridge.env.play_features import legal_card_features
from bridge.ocena.eval_dd import DENOM, PLAYER, _role, dd_values, hands_to_pbn, load_nets, to_endplay_card

MODEL_F2 = "models/play_dmc_faza2_model.pth"
NETS = {"faza0": MODEL_F0, "faza2": MODEL_F2}
DECLARER_ROLES = ("rozgrywajacy", "dziadek")
ACCURACY_GROUPS = [("wist", ("wist",)), ("obrona bez wistu", ("obrona",)), ("strona rozgrywajaca", DECLARER_ROLES)]
WORKERS = 8

_LOADED = {}


def call_index(t):
    c = parse_call(t)
    return {"P": 35, "X": 36, "XX": 37}[c[0]] if c[0] != "bid" else (c[1] - 1) * 5 + c[2]


def _init():
    torch.set_num_threads(1)
    for k, p in NETS.items():
        _LOADED[k] = load_nets(os.path.join(ROOT, p))


def _play_task(z):
    ident, player, hands, dealer, vul, calls, cards = z
    env = BridgeEnv()
    env.reset(dealer=dealer, vulnerable=tuple(vul), hands=[set(h) for h in hands])
    for a in calls:
        if env.terminated or env.phase != AUCTION:
            break
        if a not in env.legal_actions():
            return ident, None
        env.step(a)
    if env.terminated or env.phase == AUCTION:
        return ident, None
    deal = Deal(hands_to_pbn(env.hands))
    deal.trump = DENOM[env.strain]
    deal.first = PLAYER[(env.declarer + 1) % 4]
    errors, k = [], 0
    while not env.terminated:
        seat = env._seat_to_play()
        legal = env.legal_actions()
        if player == "eksperci":
            a = cards[k] + CARD_OFFSET
            k += 1
            if a not in legal:
                return ident, None
        elif len(legal) == 1:
            a = legal[0]
        else:
            dec, dfn = _LOADED[player]
            net = dec if seat % 2 == env.declarer % 2 else dfn
            feats = legal_card_features(env, legal)
            with torch.no_grad():
                q = net.q_all(torch.from_numpy(env.observe(env.current_player)).float(),
                              torch.from_numpy(feats).float())
            a = int(legal[int(torch.argmax(q).item())])
        if len(legal) > 1:
            vals = dd_values(deal)
            best = max(vals.get(x - CARD_OFFSET, -1) for x in legal)
            worst = min(vals.get(x - CARD_OFFSET, best) for x in legal)
            errors.append((_role(env, seat), len(env.completed_tricks), best - vals.get(a - CARD_OFFSET, best),
                           best != worst))
        deal.play(to_endplay_card(a - CARD_OFFSET))
        env.step(a)
    return ident, {"strain": env.strain, "errors": errors}


def _run_batch(fn, tasks):
    return [fn(z) for z in tasks]


def parallel_map(fn, tasks, init=None, chunk_size=16):
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    with concurrent.futures.ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("spawn"), initializer=init) as ex:
        return [w for batch in ex.map(_run_batch, [fn] * len(chunks), chunks) for w in batch]


def metrics(r):
    e = r["errors"]
    m = {"nt": r["strain"] == 4,
         "dec": sum(b for role, _, b, _ in e if role in DECLARER_ROLES),
         "def": sum(b for role, _, b, _ in e if role in ("wist", "obrona")),
         "lead": sum(b for role, _, b, _ in e if role == "wist")}
    for name, roles in ACCURACY_GROUPS:
        significant = [b for role, _, b, i in e if role in roles and i]
        m["t_" + name] = (sum(1 for b in significant if b == 0), len(significant))
    return m


def accuracy(t, idx):
    t = t[idx]
    return 100.0 * t[:, 0].sum() / max(1, t[:, 1].sum())


def main():
    fname = next(k for k in DATASET_CANDIDATES if os.path.exists(k))
    P = Report("f2_eksperci")
    recs = load_expert_records()
    all_deals = group_deals(recs)
    assembled = [rd for rd in all_deals if rd["hands"] is not None]
    P(f"faza 0: {MODEL_F0}, faza 2: {MODEL_F2}")
    P(f"1. {len(recs)} licytacji, {len(all_deals)} rozdan, rece zlozone w {len(assembled)}")

    fingerprint = "_".join(f"{os.path.basename(p)}-{os.stat(os.path.join(ROOT, p)).st_size}"
                           f"-{int(os.stat(os.path.join(ROOT, p)).st_mtime)}" for p in NETS.values())
    os.makedirs(os.path.join(RESULTS_DIR, "_pamiec"), exist_ok=True)
    cache_file = os.path.join(RESULTS_DIR, "_pamiec", f"eksperci_{os.stat(fname).st_size}_{fingerprint}.pkl")
    results = None
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            tables, results = pickle.load(f)
    else:
        tables = parallel_map(dd_tricks_table, [rd["hands"] for rd in assembled], chunk_size=8)

    matching = [rd for rd, t in zip(assembled, tables)
                if np.array_equal(decode_dd_table(rd["key"][3], seat_order_from(rd["player"])), t)]
    P(f"2. tablica z rak zgodna z plikiem: {len(matching)} z {len(assembled)}")

    unique_auctions, experts, tasks = {}, [], []
    for num, rd in enumerate(matching):
        dealer = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}[rd["key"][0]]
        vul = vul_at_table(rd["key"][1], rd["player"])
        hands = [sorted(h) for h in rd["hands"]]
        for r in rd["auctions"]:
            if parse_contract(r) is None:
                continue
            calls = tuple(call_index(t) for t in r["auction"])
            if (num, calls) not in unique_auctions:
                unique_auctions[(num, calls)] = len(unique_auctions)
                for g in NETS:
                    tasks.append(((g, unique_auctions[(num, calls)]), g, hands, dealer, vul, calls, None))
            if len(r["play"]) == 52:
                experts.append((num, unique_auctions[(num, calls)]))
                tasks.append((("eksperci", len(experts) - 1), "eksperci", hands, dealer, vul, calls,
                              [parse_card(t) for t in r["play"]]))
    P(f"3. {len(unique_auctions)} roznych licytacji z kontraktem, {len(experts)} z pelna rozgrywka eksperta")

    if results is None:
        results = dict(parallel_map(_play_task, tasks, init=_init))
        with open(cache_file, "wb") as f:
            pickle.dump((tables, results), f)

    ids = [u for u in range(len(unique_auctions))
           if results.get(("faza0", u)) is not None and results.get(("faza2", u)) is not None]
    deal_of = {u: num for (num, _), u in unique_auctions.items()}
    A = [metrics(results[("faza0", u)]) for u in ids]
    B = [metrics(results[("faza2", u)]) for u in ids]
    groups = {}
    for i, u in enumerate(ids):
        groups.setdefault(deal_of[u], []).append(i)
    groups = [np.array(v) for v in groups.values()]
    is_nt = np.array([x["nt"] for x in A])
    P(f"\n4. faza 0 i faza 2: {len(ids)} licytacji w {len(groups)} rozdaniach "
      f"(nieudane: {len(unique_auctions) - len(ids)}), bez atu {100 * is_nt.mean():.1f}%")

    def grouped_ci(fn):
        lo, hi, _ = bootstrap_ci(lambda i: fn(np.concatenate([groups[j] for j in i])), len(groups))
        return fn(np.arange(len(ids))), lo, hi

    P("   %-34s %22s %22s %24s" % ("miara [lewy na licytacje]", "faza 0", "faza 2", "roznica"))
    for key, desc in [("dec", "strata rozgrywajacego"), ("def", "strata obrony"), ("lead", "strata wistu")]:
        for mask, suffix in [(None, ""), (is_nt, ", bez atu"), (~is_nt, ", kolory")]:
            if key == "lead" and mask is not None:
                continue
            x0 = np.array([x[key] for x in A], float)
            x2 = np.array([x[key] for x in B], float)
            w = np.ones(len(ids), bool) if mask is None else mask
            masked_mean = lambda x: lambda i: x[i][w[i]].mean() if w[i].any() else np.nan
            P("   %-34s %6.3f (%5.3f-%5.3f) %6.3f (%5.3f-%5.3f) %+7.3f (%+6.3f;%+6.3f)"
              % (desc + suffix, *grouped_ci(masked_mean(x0)), *grouped_ci(masked_mean(x2)),
                 *grouped_ci(masked_mean(x2 - x0))))
    P("   %-34s %22s %22s %24s" % ("trafnosc [% decyzji istotnych]", "faza 0", "faza 2", "roznica [pkt proc.]"))
    for group_name, _ in ACCURACY_GROUPS:
        t0 = np.array([x["t_" + group_name] for x in A], float)
        t2 = np.array([x["t_" + group_name] for x in B], float)
        P("   %-34s %5.1f%% (%4.1f-%4.1f) %5.1f%% (%4.1f-%4.1f) %+6.1f (%+5.1f;%+5.1f)"
          % (group_name, *grouped_ci(lambda i: accuracy(t0, i)), *grouped_ci(lambda i: accuracy(t2, i)),
             *grouped_ci(lambda i: accuracy(t2, i) - accuracy(t0, i))))

    ids_set = set(ids)
    expert_idx = [e for e, (num, u) in enumerate(experts) if results.get(("eksperci", e)) is not None and u in ids_set]
    pos = {u: i for i, u in enumerate(ids)}
    E_m = [metrics(results[("eksperci", e)]) for e in expert_idx]
    F0 = [A[pos[experts[e][1]]] for e in expert_idx]
    F2 = [B[pos[experts[e][1]]] for e in expert_idx]
    groups_e = {}
    for i, e in enumerate(expert_idx):
        groups_e.setdefault(experts[e][0], []).append(i)
    groups_e = [np.array(v) for v in groups_e.values()]
    P(f"\n5. eksperci i sieci na tych samych licytacjach: {len(expert_idx)} rozgrywek w {len(groups_e)} rozdaniach "
      f"(nieudane: {len(experts) - len(expert_idx)})")

    def grouped_ci_e(fn):
        lo, hi, _ = bootstrap_ci(lambda i: fn(np.concatenate([groups_e[j] for j in i])), len(groups_e))
        return fn(np.arange(len(expert_idx))), lo, hi

    P("   %-24s %20s %20s %20s %22s %22s" % ("miara", "eksperci", "faza 0", "faza 2", "faza 0 - eksperci",
                                            "faza 2 - eksperci"))
    for key, desc in [("dec", "strata rozgrywajacego"), ("def", "strata obrony"), ("lead", "strata wistu")]:
        xe, x0, x2 = (np.array([x[key] for x in grp], float) for grp in (E_m, F0, F2))
        row = [grouped_ci_e(lambda i, x=x: x[i].mean()) for x in (xe, x0, x2)]
        row += [grouped_ci_e(lambda i: (x0 - xe)[i].mean()), grouped_ci_e(lambda i: (x2 - xe)[i].mean())]
        P("   %-24s %5.3f (%5.3f-%5.3f) %5.3f (%5.3f-%5.3f) %5.3f (%5.3f-%5.3f) %+6.3f (%+6.3f;%+6.3f) "
          "%+6.3f (%+6.3f;%+6.3f)" % (desc, *[v for s in row for v in s]))
    for group_name, _ in ACCURACY_GROUPS:
        te, t0, t2 = (np.array([x["t_" + group_name] for x in grp], float) for grp in (E_m, F0, F2))
        row = [grouped_ci_e(lambda i, t=t: accuracy(t, i)) for t in (te, t0, t2)]
        row += [grouped_ci_e(lambda i: accuracy(t0, i) - accuracy(te, i)),
                grouped_ci_e(lambda i: accuracy(t2, i) - accuracy(te, i))]
        P("   %-24s %5.1f%% (%4.1f-%4.1f) %5.1f%% (%4.1f-%4.1f) %5.1f%% (%4.1f-%4.1f) %+6.1f (%+5.1f;%+5.1f) "
          "%+6.1f (%+5.1f;%+5.1f)" % ("trafnosc: " + group_name, *[v for s in row for v in s]))
    P.close()


if __name__ == "__main__":
    main()
