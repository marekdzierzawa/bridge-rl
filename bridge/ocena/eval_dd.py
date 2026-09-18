"""Card play audit against double dummy play (endplay solver): tricks per deal and the cost of each decision."""
import argparse
import collections
import concurrent.futures
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from endplay.dds import solve_board
from endplay.types import Card, Deal, Denom, Player

from bridge.env.bridge_env import AUCTION, CARD_OFFSET, BridgeEnv
from bridge.env.play_features import check_features_version, heuristic_defense, legal_card_features
from bridge.nets.models import PlayQNetwork
from bridge.ocena.evaluate_models import load_policy
from bridge.rozgrywka.dmc_fixes import force_auction, get_hcp

RANK_CH = "23456789TJQKA"
SUIT_CH = "CDHS"
STRAIN_NAME = ["C", "D", "H", "S", "NT"]
DENOM = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
PLAYER = [Player.north, Player.east, Player.south, Player.west]
ROLE = ("wist", "obrona", "rozgrywajacy", "dziadek")

_NETS = {"dec": None, "def": None}
_LEAD_RANDOM = False
_BID_SOURCE = "regula"
_BID_POLICY_PATH = None
_LOGITS = None


def hands_to_pbn(hands):
    def one(h):
        return ".".join("".join(RANK_CH[c % 13]
                                for c in sorted((x for x in h if x // 13 == s), key=lambda y: -(y % 13)))
                        for s in (3, 2, 1, 0))
    return "N:" + " ".join(one(hands[i]) for i in range(4))


def to_endplay_card(cid):
    return Card(SUIT_CH[cid // 13] + RANK_CH[cid % 13])


def dd_values(deal):
    return {"cdhs".index(card.suit.name[0]) * 13 + RANK_CH.index(card.rank.name[1]): t
            for card, t in solve_board(deal)}


def load_nets(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    msg = check_features_version(ck, path)
    if msg:
        print(msg)
    dec, dfn = PlayQNetwork().to(device), PlayQNetwork().to(device)
    dec.load_state_dict(ck["dec"])
    dfn.load_state_dict(ck["def"])
    return dec.eval(), dfn.eval()


def pick_net(net, env, legal):
    if _LEAD_RANDOM and not env.completed_tricks and not env.current_trick.plays:
        return int(legal[np.random.randint(len(legal))])
    obs = env.observe(env.current_player)
    feats = legal_card_features(env, legal)
    q = net.q_all(torch.from_numpy(obs).float(), torch.from_numpy(feats).float())
    return int(legal[int(torch.argmax(q).item())])


def pick_dd(vals, legal):
    best, best_a = -1, legal[0]
    for a in legal:
        if vals.get(a - CARD_OFFSET, -1) > best:
            best, best_a = vals[a - CARD_OFFSET], a
    return int(best_a)


def _role(env, seat):
    if seat == env.dummy:
        return "dziadek"
    if seat == env.declarer:
        return "rozgrywajacy"
    if not env.completed_tricks and not env.current_trick.plays:
        return "wist"
    return "obrona"


def _reach_contract(env, rng):
    if _BID_SOURCE == "model" and _LOGITS is not None:
        while env.phase == AUCTION and not env.terminated:
            env.step(int(torch.argmax(_LOGITS(env)).item()))
    else:
        force_auction(env, rng)


def play_deal(seed, dec_policy, def_policy, audit, dec_net=None, def_net=None):
    rng = np.random.default_rng(seed)
    env = BridgeEnv(seed=seed)
    env.reset()
    hands0 = [set(h) for h in env.hands]
    _reach_contract(env, rng)
    if env.terminated or env.phase == AUCTION:
        return None

    dside = env.declarer % 2
    hcp = sum(get_hcp(c) for c in hands0[dside]) + sum(get_hcp(c) for c in hands0[dside + 2])
    deal = Deal(hands_to_pbn(hands0))
    deal.trump = DENOM[env.strain]
    deal.first = PLAYER[(env.declarer + 1) % 4]
    dd_declarer = 13 - max(t for _, t in solve_board(deal))

    errors = []
    need_vals = audit or "dd" in (dec_policy, def_policy)
    while not env.terminated:
        seat = env._seat_to_play()
        legal = env.legal_actions()
        is_dec_side = seat % 2 == dside
        policy = dec_policy if is_dec_side else def_policy
        vals = dd_values(deal) if need_vals and len(legal) > 1 else {}

        if policy == "dd":
            action = pick_dd(vals, legal) if vals else legal[0]
        elif policy == "heur":
            action = heuristic_defense(env, legal)
        else:
            action = pick_net(dec_net if is_dec_side else def_net, env, legal)

        if audit and vals:
            best = max(vals.get(a - CARD_OFFSET, -1) for a in legal)
            got = vals.get(action - CARD_OFFSET, best)
            worst = min(vals.get(a - CARD_OFFSET, best) for a in legal)
            errors.append((_role(env, seat), len(env.completed_tricks), best - got, best != worst))

        deal.play(to_endplay_card(action - CARD_OFFSET))
        env.step(action)

    return {"seed": seed, "hcp": hcp, "level": env.level, "strain": env.strain,
            "dd_declarer": dd_declarer, "tricks": env.tricks_won[dside],
            "made": env.tricks_won[dside] >= 6 + env.level, "dd_made": dd_declarer >= 6 + env.level,
            "errors": errors}


def set_bidding(source, path=None):
    global _BID_SOURCE, _BID_POLICY_PATH
    _BID_SOURCE, _BID_POLICY_PATH = source, path


def _init(path, bidding=("regula", None)):
    global _BID_SOURCE, _BID_POLICY_PATH, _LOGITS
    _BID_SOURCE, _BID_POLICY_PATH = bidding
    if _BID_SOURCE == "model":
        _LOGITS = load_policy(_BID_POLICY_PATH)[0]
    if path is not None:
        torch.set_num_threads(1)
        _NETS["dec"], _NETS["def"] = load_nets(path)


def _run_chunk(seeds, dec_policy, def_policy, audit):
    results = [play_deal(sd, dec_policy, def_policy, audit, _NETS["dec"], _NETS["def"]) for sd in seeds]
    return [r for r in results if r is not None]


def run_matchup(seeds, dec_policy, def_policy, audit, workers, model_path):
    if "net" not in (dec_policy, def_policy):
        model_path = None
    chunks = [seeds[i::workers] for i in range(workers)]
    recs = []
    with concurrent.futures.ProcessPoolExecutor(workers, initializer=_init,
                                                initargs=(model_path, (_BID_SOURCE, _BID_POLICY_PATH))) as ex:
        for r in ex.map(_run_chunk, chunks, [dec_policy] * workers, [def_policy] * workers, [audit] * workers):
            recs.extend(r)
    return recs


def summarize(recs, label):
    tricks = np.array([r["tricks"] for r in recs], dtype=float)
    dd = np.array([r["dd_declarer"] for r in recs], dtype=float)
    made = np.mean([r["made"] for r in recs]) * 100
    dd_made = np.mean([r["dd_made"] for r in recs]) * 100
    print(f"{label:<22} tricks {tricks.mean():5.2f}  DD {dd.mean():5.2f}  diff {(tricks - dd).mean():+5.2f}  "
          f"made {made:5.1f}% (DD {dd_made:5.1f}%)")


def report_by_strain(recs):
    print(f"\n{'strain':<7}{'n':>6}{'tricks':>8}{'DD':>8}{'diff':>10}{'made':>11}{'DD made':>10}")
    for s in range(5):
        sub = [r for r in recs if r["strain"] == s]
        if not sub:
            continue
        t = np.mean([r["tricks"] for r in sub])
        d = np.mean([r["dd_declarer"] for r in sub])
        m = np.mean([r["made"] for r in sub]) * 100
        dm = np.mean([r["dd_made"] for r in sub]) * 100
        print(f"{STRAIN_NAME[s]:<7}{len(sub):>6}{t:>8.2f}{d:>8.2f}{t - d:>+10.2f}{m:>10.1f}%{dm:>9.1f}%")


def report_audit(recs):
    by_role = collections.defaultdict(list)
    by_trick = collections.defaultdict(list)
    for r in recs:
        for role, tr, err, matters in r["errors"]:
            by_role[role].append((err, matters))
            by_trick[tr].append(err)
    print(f"{'role':<15}{'decisions':>10}{'significant':>12}{'avg loss':>10}{'% opt.':>9}{'% opt. sig.':>13}")
    for role in ROLE:
        if role not in by_role:
            continue
        e = np.array([x[0] for x in by_role[role]], dtype=float)
        m = np.array([x[1] for x in by_role[role]], dtype=bool)
        opt_rel = (e[m] == 0).mean() * 100 if m.any() else float("nan")
        print(f"{role:<15}{len(e):>10}{int(m.sum()):>12}{e.mean():>10.3f}"
              f"{(e == 0).mean() * 100:>8.1f}%{opt_rel:>12.1f}%")
    print("trick  " + "".join(f"{t + 1:>6}" for t in range(13) if t in by_trick))
    print("loss   " + "".join(f"{np.mean(by_trick[t]):>6.2f}" for t in range(13) if t in by_trick))


def _role_stats(recs, wanted_role):
    rows = [(e, m) for r in recs for role, _, e, m in r["errors"] if role == wanted_role]
    e = np.array([x[0] for x in rows], dtype=float)
    m = np.array([x[1] for x in rows], dtype=bool)
    if e.size == 0:
        return float("nan"), float("nan"), 0
    opt = (e[m] == 0).mean() * 100 if m.any() else float("nan")
    return opt, e.mean(), int(m.sum())


def _paired(recs_a, recs_b, key):
    a = {r["seed"]: r for r in recs_a}
    b = {r["seed"]: r for r in recs_b}
    common = sorted(set(a) & set(b))
    d = np.array([float(b[s][key]) - float(a[s][key]) for s in common])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)


def _binom_se(p_pct, n):
    p = p_pct / 100.0
    return 100.0 * np.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def run_compare(seeds, path_a, path_b, workers):
    dec_a = run_matchup(seeds, "net", "dd", True, workers, path_a)
    dec_b = run_matchup(seeds, "net", "dd", True, workers, path_b)
    def_a = run_matchup(seeds, "dd", "net", True, workers, path_a)
    def_b = run_matchup(seeds, "dd", "net", True, workers, path_b)

    print(f"{'':<34}{'A':>10}{'B':>10}{'B - A':>11}{'std err':>11}")
    for desc, ra, rb in (("declarer: tricks taken", dec_a, dec_b),
                         ("defence: tricks conceded", def_a, def_b)):
        m, se, _ = _paired(ra, rb, "tricks")
        print(f"{desc:<34}{np.mean([r['tricks'] for r in ra]):>10.3f}"
              f"{np.mean([r['tricks'] for r in rb]):>10.3f}{m:>+11.3f}{se:>11.3f}")
    for wanted_role, name in (("wist", "lead"), ("obrona", "rest of defence")):
        oa, la, na = _role_stats(def_a, wanted_role)
        ob, lb, nb = _role_stats(def_b, wanted_role)
        se = np.sqrt(_binom_se(oa, na) ** 2 + _binom_se(ob, nb) ** 2)
        print(f"{name + ': % opt. sig.':<34}{oa:>10.1f}{ob:>10.1f}{ob - oa:>+11.1f}{se:>11.1f}")
        print(f"{name + ': loss per decision':<34}{la:>10.3f}{lb:>10.3f}{lb - la:>+11.3f}")
    for s in range(5):
        sa = [r for r in def_a if r["strain"] == s]
        sb = [r for r in def_b if r["strain"] == s]
        if len(sa) >= 5:
            m, se, _ = _paired(sa, sb, "tricks")
            print(f"{'defence ' + STRAIN_NAME[s]:<34}{np.mean([r['tricks'] for r in sa]):>10.3f}"
                  f"{np.mean([r['tricks'] for r in sb]):>10.3f}{m:>+11.3f}{se:>11.3f}")


def plot_vs_dd(recs, path):
    acc_a, acc_d = collections.defaultdict(list), collections.defaultdict(list)
    for r in recs:
        acc_a[r["hcp"]].append(r["tricks"])
        acc_d[r["hcp"]].append(r["dd_declarer"])
    hs = sorted(h for h in acc_a if len(acc_a[h]) >= 3)
    plt.figure(figsize=(10, 6))
    plt.plot(hs, [np.mean(acc_d[h]) for h in hs], "s--", c="#e74c3c", label="Double dummy (ten sam kontrakt)")
    plt.plot(hs, [np.mean(acc_a[h]) for h in hs], "o-", c="#2c3e50", label="Agent DMC (obrona DD)")
    plt.xlabel("PC strony rozgrywajacej")
    plt.ylabel("Srednia liczba lew")
    plt.title("Agent vs double dummy na identycznych rozdaniach")
    plt.legend()
    plt.grid(alpha=.4, ls=":")
    plt.savefig(path, dpi=160, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--play-model", default="models/play_dmc.pth")
    p.add_argument("--deals", type=int, default=300)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--seed0", type=int, default=9_000_000)
    p.add_argument("--no-audit", dest="audit", action="store_false")
    p.add_argument("--plot", default="dd_audit.png")
    p.add_argument("--compare")
    p.add_argument("--bidding", choices=["regula", "model"], default="regula")
    p.add_argument("--policy", default="models/best_policy.pth")
    a = p.parse_args()

    if a.bidding == "model":
        set_bidding("model", a.policy)
    seeds = list(range(a.seed0, a.seed0 + a.deals))
    if a.compare:
        run_compare(seeds, a.play_model, a.compare, a.workers)
        return

    main_recs = run_matchup(seeds, "net", "dd", a.audit, a.workers, a.play_model)
    summarize(main_recs, "agent declares")
    summarize(run_matchup(seeds, "heur", "dd", False, a.workers, a.play_model), "heuristic declares")
    def_recs = run_matchup(seeds, "dd", "net", a.audit, a.workers, a.play_model)
    summarize(def_recs, "agent defends")
    summarize(run_matchup(seeds, "dd", "heur", False, a.workers, a.play_model), "heuristic defends")
    summarize(run_matchup(seeds, "net", "heur", False, a.workers, a.play_model), "agent vs heuristic")

    x = np.array([r["hcp"] for r in main_recs], dtype=float)
    sa = np.polyfit(x, [r["tricks"] for r in main_recs], 1)[0]
    sd = np.polyfit(x, [r["dd_declarer"] for r in main_recs], 1)[0]
    print(f"\ntricks per HCP: agent {sa:.3f}, DD {sd:.3f} ({sa / sd * 100:.0f}%)")
    report_by_strain(main_recs)
    if a.audit:
        print("\nagent as declarer (DD defence)")
        report_audit(main_recs)
        print("\nagent as defender (DD declarer)")
        report_audit(def_recs)
    plot_vs_dd(main_recs, a.plot)


if __name__ == "__main__":
    main()
