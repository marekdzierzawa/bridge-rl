"""Hierarchical regret matching (category, strain, level) and targets for the Deep CFR regret network."""
import numpy as np
import torch

from bridge.env.bridge_env import DOUBLE, NUM_ACTIONS, NUM_BIDS, PASS, REDOUBLE

N_CAT = 4
CAT_PASS, CAT_DBL, CAT_RDBL, CAT_BID = 0, 1, 2, 3
N_STRAIN = 5
N_LEVEL = NUM_BIDS // N_STRAIN

RM_PRUNE_EPS = 0.25
RM_PRUNE_EPS_CAT = 0.0


def _rm(x, idx, eps):
    out = np.zeros(x.shape[0], dtype=np.float64)
    if idx.size == 0:
        return out
    if idx.size == 1:
        out[idx[0]] = 1.0
        return out
    v = x[idx]
    mx = float(np.max(v))
    if mx <= 0.0:
        out[idx[int(np.argmax(v))]] = 1.0
        return out
    p = np.clip(v - eps * mx, 0.0, None)
    s = p.sum()
    if s <= 0.0:
        out[idx[int(np.argmax(v))]] = 1.0
        return out
    out[idx] = p / s
    return out


def split_legal(mask):
    legal = np.flatnonzero(mask)
    bids = legal[legal < NUM_BIDS]
    return bids, bool(mask[PASS]), bool(mask[DOUBLE]), bool(mask[REDOUBLE])


def cat_mask(mask):
    bids, has_pass, has_d, has_r = split_legal(mask)
    return np.array([has_pass, has_d, has_r, bids.size > 0], dtype=bool)


def hier_targets_torch(probs, mask, scale=0.5):
    p = probs * mask
    p = p / p.sum(1, keepdim=True).clamp(min=1e-12)
    b = p.shape[0]

    bid = p[:, :NUM_BIDS]
    bid_tot = bid.sum(1)
    t_cat = torch.stack([p[:, PASS], p[:, DOUBLE], p[:, REDOUBLE], bid_tot], 1)

    by_s = bid.view(b, N_LEVEL, N_STRAIN).sum(1)
    t_str = by_s / bid_tot.unsqueeze(1).clamp(min=1e-12)

    t_act = torch.zeros_like(p)
    t_act[:, :NUM_BIDS] = (bid.view(b, N_LEVEL, N_STRAIN)
                           / by_s.unsqueeze(1).clamp(min=1e-12)).view(b, NUM_BIDS)
    t_act[:, DOUBLE] = mask[:, DOUBLE]
    t_act[:, REDOUBLE] = mask[:, REDOUBLE]
    return t_cat * scale, t_str * scale, t_act * scale


def hier_sigma(cat_d, strain_d, act_d,
               mask, eps=RM_PRUNE_EPS, eps_cat=RM_PRUNE_EPS_CAT):
    sigma = np.zeros(NUM_ACTIONS, dtype=np.float64)
    pa = np.zeros(NUM_ACTIONS, dtype=np.float64)
    ps = np.zeros(N_STRAIN, dtype=np.float64)
    bids, has_pass, has_d, has_r = split_legal(mask)

    pc = _rm(np.asarray(cat_d, dtype=np.float64), np.flatnonzero(cat_mask(mask)), eps_cat)
    if has_pass:
        sigma[PASS] = pc[CAT_PASS]
    if has_d:
        sigma[DOUBLE] = pc[CAT_DBL]
    if has_r:
        sigma[REDOUBLE] = pc[CAT_RDBL]

    a = np.asarray(act_d, dtype=np.float64)
    if bids.size:
        st_idx = np.unique(bids % 5)
        ps = _rm(np.asarray(strain_d, dtype=np.float64), st_idx, eps)
        for s in st_idx:
            members = bids[bids % 5 == s]
            pl = _rm(a, members, eps)
            pa[members] = pl[members]
            sigma[members] = pc[CAT_BID] * ps[s] * pl[members]

    tot = sigma.sum()
    if tot <= 1e-12:
        sigma = mask.astype(np.float64)
        tot = sigma.sum()
    return sigma / tot, (pc, ps, pa)


def hier_regrets(v, parts, mask):
    v = np.asarray(v, dtype=np.float64)
    pc, ps, pa = parts
    r_cat = np.zeros(N_CAT, dtype=np.float32)
    r_str = np.zeros(N_STRAIN, dtype=np.float32)
    r_act = np.zeros(NUM_ACTIONS, dtype=np.float32)
    bids, has_pass, has_d, has_r = split_legal(mask)

    v_bid = 0.0
    v_str = np.zeros(N_STRAIN, dtype=np.float64)
    st_idx = np.unique(bids % 5) if bids.size else np.empty(0, dtype=int)
    for s in st_idx:
        members = bids[bids % 5 == s]
        v_str[s] = float(np.dot(pa[members], v[members]))
    if bids.size:
        v_bid = float(np.dot(ps[st_idx], v_str[st_idx]))

    v_node = (pc[CAT_PASS] * (v[PASS] if has_pass else 0.0)
              + pc[CAT_DBL] * (v[DOUBLE] if has_d else 0.0)
              + pc[CAT_RDBL] * (v[REDOUBLE] if has_r else 0.0)
              + pc[CAT_BID] * v_bid)

    if has_pass:
        r_cat[CAT_PASS] = v[PASS] - v_node
    if has_d:
        r_cat[CAT_DBL] = v[DOUBLE] - v_node
    if has_r:
        r_cat[CAT_RDBL] = v[REDOUBLE] - v_node
    if bids.size:
        r_cat[CAT_BID] = v_bid - v_node
        r_str[st_idx] = (v_str[st_idx] - v_bid).astype(np.float32)
        for s in st_idx:
            members = bids[bids % 5 == s]
            r_act[members] = v[members] - v_str[s]
    return r_cat, r_str, r_act


def hier_masks_torch(mask):
    bid = mask[:, :NUM_BIDS]
    m_cat = torch.stack([mask[:, PASS], mask[:, DOUBLE], mask[:, REDOUBLE],
                         bid.sum(-1)], dim=-1).clamp(max=1.0)
    m_str = bid.reshape(-1, N_LEVEL, N_STRAIN).sum(1).clamp(max=1.0)
    m_act = mask.clone()
    m_act[:, PASS] = 0.0
    m_act[:, DOUBLE] = 0.0
    m_act[:, REDOUBLE] = 0.0
    return m_cat, m_str, m_act
