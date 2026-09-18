"""Reading and writing the validation history of bidding training runs (CSV file)."""
import csv
import os


def append_row(path, row):
    is_new = not os.path.exists(path)
    fields = list(row)
    if not is_new:
        with open(path, newline="", encoding="utf-8") as f:
            fields = next(csv.reader(f))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        if is_new:
            w.writeheader()
        w.writerow(row)


def load_history(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items() if v} for r in csv.DictReader(f)]


def column(rows, name):
    e = [w["epoka"] for w in rows if name in w]
    v = [w[name] for w in rows if name in w]
    return e, v


def ctde_row(epoch, r):
    rep = r.get("repertoire") or {}
    w = {
        "epoka": epoch,
        "strata": round(r["loss"], 2),
        "strata_se": round(r["se"], 2),
        "strata_pas": round(r["pass_loss"], 1),
        "poziom": round(r["level"], 3),
        "pas": round(r["pass"], 2),
        "poz6plus": round(r["level6plus"], 2),
        "ba": round(r["strain_hist"].get("NT", 0.0), 2),
        "kontra": round(r["doubled"], 2),
        "entropia": round(r["entropy"], 4),
        "rep_naj": round(rep.get("top", float("nan")), 2),
        "rep_na80": rep.get("n80", ""),
        "rep_roznych": rep.get("distinct", ""),
        "krytyk_r2": round(r.get("critic_r2", float("nan")), 4),
        "pc_siec_po": round(r.get("hcp_net_bid", float("nan")), 3),
        "pc_reka_po": round(r.get("hcp_hand_bid", float("nan")), 3),
    }
    if r.get("hcp_hand_bid") == r.get("hcp_hand_bid") and r.get("hcp_net_bid") == r.get("hcp_net_bid"):
        w["konwencja"] = round(r["hcp_hand_bid"] - r["hcp_net_bid"], 3)
    k = r.get("shape")
    if k:
        w["dl_siec_H"] = round(float(k["net"][2]), 3)
        w["dl_siec_S"] = round(float(k["net"][3]), 3)
        w["dl_reka_H"] = round(float(k["hand"][2]), 3)
        w["dl_reka_S"] = round(float(k["hand"][3]), 3)
        if "majors_net" in k:
            w["starsze_siec"] = round(k["majors_net"], 2)
            w["starsze_reka"] = round(k["majors_hand"], 2)
    for lv in range(1, 8):
        w[f"poziom{lv}"] = round(r["level_hist"].get(lv, 0.0), 2)
    return w


def cfr_row(epoch, g, d, pass_loss):
    rep = g.get("repertoire") or {}
    w = {
        "epoka": epoch,
        "strata": round(g["loss"], 2),
        "strata_se": round(g["loss_se"], 2),
        "strata_pas": round(pass_loss, 1),
        "poziom": round(g["level"], 3),
        "pas": round(g["pass"], 2),
        "poz6plus": round(g["tail6"], 2),
        "ba": round(g["nt"], 2),
        "kontra": round(g["dbl"], 2),
        "entropia": round(g["ent"], 4),
        "rep_naj": round(rep.get("top", float("nan")), 2),
        "rep_na80": rep.get("n80", ""),
        "rep_roznych": rep.get("distinct", ""),
        "strata_t1": round(d["loss"], 2),
        "zapis": round(g["score"], 2),
        "major4": round(g["major4"], 2),
        "dlugosc": round(g["len"], 2),
    }
    for lv in range(1, 8):
        w[f"poziom{lv}"] = round(g["hist"][lv - 1], 2)
    return w
