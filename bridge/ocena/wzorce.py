"""Reference values: expert contract distributions, loss of the simple system and the repertoire measure."""
EXPERT_LEVEL = {1: 7.1, 2: 15.9, 3: 36.3, 4: 30.0, 5: 6.0, 6: 4.4, 7: 0.4}
EXPERT_STRAIN = {"C": 7.2, "D": 9.4, "H": 26.2, "S": 27.1, "NT": 30.1}
EXPERT_DOUBLE = 4.9
EXPERT_PASS = 0.7
EXPERT_MEAN_LEVEL = 3.27
HEURISTIC_LOSS = 320.0
PAR_REPERTOIRE_80 = 18
STRAIN_CODES = ["C", "D", "H", "S", "NT"]


def repertoire(contracts):
    if not contracts:
        return None
    counts = {}
    for k in contracts:
        counts[k] = counts.get(k, 0) + 1
    shares = sorted((v / len(contracts) for v in counts.values()), reverse=True)
    cumulative, count = 0.0, 0
    for u in shares:
        cumulative += u
        count += 1
        if cumulative >= 0.8:
            break
    return {"top": shares[0] * 100, "n80": count, "distinct": len(counts)}
