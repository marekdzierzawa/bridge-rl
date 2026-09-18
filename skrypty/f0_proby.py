"""Phase 0: mechanisms of successive training versions and the architecture of saved networks."""
import os
import re

import torch

from wspolne import ROOT, Report
from bridge.env.play_features import CARD_FEAT_DIM, FEATURES_VERSION

HISTORIC_CODE_DIR = os.path.join(ROOT, "praca", "kod_historyczny")
HISTORIC_FILES = [
    ("1. PPO, licytacja losowa", "1_ppo_losowa_licytacja.py"),
    ("2. PPO, licytacja z sily", "2_ppo_licytacja_z_sily.py"),
    ("3. DMC, karta na wejsciu", "3_dmc_karta_jako_wejscie.py"),
    ("4. DMC, siec rezydualna", "4_dmc_resnet_belief.py"),
]
CURRENT_FILES = ["bridge/rozgrywka/train_play_dmc.py", "bridge/nets/models.py",
                 "bridge/rozgrywka/dmc_fixes.py", "praca/skrypty/wspolne.py"]

MECHANISMS = [
    ("karta jako wejscie sieci", "card_enc"),
    ("osobne sieci rozgr./obrona", "def_net"),
    ("cel: lewy od tej lewy do konca", "suf[s, t + 1]"),
    ("nagroda: jedna na cale rozdanie", "rewards.append"),
    ("zgadywanie ukrytych kart", "belief_head"),
    ("bloki rezydualne", "class ResBlock"),
    ("AdamW + kosinusowy krok", "CosineAnnealingLR"),
    ("praca na wielu procesach", "ProcessPoolExecutor"),
    ("ocena na stalym przeciwniku", "heuristic_defense"),
    ("ocena wobec gry idealnej", "eval_dd"),
]
AUCTION_MARKERS = [
    ("np.random.choice(legal_actions)", "kazda odzywka losowa"),
    ("declarer_side = 0 if ns_hcp >= 20", "kontrakt dobierany do sily reki"),
    ("force_auction", "funkcja force_auction (jej owczesnej wersji nie zachowano)"),
]
MODEL_FILES = ["models/play_dmc_32cech.pth", "models/play_dmc_40cech.pth", "models/play_dmc.pth"]


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def main():
    P = Report("f0_proby")
    names = [desc for desc, _ in HISTORIC_FILES] + ["5. obecna"]
    texts = [read_text(os.path.join(HISTORIC_CODE_DIR, fname)) for _, fname in HISTORIC_FILES]
    texts.append("\n".join(read_text(os.path.join(ROOT, p)) for p in CURRENT_FILES))

    P("1. mechanizmy w kolejnych wersjach (+ obecny, . brak)")
    P("   %-34s %s" % ("mechanizm", "  ".join(f"{i + 1}" for i in range(len(texts)))))
    for desc, pattern in MECHANISMS:
        P("   %-34s %s" % (desc, "  ".join("+" if pattern in t else "." for t in texts)))

    P("\n   kontrakt:")
    for t, nz in zip(texts[:-1], names):
        desc = "?"
        for pattern, name in AUCTION_MARKERS:
            if pattern in t:
                desc = name
        P("   %-34s %s" % (nz, desc))
    P("   %-34s %s" % (names[-1], "force_auction, wersja obecna"))

    P("\n   liczby z kodu:")
    for t, nz in zip(texts, names):
        vals = []
        for key in ("epochs=", "episodes_per_epoch=", "batch=", "eps_end=", "lr="):
            m = re.search(r"(?<![A-Za-z_])" + key, t)
            if m:
                vals.append(key + re.match(r"[^,)\n ]*", t[m.end():]).group())
        P(f"   {nz:34s} " + " | ".join(vals))

    P("\n2. sieci fazy 0")
    P("   %-30s %-16s %12s %14s" % ("plik", "budowa", "cech karty", "wersja cech"))
    for fname in MODEL_FILES:
        ck = torch.load(os.path.join(ROOT, fname), map_location="cpu", weights_only=True)
        sd = ck["dec"]
        arch = "rezydualna" if "res1.net.0.weight" in sd else "prosta"
        P("   %-30s %-16s %12d %14s" % (fname, arch, sd["card_enc.0.weight"].shape[1],
                                          ck.get("wersja_cech", "brak znacznika")))
    P(f"   obecny kod: {CARD_FEAT_DIM} cech, wersja {FEATURES_VERSION}")
    P.close()


if __name__ == "__main__":
    main()
