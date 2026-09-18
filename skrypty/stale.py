"""Configuration constants from the code that are quoted in the thesis."""
import inspect

from wspolne import Report
from bridge.env import bridge_env as B
from bridge.env.play_features import CARD_FEAT_DIM
from bridge.licytacja import deal_value as DV
from bridge.licytacja import train_ctde as C
from bridge.licytacja import trening_cfr as CFR
from bridge.nets.ctde import N_HIDDEN_CARDS, N_PARTNER_FEATURES, BidActor, BidCritic
from bridge.nets.models import HierAdvantageNetwork, PlayQNetwork, PolicyNetwork
from bridge.rozgrywka import train_play_dmc as D


def n_params(m):
    return f"{sum(p.numel() for p in m.parameters()):,}".replace(",", " ")


def main():
    P = Report("stale")
    P("SRODOWISKO")
    P(f"  dlugosc opisu sytuacji (wektor obserwacji): {B.OBS_SIZE}")
    P(f"  liczba akcji wspolnej listy: {B.NUM_ACTIONS} (odzywek {B.NUM_BIDS}, pas/kontra/rekontra: "
      f"{B.PASS}/{B.DOUBLE}/{B.REDOUBLE})")

    d = inspect.signature(D.train_dmc).parameters
    P("\nFAZA 0")
    P(f"  liczba cech karty: {CARD_FEAT_DIM}")
    P(f"  domyslna liczba epok: {d['epochs'].default}, rozdan na epoke: {d['episodes_per_epoch'].default}")
    P(f"  losowe zagrania: od {d['eps_start'].default:.0%} do {d['eps_end'].default:.0%} decyzji, malejaco liniowo")
    P(f"  wag w jednej sieci rozgrywki: {n_params(PlayQNetwork())}")
    P(f"  partia uczaca: {d['batch'].default}, przejsc po danych epoki: {d['passes'].default}, "
      f"krok uczenia: {d['lr'].default:g} malejacy kosinusowo")

    P("\nFAZA 1 - wspolne")
    P(f"  skala nagrody (dzielnik w tanh): {DV.REWARD_SCALE:.0f}")
    P(f"  kontraktow na rozdanie w tablicy 5x4: 7 poziomow x 5 mian x 4 rozgrywajacych x 3 "
      f"(bez kontry/kontra/rekontra) = {7 * 5 * 4 * 3}")

    c = inspect.signature(CFR.train_phase_1_cfr).parameters
    P("\nFAZA 1 - Deep CFR")
    P(f"  wag w sieci zalu: {n_params(HierAdvantageNetwork())}")
    P(f"  wag w sieci strategii sredniej: {n_params(PolicyNetwork())}")
    P(f"  rozdan bazowych na epoke: {c['deals_per_epoch'].default}, kazde rozdawane ponownie {CFR.REDEALS} razy, "
      f"{CFR.TRAVERSALS_PER_DEAL} przejscia na rozdanie")
    P(f"  partia uczaca: {c['batch_size'].default}, krokow uczenia na epoke: {c['grad_steps'].default}")
    P("  pojemnosc kazdego z dwoch zbiorow probek: 3 000 000")

    t = inspect.signature(C.train_phase_1_ctde).parameters
    P("\nFAZA 1 - aktor i krytyk")
    P(f"  wag w aktorze: {n_params(BidActor())}")
    P(f"  wag w krytyku: {n_params(BidCritic())}")
    P(f"  ukryte karty zgadywane przez aktora: {N_HIDDEN_CARDS} (3 rece x 52)")
    P(f"  cechy partnera zgadywane przez aktora: {N_PARTNER_FEATURES} (punkty + 4 dlugosci)")
    P(f"  licytacji z jednego rozdania: {C.AUCTIONS_PER_DEAL}")
    P(f"  rozdan na epoke: {t['deals_per_epoch'].default}, przejsc po danych epoki: {C.PPO_EPOCHS}, "
      f"partia uczaca: {C.BATCH}")
    P(f"  krok uczenia: aktor {C.LR_ACTOR:g}, krytyk {C.LR_CRITIC:g}; ograniczenie kroku PPO: {C.PPO_CLIP}")
    P(f"  wagi zadan dodatkowych aktora: karty {C.W_CARDS}, cechy partnera {C.W_PARTNER}")
    P(f"  przechylenie w strone pasa na starcie: {C.PASS_BIAS}")
    P(f"  premia za niezdecydowanie (entropie): od {C.ENTROPY_START} do {C.ENTROPY_END} "
      f"przez {C.ENTROPY_DECAY_EPOCHS} epok")
    P(f"  zapis stanu modelu co {C.CKPT_EVERY} epok")
    P(f"  sprawdzian co {t['val_every'].default} epok na {t['val_deals'].default} rozdaniach")
    P.close()


if __name__ == "__main__":
    main()
