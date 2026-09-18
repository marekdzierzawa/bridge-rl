"""Środowisko brydża: licytacja, rozgrywka, zapis turniejowy i obserwacje graczy."""
from dataclasses import dataclass, field

import numpy as np

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
SEAT_NAMES = ["North", "East", "South", "West"]
CLUBS, DIAMONDS, HEARTS, SPADES, NOTRUMP = 0, 1, 2, 3, 4
SUIT_CH = "CDHS"
RANK_CH = "23456789TJQKA"
STRAIN_CH = ["C", "D", "H", "S", "NT"]
NUM_CARDS, NUM_BIDS = 52, 35
PASS, DOUBLE, REDOUBLE = 35, 36, 37
CARD_OFFSET = 38
NUM_ACTIONS = CARD_OFFSET + NUM_CARDS
AUCTION, PLAY = 0, 1
MAX_ABS_SCORE = 500.0

OWN_HAND, DUMMY_HAND = 0, 52
VUL, PHASE = 104, 106
BID_HIST = 108
CONTRACT = 528
CUR_TRICK = 569
SUIT_LED = 777
PLAYED = 781
TRICKS, DUMMY_FLAG = 989, 991
VOID = 992
TACT = 1008
OBS_SIZE = 1020


def card_suit(c):
    return c // 13


def card_rank(c):
    return c % 13


def card_str(c):
    return RANK_CH[card_rank(c)] + SUIT_CH[card_suit(c)]


def action_str(a):
    if a < NUM_BIDS:
        return f"{a // 5 + 1}{STRAIN_CH[a % 5]}"
    if a == PASS:
        return "Pass"
    if a == DOUBLE:
        return "Dbl"
    if a == REDOUBLE:
        return "Rdbl"
    return "Play " + card_str(a - CARD_OFFSET)


class Deck:
    def __init__(self, rng):
        self._rng = rng

    def deal(self):
        order = self._rng.permutation(NUM_CARDS)
        return [set(int(c) for c in order[i * 13:(i + 1) * 13]) for i in range(4)]


@dataclass
class AuctionState:
    dealer: int
    calls: list[tuple[int, int]] = field(default_factory=list)
    last_bid: int = None
    last_bidder: int = None
    doubled: bool = False
    redoubled: bool = False
    first_to_bid_strain: dict[tuple[int, int], int] = field(default_factory=dict)

    def current_player(self):
        return (self.dealer + len(self.calls)) % 4

    def legal_calls(self):
        p = self.current_player()
        legal = [PASS]
        start = 0 if self.last_bid is None else self.last_bid + 1
        legal.extend(range(start, NUM_BIDS))
        if self.last_bid is not None:
            same_side = (self.last_bidder % 2) == (p % 2)
            if not self.doubled and not same_side:
                legal.append(DOUBLE)
            if self.doubled and not self.redoubled and same_side:
                legal.append(REDOUBLE)
        return sorted(legal)

    def apply(self, action):
        p = self.current_player()
        self.calls.append((p, action))
        if action < NUM_BIDS:
            self.last_bid, self.last_bidder = action, p
            self.doubled = self.redoubled = False
            self.first_to_bid_strain.setdefault((p % 2, action % 5), p)
        elif action == DOUBLE:
            self.doubled = True
        elif action == REDOUBLE:
            self.redoubled = True

    def is_over(self):
        n = len(self.calls)
        if self.last_bid is not None:
            return n >= 4 and all(a == PASS for _, a in self.calls[-3:])
        return n >= 4

    @property
    def passed_out(self):
        return self.is_over() and self.last_bid is None

    def contract(self):
        level, strain = self.last_bid // 5 + 1, self.last_bid % 5
        declarer = self.first_to_bid_strain[(self.last_bidder % 2, strain)]
        return level, strain, self.doubled, self.redoubled, declarer


@dataclass
class Trick:
    leader: int
    plays: list[tuple[int, int]] = field(default_factory=list)

    def suit_led(self):
        return card_suit(self.plays[0][1]) if self.plays else None

    def seat_to_play(self):
        return (self.leader + len(self.plays)) % 4

    def is_complete(self):
        return len(self.plays) == 4

    @staticmethod
    def _beats(c, best, trump):
        if card_suit(c) == card_suit(best):
            return card_rank(c) > card_rank(best)
        return card_suit(c) == trump

    def winner(self, trump):
        w_seat, w_card = self.plays[0]
        for seat, c in self.plays[1:]:
            if self._beats(c, w_card, trump):
                w_seat, w_card = seat, c
        return w_seat


def duplicate_score(level, strain, doubled, redoubled,
                    vulnerable, tricks_taken):
    target = 6 + level
    if tricks_taken >= target:
        if strain in (CLUBS, DIAMONDS):
            base = 20 * level
        elif strain in (HEARTS, SPADES):
            base = 30 * level
        else:
            base = 40 + 30 * (level - 1)
        if redoubled:
            base *= 4
        elif doubled:
            base *= 2
        score = base
        score += (500 if vulnerable else 300) if base >= 100 else 50
        if level == 6:
            score += 750 if vulnerable else 500
        elif level == 7:
            score += 1500 if vulnerable else 1000
        if redoubled:
            score += 100
        elif doubled:
            score += 50
        over = tricks_taken - target
        if over > 0:
            if redoubled:
                score += over * (400 if vulnerable else 200)
            elif doubled:
                score += over * (200 if vulnerable else 100)
            else:
                score += over * (20 if strain in (CLUBS, DIAMONDS) else 30)
        return score

    down = target - tricks_taken
    if not doubled and not redoubled:
        return -down * (100 if vulnerable else 50)
    if vulnerable:
        penalty = 200 + (down - 1) * 300
    else:
        penalty = 100 + 200 * min(down - 1, 2) + 300 * max(down - 3, 0)
    return -(penalty * 2 if redoubled else penalty)


class BridgeEnv:
    def __init__(self, seed=None, normalize_rewards=True):
        self._rng = np.random.default_rng(seed)
        self.normalize_rewards = normalize_rewards
        self._deal_count = 0

    def reset(self, dealer=None, vulnerable=None, hands=None):
        self.dealer = dealer if dealer is not None else self._deal_count % 4
        self._deal_count += 1
        if vulnerable is None:
            vulnerable = (bool(self._rng.integers(2)), bool(self._rng.integers(2)))
        self.vulnerable = vulnerable
        self.hands = [set(h) for h in hands] if hands is not None else Deck(self._rng).deal()
        self.auction = AuctionState(self.dealer)
        self.phase = AUCTION
        self.level = self.strain = self.declarer = self.dummy = None
        self.contract_doubled = self.contract_redoubled = False
        self.dummy_revealed = False
        self.current_trick = None
        self.completed_tricks = []
        self.tricks_won = [0, 0]
        self.terminated = False
        self.rewards = {p: 0.0 for p in range(4)}
        self.info = {}
        return self.observe(self.current_player)

    @property
    def current_player(self):
        if self.terminated:
            return None
        if self.phase == AUCTION:
            return self.auction.current_player()
        seat = self.current_trick.seat_to_play()
        return self.declarer if seat == self.dummy else seat

    def _seat_to_play(self):
        return self.current_trick.seat_to_play()

    def legal_actions(self):
        if self.terminated:
            return []
        if self.phase == AUCTION:
            return self.auction.legal_calls()
        hand = self.hands[self._seat_to_play()]
        led = self.current_trick.suit_led()
        follow = [c for c in hand if card_suit(c) == led] if led is not None else []
        return sorted(CARD_OFFSET + c for c in (follow or hand))

    def get_action_mask(self):
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        mask[self.legal_actions()] = 1
        return mask

    def step(self, action):
        if self.phase == AUCTION:
            self.auction.apply(action)
            if self.auction.is_over():
                if self.auction.passed_out:
                    self._finish(passed_out=True)
                else:
                    (self.level, self.strain, self.contract_doubled,
                     self.contract_redoubled, self.declarer) = self.auction.contract()
                    self.dummy = (self.declarer + 2) % 4
                    self.phase = PLAY
                    self.current_trick = Trick(leader=(self.declarer + 1) % 4)
        else:
            seat = self._seat_to_play()
            card = action - CARD_OFFSET
            self.hands[seat].remove(card)
            self.current_trick.plays.append((seat, card))
            self.dummy_revealed = True
            if self.current_trick.is_complete():
                w = self.current_trick.winner(self.strain)
                self.tricks_won[w % 2] += 1
                self.completed_tricks.append(self.current_trick)
                if len(self.completed_tricks) == 13:
                    self._finish(passed_out=False)
                else:
                    self.current_trick = Trick(leader=w)
        obs = (np.zeros(OBS_SIZE, dtype=np.float32) if self.terminated
               else self.observe(self.current_player))
        return obs, dict(self.rewards), self.terminated, dict(self.info)

    def _finish(self, passed_out):
        self.terminated = True
        if passed_out:
            self.info = {"contract": "All Passes", "score": 0}
            return

        side = self.declarer % 2
        tricks = self.tricks_won[side]
        score = duplicate_score(self.level, self.strain, self.contract_doubled,
                                self.contract_redoubled, self.vulnerable[side], tricks)
        for p in range(4):
            raw = score if p % 2 == side else -score
            self.rewards[p] = float(np.tanh(raw / MAX_ABS_SCORE)) if self.normalize_rewards else raw

        dbl = "XX" if self.contract_redoubled else "X" if self.contract_doubled else ""
        self.info = {
            "contract": f"{self.level}{STRAIN_CH[self.strain]}{dbl} by {SEAT_NAMES[self.declarer]}",
            "tricks_taken": tricks, "target": 6 + self.level,
            "made": tricks >= 6 + self.level, "score_declarer_side": score,
        }

    def observe(self, player):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        def rel(p):
            return (p - player) % 4

        for c in self.hands[player]:
            obs[OWN_HAND + c] = 1.0
        if self.dummy_revealed:
            for c in self.hands[self.dummy]:
                obs[DUMMY_HAND + c] = 1.0
        obs[VUL] = float(self.vulnerable[player % 2])
        obs[VUL + 1] = float(self.vulnerable[(player + 1) % 2])
        obs[PHASE + self.phase] = 1.0

        last = None
        for seat, a in self.auction.calls:
            if a < NUM_BIDS:
                obs[BID_HIST + a * 12 + rel(seat)] = 1.0
                last = a
            elif a == DOUBLE:
                obs[BID_HIST + last * 12 + 4 + rel(seat)] = 1.0
            elif a == REDOUBLE:
                obs[BID_HIST + last * 12 + 8 + rel(seat)] = 1.0

        if self.phase != PLAY:
            return obs

        obs[CONTRACT + (self.level - 1) * 5 + self.strain] = 1.0
        obs[CONTRACT + 35] = float(self.contract_doubled)
        obs[CONTRACT + 36] = float(self.contract_redoubled)
        obs[CONTRACT + 37 + rel(self.declarer)] = 1.0
        for seat, c in self.current_trick.plays:
            obs[CUR_TRICK + rel(seat) * 52 + c] = 1.0
        led = self.current_trick.suit_led()
        if led is not None:
            obs[SUIT_LED + led] = 1.0
        for trick in self.completed_tricks:
            led_s = card_suit(trick.plays[0][1])
            for seat, c in trick.plays:
                obs[PLAYED + rel(seat) * 52 + c] = 1.0
                if card_suit(c) != led_s:
                    obs[VOID + rel(seat) * 4 + led_s] = 1.0
        if led is not None:
            for seat, c in self.current_trick.plays:
                if card_suit(c) != led:
                    obs[VOID + rel(seat) * 4 + led] = 1.0
        obs[TRICKS] = self.tricks_won[player % 2] / 13.0
        obs[TRICKS + 1] = self.tricks_won[(player + 1) % 2] / 13.0

        seen = np.zeros(NUM_CARDS, dtype=bool)
        vis_len = np.zeros(4, dtype=np.float32)
        for c in self.hands[player]:
            seen[c] = True
            vis_len[card_suit(c)] += 1
        if self.dummy_revealed and self.dummy != player:
            for c in self.hands[self.dummy]:
                seen[c] = True
                vis_len[card_suit(c)] += 1
        for trick in self.completed_tricks:
            for _, c in trick.plays:
                seen[c] = True
        for _, c in self.current_trick.plays:
            seen[c] = True
        unseen = ~seen
        for s in range(4):
            obs[TACT + s] = vis_len[s] / 13.0
            obs[TACT + 4 + s] = float(unseen[s * 13:(s + 1) * 13].sum()) / 13.0
        if self.strain != NOTRUMP:
            obs[TACT + 8] = vis_len[self.strain] / 13.0
            obs[TACT + 9] = float(unseen[self.strain * 13:(self.strain + 1) * 13].sum()) / 13.0
        my_side = player % 2
        goal = (6 + self.level) if my_side == self.declarer % 2 else (14 - (6 + self.level))
        obs[TACT + 10] = max(0, goal - self.tricks_won[my_side]) / 13.0
        obs[TACT + 11] = (13 - len(self.completed_tricks)) / 13.0
        if self._seat_to_play() == self.dummy and player == self.declarer:
            obs[DUMMY_FLAG] = 1.0
        return obs
