import random
from typing import List
from AppUtils.cards_utils import RANKS, SUITS
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER

class Card:
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def __repr__(self):
        cards_symbols = {'h' : '\u2665', 'd' : '\u2666', 'c' : '\u2663', 's' : '\u2660'}
        card_str = f"{self.rank}{cards_symbols[self.suit]}"
        if self.suit == 'h' or self.suit == 'd':
            return f"\033[91m{card_str}\033[0m"
        else:
            return f"\033[90m{card_str}\033[0m" 

class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        random.shuffle(self.cards)

    def deal(self, num_cards: int) -> List[Card]:
      #  self.cards.sort(key=lambda x: (RANKS.index(x.rank), SUITS.index(x.suit)), reverse=False)
        return [self.cards.pop() for _ in range(num_cards)] 

    def deal_next_cards(self, street):
        return self.deal(3 if street == PREFLOP else 1)
    
    def reshuffle(self):
        random.shuffle(self.cards)

    def deal_players_cards(self, players):
        for player in players:
            player.set_hand(self.deal(2))
    
    def remove_cards(self, cards):
        for card in cards:
            if card in self.cards:
                self.cards.remove(card)
            else:
                print(f"{card} is already dealt")