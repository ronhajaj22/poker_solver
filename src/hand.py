from deck import RANKS, SUITS
from deck import Card
from typing import List
from deck import RANKS

class Hand:
    HAND_RANKS = {
        "High Card": 0,
        "Pair": 1,
        "Two Pair": 2,
        "Three of a Kind": 3,
        "Straight": 4,
        "Flush": 5,
        "Full House": 6,
        "Four of a Kind": 7,
        "Straight Flush": 8,
    }
    
    def __init__(self, cards: List[Card]):
        self.cards = cards
            
    def __repr__(self):
        return ' '.join(str(card) for card in self.cards)
    
    def to_string(self, cards = []):
        if cards == []:
            cards = self.cards

        card1, card2 = cards[0], cards[1]
        ranks = [card1.rank, card2.rank]
        suits = [card1.suit, card2.suit]

        # Sort by rank strength (highest first)
        rank_order = {r: i for i, r in enumerate(['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'])}
        if rank_order[ranks[1]] < rank_order[ranks[0]]:
            ranks = [ranks[1], ranks[0]]
            suits = [suits[1], suits[0]]

        if ranks[0] == ranks[1]:
            return ranks[0] + ranks[1]  # e.g., 'TT'
        elif suits[0] == suits[1]:
            return ranks[0] + ranks[1] + 's'  # e.g., 'ATs'
        else:
            return ranks[0] + ranks[1] + 'o'  # e.g., 'QTo'