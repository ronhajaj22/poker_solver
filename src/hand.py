from typing import List, Optional
from deck import Card
from AppUtils.cards_utils import RANKS

#NOTE: this class is redundant - can be removed
class Hand:
    def __init__(self, cards: List[Card]):
        self.cards = cards
            
    def __repr__(self):
        return ' '.join(str(card) for card in self.cards)
    
    def to_string(self, cards: Optional[List[Card]] = None) -> str:
        if cards is None:
            cards = self.cards

        # Sort by rank strength (highest first)
        sorted_cards = sorted(cards, key=lambda card: RANKS.index(card.rank), reverse=True)
        rank1, rank2 = sorted_cards[0].rank, sorted_cards[1].rank
        suit1, suit2 = sorted_cards[0].suit, sorted_cards[1].suit

        if rank1 == rank2:
            return f"{rank1}{rank2}"  # e.g., 'TT'
        elif suit1 == suit2:
            return f"{rank1}{rank2}s"  # e.g., 'ATs'
        else:
            return f"{rank1}{rank2}o"  # e.g., 'QTo'

