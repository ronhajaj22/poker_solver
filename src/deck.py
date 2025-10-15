import random
from collections import Counter
from itertools import combinations
from typing import List
from AppUtils.cards_utils import RANKS, SUITS, CARD_RANK_TO_INT_VALUE
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER

class Card:
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

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
    
    def deal_players_cards(self, players):
        for player in players:
            player.set_hand(self.deal(2))

    def get_hand_rank(self, cards):
        best_rank = -1
        best_kicker = []
        best_hand = []

        for combo in combinations(cards, 5):
            ranks = sorted([CARD_RANK_TO_INT_VALUE[c.rank] for c in combo], reverse=True)
            suits = [c.suit for c in combo]
            rank_counts = Counter(ranks)

            is_flush = len(set(suits)) == 1
            is_straight = len(set(ranks)) == 5 and max(ranks) - min(ranks) == 4
            counts = sorted(rank_counts.values(), reverse=True)
            sorted_ranks_by_count = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))
            main_ranks = [r for r, _ in sorted_ranks_by_count]

            if is_straight and is_flush:
                rank_index = 8
            elif counts == [4, 1]:
                rank_index = 7
            elif counts == [3, 2]:
                rank_index = 6
            elif is_flush:
                rank_index = 5
            elif is_straight:
                rank_index = 4
            elif counts == [3, 1, 1]:
                rank_index = 3
            elif counts == [2, 2, 1]:
                rank_index = 2
            elif counts == [2, 1, 1, 1]:
                rank_index = 1
            else:
                rank_index = 0

            if rank_index > best_rank or (rank_index == best_rank and main_ranks > best_kicker):
                best_rank = rank_index
                best_kicker = main_ranks
                best_hand = combo

        return best_rank, best_kicker, best_hand

    def find_winners(self, players, community_cards):
        hands = []

        for player in players:
            all_cards = player.hand.cards + community_cards;
            rank, kicker, best_hand = self.get_hand_rank(all_cards)
            hands.append({
                "rank": rank,
                "kicker": kicker,
                "hand": best_hand,
                "player": player.name
            })

        hands.sort(key=lambda x: (x["rank"], x["kicker"]), reverse=True)
        
        # Get the best hand
        best_rank = hands[0]["rank"]
        best_kicker = hands[0]["kicker"]
        
        # Find all players with the same best hand (ties)
        winning_hands = [h for h in hands if h["rank"] == best_rank and h["kicker"] == best_kicker]
        winning_players = [player for player in players if any(h["player"] == player.name for h in winning_hands)]
        
        # Return array of winners and the winning hand info
        return winning_players, winning_hands
    