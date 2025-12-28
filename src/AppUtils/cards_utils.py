""" CARDS UTILS ===
Utility module for card format conversions and poker hand operations.

Provides conversion functions between different card formats (Card objects ↔ treys integers ↔ strings)
and helper utilities for working with cards, ranges, and preflop combinations.

FUNCTIONS (signatures):
- create_board_trey(board_cards, street=RIVER) - Convert board to treys format
- create_player_cards_trey(player_cards) - Convert player cards to treys format
- is_already_trey(cards) - Check if cards are already in treys format

- trey_to_card_list(hand) - Convert treys tuple to list of Card objects
- trey_to_str(hand) - Convert treys tuple to hand string (e.g., 'AKo')
- get_trey_card_rank(card) - Extract rank from treys card
- get_trey_card_suit(card) - Extract suit from treys card

- get_all_combinations(hand_str) - Get all combinations of a hand (12 for offsuit, 4 for suited, 6 for pair)

- card_to_str(card) - Convert Card object to string
- hand_to_str(hand) - Convert list of Card objects to hand string (e.g., 'AKo')
- cards_str(cards) - Convert list of Card objects to list of strings (e.g., ['As', 'Jd'])

- get_cards_from_str(hand_str) - Extract ranks from hand string
- is_suited(hand_str) - Check if hand string is suited
- is_offsuit(hand_str) - Check if hand string is offsuit

- pretty_suit(suit) - Convert suit to Unicode symbol
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Any, Tuple
if TYPE_CHECKING:
    from ..deck import Card as DeckCard

from treys import Card as TreysCard
from AppUtils.constants import RIVER

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']

CARD_RANK_TO_INT_VALUE = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

# All possible hand types from classify_hand_class
hand_types = ['FULL_HOUSE', 'FLUSH', 'STRAIGHT', 'SET', 'TRIPS', 'TWO_PAIRS', 'OVER_PAIR', 'TOP_PAIR', 'MID_PAIR', 'UNDER_PAIR', 'HIGH_CARD']

# All possible hand types from classify_hand_class_with_draw
flop_hand_types = hand_types + [
    # Draw types (when hand_type is HIGH_CARD)
    'BIG_COMBO_DRAW', 'FLUSH_DRAW', 'STRAIGHT_DRAW', 'DOUBLE_BACKDOOR_DRAW', 'BACKDOOR_FLUSH_DRAW', 'BACKDOOR_STRAIGHT_DRAW',
    # TOP_PAIR with draw
    'TOP_PAIR_W_BIG_COMBO_DRAW', 'TOP_PAIR_W_FLUSH_DRAW', 'TOP_PAIR_W_STRAIGHT_DRAW', 'TOP_PAIR_W_DOUBLE_BACKDOOR_DRAW', 'TOP_PAIR_W_BACKDOOR_DRAW',
    # PAIR with draw
    'PAIR_W_BIG_COMBO_DRAW', 'PAIR_W_FLUSH_DRAW', 'PAIR_W_STRAIGHT_DRAW', 'PAIR_W_DOUBLE_BACKDOOR_DRAW', 'PAIR_W_BACKDOOR_DRAW', 'PAIR_W_BACKDOOR_DRAW'
   ]

ALL_POCKET_PAIRS = ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22']
SUITED_CARDS = ['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s', 'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s', '98s', '97s', '96s', '95s', '94s', '93s', '92s', '87s', '86s', '85s', '84s', '83s', '82s', '76s', '75s', '74s', '73s', '72s', '65s', '64s', '63s', '62s', '54s', '53s', '52s', '43s', '42s', '32s']
UN_SUITED_CARDS = ['AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o', 'KQo', 'KJo', 'KTo', 'K9o', 'K8o', 'K7o', 'K6o', 'K5o', 'K4o', 'K3o', 'K2o', 'QJo', 'QTo', 'Q9o', 'Q8o', 'Q7o', 'Q6o', 'Q5o', 'Q4o', 'Q3o', 'Q2o', 'JTo', 'J9o', 'J8o', 'J7o', 'J6o', 'J5o', 'J4o', 'J3o', 'J2o', 'T9o', 'T8o', 'T7o', 'T6o', 'T5o', 'T4o', 'T3o', 'T2o', '98o', '97o', '96o', '95o', '94o', '93o', '92o', '87o', '86o', '85o', '84o', '83o', '82o', '76o', '75o', '74o', '73o', '72o', '65o', '64o', '63o', '62o', '54o', '53o', '52o', '43o', '42o', '32o']
ALL_CARD_COMBINATIONS = ALL_POCKET_PAIRS + SUITED_CARDS + UN_SUITED_CARDS

pocket_pairs = ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22']
a_high_suited = ['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A5s', 'A4s', 'A3s', 'A7s', 'A2s', 'A6s']
off_premiun = ['AKo', 'AQo', 'KQo', 'AJo', 'ATo', 'KJo']
high_suited = ['KQs', 'KJs', 'KTs', 'K9s', 'QTs', 'Q9s', 'J9s']
suited_connected = ['QJs', 'JTs', 'T9s', '98s', '87s', '65s', '54s']
weak_a_high = ['A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o']
weak_k_high = ['KTo', 'K9o', 'K8s', 'K7s']
suited_connected_with_gap = ['T8s', '97s', '86s', '75s', '64s', '43s', '32s']
ok_cards = ['QJo', 'QTo', 'T9o', 'K9o', 'K6s', 'K5s', 'K4s']
utg_open_range = pocket_pairs + a_high_suited + off_premiun + suited_connected + high_suited
co_open_range = utg_open_range + weak_a_high + weak_k_high + suited_connected_with_gap
btn_open_range = co_open_range + ok_cards

def is_already_trey(cards: List[Any]) -> bool:
    if not cards:
        return False
    return isinstance(cards[0], int)

def create_trey_list(cards: List[Any]) -> Tuple[TreysCard, ...]:
    if is_already_trey(cards):
        return tuple(sorted(cards))
    return tuple (sorted([TreysCard.new(card_to_str(card)) for card in cards]))

def create_board_trey(board_cards: List[Any], street: int = RIVER) -> Tuple[TreysCard, ...]:
    if is_already_trey(board_cards):
        return tuple(sorted(board_cards[:street+2]))
    return tuple(sorted([TreysCard.new(card_to_str(card)) for card in board_cards[:street+2]]))

def create_player_cards_trey(player_cards: List[Any]) -> Tuple[TreysCard, ...]:
    if is_already_trey(player_cards):
        return tuple(sorted(player_cards))
    return tuple(sorted([TreysCard.new(card_to_str(card)) for card in player_cards]))


def card_to_str(card: DeckCard) -> str:
    if isinstance(card, str):
        return card
    return card.rank + card.suit

# get list of cards as object return list of strings (result will look like ['As', 'Jd])
def cards_str(cards: List[DeckCard]) -> List[str]:
    return [card.rank + card.suit for card in cards]

def get_trey_card_rank(card: TreysCard) -> str:
    rank_int = TreysCard.get_rank_int(card)
    return RANKS[rank_int]
    
def get_trey_card_suit(card: TreysCard) -> str:
    suit_int = TreysCard.get_suit_int(card)
    suit_map = {1: 's', 2: 'h', 4: 'd', 8: 'c'}
    return suit_map[suit_int]

def trey_to_card_list(hand: Tuple[TreysCard, ...]) -> List[DeckCard]:
    from deck import Card as DeckCard  # Lazy import to avoid circular dependency
    card_list = []
    for card in hand:
        card_list.append(DeckCard(get_trey_card_rank(card), get_trey_card_suit(card)))
    return card_list

# get a player's hand when each card is a treys object return it as str (like 'AKo')
def trey_to_str(hand: Tuple[TreysCard, ...]) -> str:
    # Sort cards by rank strength (highest first)
    sorted_hand = sorted(hand, key=lambda card: RANKS.index(get_trey_card_rank(card)), reverse=True)
    
    rank1 = get_trey_card_rank(sorted_hand[0])
    rank2 = get_trey_card_rank(sorted_hand[1])
    suit1 = get_trey_card_suit(sorted_hand[0])
    suit2 = get_trey_card_suit(sorted_hand[1])
    
    if (rank1 == rank2):
        return rank1 + rank2
    else:
        return rank1 + rank2 + ('o' if suit1 != suit2 else 's')

# get a player's hand when each card is a Card object and return it as str (like 'AKo')
def hand_to_str(hand: List[DeckCard]) -> str:
    # Sort cards by rank strength (highest first)
    sorted_hand = sorted(hand, key=lambda card: RANKS.index(card.rank), reverse=True)
    
    if (sorted_hand[0].rank == sorted_hand[1].rank):
        return sorted_hand[0].rank + sorted_hand[1].rank
    else:
        return sorted_hand[0].rank + sorted_hand[1].rank + ('o' if sorted_hand[0].suit != sorted_hand[1].suit else 's')


def get_cards_from_str(hand_str: str) -> List[str]:
    return [hand_str[0], hand_str[1]]

def is_suited(hand_str: str) -> bool:
    return hand_str[-1] == 's'

def is_offsuit(hand_str: str) -> bool:
    return hand_str[-1] == 'o'

def pretty_suit(suit: str) -> str:
    cards_symbols = {'h' : '\u2665', 'd' : '\u2666', 'c' : '\u2663', 's' : '\u2660'}
    return cards_symbols[suit]

def get_all_combinations(hand_str: str) -> List[Tuple[TreysCard, ...]]:
    combinations = []
    cards = get_cards_from_str(hand_str)
    if is_suited(hand_str):
        for suit in SUITS:
            tmp_card1 = cards[0] + suit
            tmp_card2 = cards[1] + suit
            combinations.append(tuple(sorted([TreysCard.new(tmp_card1), TreysCard.new(tmp_card2)])))
    elif is_offsuit(hand_str):
        for suit1 in SUITS:
            for suit2 in SUITS:
                if suit1 != suit2:
                    tmp_card1 = cards[0] + suit1
                    tmp_card2 = cards[1] + suit2
                    
                    combinations.append(tuple(sorted([TreysCard.new(tmp_card1), TreysCard.new(tmp_card2)])))
    else:
        for i in range(len(SUITS)):
            for j in range(i+1, len(SUITS)):
                tmp_card1 = cards[0] + SUITS[i]
                tmp_card2 = cards[1] + SUITS[j]
                combinations.append(tuple(sorted([TreysCard.new(tmp_card1), TreysCard.new(tmp_card2)])))

    return combinations