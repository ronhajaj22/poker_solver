"""
=== FEATURES HEURISTICS ===
Heuristic feature extraction and normalization functions for neural network agents.

Converts raw poker game state (cards, board, actions) into normalized numerical features
suitable for neural network input. Includes board analysis, hand classification, action encoding,
and normalization utilities.

FUNCTIONS (signatures):
- classify_hand_class_heuristic(player_cards: tuple[Card], board_cards: tuple[Card]) - Classify hand and return index
- calc_board_flush_heuristic(board_cards) - Calculate board flush potential
- calc_board_connectivity_heuristic(board_cards) - Calculate board connectivity
- is_board_paired_heuristic(board_cards) - Check if board is paired
- calc_board_highest_card(board_cards) - Get highest card on board (normalized)
- is_over_card(board_cards) - Check if last card is overcard
- is_complete_flush(board_cards) - Check if last card completes flush
- is_connected_to_board(board_cards) - Check if last card connects to board
- update_street_actions_value(action, action_size, pot_size, last_bet_size=0) - Convert action to numeric value
- normalize_value(value, min_value, max_value) - Normalize value to [0, 1] range
- normalize_log_value(value, max_value) - Log normalization
- normalize_boolean_value(value) - Convert boolean to 0.0 or 1.0
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from game import Game
    from deck import Card

from typing import List
import numpy as np
import math
from hands_classifier.hand_board_calculations import classify_hand_class, find_flush_type
from hands_classifier.hands_river_category_classifier import HAND_CLASS_TO_IDX

from AppUtils.board_utils import calc_board_flush, calc_board_connectivity, is_board_paired
from AppUtils.cards_utils import CARD_RANK_TO_INT_VALUE
import AppUtils.StringUtils as strings



ACE_INTEGER_VALUE = 14

def classify_hand_class_heuristic(player_cards: tuple[Card], board_cards: tuple[Card]) -> int:
    hand_class = classify_hand_class(player_cards, board_cards)
    return HAND_CLASS_TO_IDX[hand_class]

### BOARD HEURISTICS ###

# how strong is the flush potential on the board
def calc_board_flush_heuristic(board_cards: tuple[Card]) -> float:
    board_flush = calc_board_flush(board_cards)
    return board_flush

# how connected is the board
def calc_board_connectivity_heuristic(board_cards: tuple[Card]) -> float:
    connectivity = calc_board_connectivity(board_cards)
    return connectivity

def is_board_paired_heuristic(board_cards: tuple[Card]) -> int:
    is_paired = is_board_paired(board_cards)
    return 1 if is_paired else 0

# how high is the highest card on the board
def calc_board_highest_card(board_cards: tuple[Card]) -> float:
    highest_card = max(sorted([CARD_RANK_TO_INT_VALUE[card.rank] for card in board_cards]))
    return normalize_value(highest_card, 2, ACE_INTEGER_VALUE)

### Last Card Heuristics - how the last card affects the board ###

def is_over_card(board_cards: tuple[Card]) -> float:
    last_card_rank = board_cards[-1].rank
    is_over_card = True
    for card in board_cards[:-1]:
        if CARD_RANK_TO_INT_VALUE[card.rank] >= CARD_RANK_TO_INT_VALUE[last_card_rank]:
            is_over_card = False
            break
    
    value = normalize_value(CARD_RANK_TO_INT_VALUE[card.rank], 4, 14) if is_over_card else 0
    return value

def is_complete_flush(board_cards: tuple[Card]) -> float:
    last_card_suit = board_cards[-1].suit
    is_complete_flush = 0
    num_card_in_suit = sum(1 for card in board_cards[:-1] if card.suit == last_card_suit)
    if num_card_in_suit == 2:
        is_complete_flush = 0.75
    elif num_card_in_suit >= 3:
        is_complete_flush = 1
    return is_complete_flush

def is_connected_to_board(board_cards: tuple[Card]) -> float:
    last_card_rank = board_cards[-1].rank
    unique_ranks = list(set([card.rank for card in board_cards[:-1]]))
    is_connected_to_board = 0
    for card in unique_ranks:
        if last_card_rank == 'A':
            dist = min(abs(CARD_RANK_TO_INT_VALUE[card] - CARD_RANK_TO_INT_VALUE[last_card_rank]), 
                abs(CARD_RANK_TO_INT_VALUE[card] - 1))
        else:
            dist = abs(CARD_RANK_TO_INT_VALUE[card] - CARD_RANK_TO_INT_VALUE[last_card_rank])
        if 0 < dist <= 3:
            is_connected_to_board += (0.1 * (4-dist))

    if is_connected_to_board == 0.1:
        is_connected_to_board = 0
    
    is_connected_to_board = normalize_value(is_connected_to_board, 0, 0.2*len(board_cards))
    return is_connected_to_board


### ACTIONS HEURISTICS - turn actions into a numeric value
def update_street_actions_value(action: str, action_size: float, pot_size: float, last_bet_size: float = 0) -> float:
    action_strength = 0
    if action == strings.CALL or action == strings.RAISE:
        action_strength = math.tanh(action_size/pot_size)

    ''' gemini idea - block call action to max value of 0.5
    pot_size_before_bet = pot_size - last_bet_size
    if action == strings.CALL:
        action_size = action_size/pot_size_before_bet
        action_strength = 0.5*math.tanh(1.5*action_size) # normalize call action to max value of 0.75

    elif action == strings.RAISE:
        action_strength = 0.5+0.5*math.tanh(action_size/pot_size_before_bet)
    '''
    return action_strength

### NORMALIZATION FUNCTIONS ###
def normalize_value(value: float, min_value: float, max_value: float) -> float:
    normalized_value = min(1, (max(0, value - min_value) / (max_value - min_value))) 
    return normalized_value

def normalize_log_value(value: float, max_value: float) -> float:
    normalized_value = (np.log(min(max_value, value)+1)/ np.log(max_value+1))
    return normalized_value

def normalize_boolean_value(value: bool) -> float:
    return 1.0 if value else 0.0

def enhance_hand_strength(hand_strength: float) -> float:
    normalized_value = (1-hand_strength)**2
    return normalized_value