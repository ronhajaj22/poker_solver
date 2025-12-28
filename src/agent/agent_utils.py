""" AGENT UTILS ===

Utility module for agent-related constants and helper functions.

Provides constants for feature counts, action mappings, and utility functions for action masking,
bet size masking, and action encoding/decoding for neural network agents.

FUNCTIONS (signatures):
- sigmoid(x) - Sigmoid activation function
- tanh(x) - Hyperbolic tangent activation function
- mask_possible_actions_in_live_play(board, player) - Create action mask for live play
- mask_possible_actions(board_stats, player_hand_stats) - Create action mask from stats
- create_action_mask(possible_actions) - Convert action list to tensor mask
- mask_bet_size(player_hand_stats, board_stats, pot_size) - Create bet size mask function
- create_actions_one_hot(action) - Convert action string to one-hot vector

CONSTANTS:
- FLOP_FEATURES_COUNT, TURN_FEATURES_COUNT, RIVER_FEATURES_COUNT - Feature dimensions for each street
- BLIND_FLOP_FEATURES_COUNT, BLIND_TURN_FEATURES_COUNT, BLIND_RIVER_FEATURES_COUNT - Blind agent feature dimensions
- FOLD, CHECK, CALL, RAISE - Action integer constants
- STRING_ACTION_TO_INT_VALUE_MAP - Map action strings to integers
- INT_VALUE_TO_STRING_ACTION_MAP - Map integers to action strings
"""

import math, torch, numpy as np
import AppUtils.StringUtils as strings
from AppUtils.actions_utils import find_possible_actions_in_live_play, find_possible_actions_with_stack_sizes
from AppUtils.constants import FLOP, TURN, RIVER

# FEATURES COUNTS
FLOP_FEATURES_COUNT = 25 # 34
TURN_FEATURES_COUNT = 30 # 39
RIVER_FEATURES_COUNT = 30 # 39
BLIND_FLOP_FEATURES_COUNT = 18 # 20
BLIND_TURN_FEATURES_COUNT = 23 # 25
BLIND_RIVER_FEATURES_COUNT = 25 # 27

FEATURES_COUNT = {
    FLOP: FLOP_FEATURES_COUNT,
    TURN: TURN_FEATURES_COUNT,
    RIVER: RIVER_FEATURES_COUNT,
}

BLIND_FEATURES_COUNT = {
    FLOP: BLIND_FLOP_FEATURES_COUNT,
    TURN: BLIND_TURN_FEATURES_COUNT,
    RIVER: BLIND_RIVER_FEATURES_COUNT,
}
# ACTIONS
FOLD = 0
CHECK = 1
CALL = 2
RAISE = 3

STRING_ACTION_TO_INT_VALUE_MAP = {
    strings.FOLD: 0,
    strings.CHECK: 1,
    strings.CALL: 2,
    strings.RAISE: 3
}

INT_VALUE_TO_STRING_ACTION_MAP = {
    FOLD: strings.FOLD, 
    CHECK: strings.CHECK, 
    CALL: strings.CALL, 
    RAISE: strings.RAISE
}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    #Tanh value: (e^x - e^(-x)) / (e^x + e^(-x))
    return math.tanh(x)

def mask_possible_actions_in_live_play(board, player):
    possible_actions = find_possible_actions_in_live_play(board, player)
    return create_action_mask(possible_actions)

def mask_possible_actions(board_stats, player_hand_stats):
    last_bet_size = getattr(board_stats, 'last_bet_size', 0)
    stack_size = getattr(player_hand_stats, 'stack_size', 0)
    villains_stacks = [player.hand_stats.stack_size for player in board_stats.active_players if player.hand_stats.position != player_hand_stats.position]
    possible_actions = find_possible_actions_with_stack_sizes(last_bet_size, stack_size, villains_stacks)
    return create_action_mask(possible_actions)

def create_action_mask(possible_actions):
    action_mask = torch.zeros(4, dtype=torch.float32)
    for action_str in possible_actions:
        action_mask[STRING_ACTION_TO_INT_VALUE_MAP[action_str]] = 1.0
    return action_mask

def mask_bet_size(player_hand_stats, board_stats, pot_size):
    last_bet_size = getattr(board_stats, 'last_bet_size', 0)
    stack_size = getattr(player_hand_stats, 'stack_size', 0)

    min_size = 1
    if last_bet_size > 0: # if there was a bet, min size is 2x the last bet size
        min_size = min(last_bet_size*2, stack_size)
    min_size = min_size/pot_size

    # find max size - if two players left, max size is the smaller stack - else, current stack size
    if len(board_stats.active_players) == 2:
        max_size =  min(player.hand_stats.stack_size for player in board_stats.active_players)
    else:
        max_size = stack_size
    max_size = max_size/pot_size

    # Return both lambda function and bounds
    mask_func = lambda size: min_size <= size <= max_size
    mask_func.min_size = min_size
    mask_func.max_size = max_size
    return mask_func

def create_actions_one_hot(action):
    if action not in STRING_ACTION_TO_INT_VALUE_MAP:
        raise ValueError(f"Action '{action}' is not recognized.")
        
    one_hot_vector = np.zeros(len(STRING_ACTION_TO_INT_VALUE_MAP))
    one_hot_vector[STRING_ACTION_TO_INT_VALUE_MAP[action]] = 1.0
    
    return one_hot_vector