""" PARSE DATA TO AGENT ===
Feature extraction and tensor creation for neural network agents.
Converts poker game state and player information into feature tensors for neural network input.
Handles both training data parsing (from game stats) and live play feature extraction (from game objects).
Supports both regular agents (with hand information) and blind agents (without hand information).

FUNCTIONS (signatures):
- parse_training_data(street, is_main_player, player_hand_stats, board_stats, action, size) - Parse training sample
- parse_live_play_features(game, player, is_blind_agent=False) - Extract base features from game
- parse_live_play_hand_features(game, hand_cards, strength) - Extract hand-specific features
- parse_training_features(street, is_club_player, hand_stats, board_stats) - Parse features from stats
- parse_and_create_features_tensor(street, is_blind_agent, board_cards, player_cards, ...) - Main feature tensor creation
- create_board_tensor(board_cards) - Create board feature tensor
- create_last_card_tensor(board_cards) - Create last card feature tensor (turn/river)
- check_board_cards_errors(board_cards) - Validate board cards
- create_position_one_hot(position, is_hu_game) - Create position one-hot vector
- create_hand_class_one_hot(hand_class, value=1.0) - Create hand class one-hot vector
- create_features_tensor(street, is_blind_agent, ...) - Combine all features into final tensor
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..deck import Card
    from ..game import Game
    from ..player import Player
    from gg_hands_parser.game_stats import GameStats
    from player_stats.player_hand_stats import CurrentHandStats

from typing import List, Any
import torch, numpy as np, json
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, USED_POSITIONS_POST_FLOP, get_general_position
import AppUtils.StringUtils as strings
from AppUtils.cards_utils import cards_str
from AppUtils.constants import STREETS
import agent.agent_utils as agent_utils
import agent.features_heuristics as heuristics
from agent.training_storage import get_training_storage
from hands_classifier.hands_strength_classifier import calc_hand_strength
from hands_classifier.hand_board_calculations import calculate_flush_draw, calculate_straight_draw


MIN_PLAYERS = 2; MAX_PLAYERS = 4
MIN_RAISES_PREFLOP = 0; MAX_RAISES_PREFLOP = 4
MAX_POT_SIZE = 150
MAX_STACK_SIZE = 250
DEBUG_MODE = False

training_data_storage = get_training_storage()

def save_training_data_to_file(filepath: str = "data/training_data.pkl"):
    # Save all collected training data to a file. Should be called after parse_sessions() completes
    training_data_storage.save_to_file(filepath)

def load_training_data_from_file(filepath: str = "data/training_data.pkl"):
    # Load training data from a saved file. Returns True if successful, False otherwise
    return training_data_storage.load_from_file(filepath)

def parse_training_data(street: int, is_main_player: bool, player_hand_stats: CurrentHandStats, board_stats: GameStats, action: str, size: float):
    if is_main_player and (player_hand_stats.hand_range == None or len(player_hand_stats.hand_range) == 0):
        print("Error: player range is empty")
        return
    # create the features tensor
    features_values = parse_training_features(street, not is_main_player, player_hand_stats, board_stats)

    #create one-hot vector for the action
    normalized_action = agent_utils.create_actions_one_hot(action)
    
    # normalize the bet size to the pot size
    if action == strings.RAISE:
        size = size/board_stats.pot_size[street]

    # remove impossible actions and bet sizes
    masked_actions = agent_utils.mask_possible_actions(board_stats, player_hand_stats )
    
    # Create lambda function for bet size masking
    bet_size_mask_function = None
    if action == strings.RAISE:
        bet_size_mask_function = agent_utils.mask_bet_size(player_hand_stats, board_stats, board_stats.pot_size[street])
    
    # Add training sample to storage
    training_data_storage.add_training_sample(
        street=street,
        is_blind_agent=not is_main_player,
        feature_tensor=features_values,
        action=normalized_action,
        bet_size=size if action == strings.RAISE else None,
        masked_actions=masked_actions,
        bet_size_mask=bet_size_mask_function,
    )

def parse_live_play_features(game: Game, player: Player, is_blind_agent = False):
    board_cards = game.community_cards
    player_cards = player.hand.cards

    is_club_player_agent = is_blind_agent
    
    num_of_preflop_raises = game.get_num_raises_preflop()
    hero_range = player.range
    is_hero_pfr = player.is_pre_flop_agressor
    
    is_ip = player.is_ip(game.get_positions_by_order(), game.get_active_players_in_hand())
    hero_usable_position = get_general_position(player.position)
    
    num_players_pre_flop = game.num_of_players
    num_active_players = len(game.get_active_players_in_hand())
    
    pot_size = game.pot_size
    
    villains_stack_sizes = [villain.stack_size for villain in game.get_active_players_in_hand() if villain.position != player.position]
    hero_stack_size = player.stack_size
    effective_stack_size = min(hero_stack_size, max(villains_stack_sizes)) if len(villains_stack_sizes) > 0 else hero_stack_size

    hero_actions_value = player.action_strength[game.street]
    
    active_players = game.get_active_players_in_hand()
    if len(active_players) > 1:
        villains_actions_value = max([villain.action_strength[game.street] for villain in active_players if villain.position != player.position])
    else:
        villains_actions_value = 0
    
    is_hero_flop_agressor = player.is_flop_agressor
    is_hero_turn_agressor = player.is_turn_agressor

    hero_flop_actions_value = player.action_strength[FLOP]
    hero_turn_actions_value = player.action_strength[TURN]

    feature_tensor = parse_and_create_features_tensor(game.street, is_club_player_agent, board_cards, player_cards, num_of_preflop_raises, hero_range, is_hero_pfr, is_hero_flop_agressor, is_hero_turn_agressor, is_ip, hero_usable_position, num_players_pre_flop, num_active_players, pot_size, effective_stack_size, hero_actions_value, villains_actions_value, hero_flop_actions_value, hero_turn_actions_value)
    return feature_tensor

def parse_live_play_hand_features(game: Game, hand_cards: tuple[Card], strength: float):
    hand_strength = heuristics.enhance_hand_strength(strength)
    hand_class_idx = heuristics.classify_hand_class_heuristic(hand_cards, game.community_cards)
    hand_class_one_hot = create_hand_class_one_hot(hand_class_idx)
    if not isinstance(hand_class_one_hot, torch.Tensor):
        hand_class_one_hot = torch.tensor(hand_class_one_hot, dtype=torch.float32)

    if game.street < RIVER:
        flush_draw = calculate_flush_draw(hand_cards, game.community_cards)
        straight_draw = calculate_straight_draw(hand_cards, game.community_cards)
        hand_features = torch.tensor([hand_strength, flush_draw, straight_draw], dtype=torch.float32)
    else:
        hand_features = torch.tensor([hand_strength], dtype=torch.float32)
    
    return torch.cat([hand_features, hand_class_one_hot])
    return hand_features
    
def parse_training_features(street: int, is_club_player: bool, hand_stats: CurrentHandStats, board_stats: GameStats):
    board_cards = board_stats.board_cards[:street+2]
    player_cards = hand_stats.player_cards
    
    #print("board cards: ", [card for card in board_cards])
    #print("player cards: ", [card for card in player_cards])
    num_of_preflop_raises = board_stats.num_raises[PREFLOP]
    hero_range = hand_stats.hand_range
    is_hero_pfr = hand_stats.pfr

    is_ip = hand_stats.ip
    hero_usable_position = hand_stats.usable_position
    num_players_pre_flop = board_stats.num_players_pre_flop
    num_active_players = board_stats.num_players
    pot_size = board_stats.pot_size[street]
    hero_stack_size = hand_stats.stack_size
    villains_stack_sizes = [villain.hand_stats.stack_size for villain in board_stats.active_players if villain.hand_stats.position != hand_stats.position]
    effective_stack_size = min(hero_stack_size, max(villains_stack_sizes)) if len(villains_stack_sizes) > 0 else hero_stack_size
    hero_actions_value = hand_stats.action_strength[street]
    if len(board_stats.active_players) == 0:
        villains_actions_value = 0.4 # TODO - this is a temporary fix (villain bet and fold)
    else:
        villains_actions_value = max([villain.hand_stats.action_strength[street] for villain in board_stats.active_players if villain.hand_stats.position != hand_stats.position])
    
   # if not is_club_player:
   #     save_game_to_json_file(board_cards, player_cards, street, pot_size, hero_usable_position, effective_stack_size)
    
    is_hero_flop_agressor = hand_stats.is_flop_agressor
    is_hero_turn_agressor = hand_stats.is_turn_agressor

    hero_flop_actions_value = hand_stats.action_strength[FLOP]
    hero_turn_actions_value = hand_stats.action_strength[TURN]

    return parse_and_create_features_tensor(street, is_club_player, board_cards, player_cards, num_of_preflop_raises, hero_range, is_hero_pfr, is_hero_flop_agressor, is_hero_turn_agressor, is_ip, hero_usable_position, num_players_pre_flop, num_active_players, pot_size, effective_stack_size, hero_actions_value, villains_actions_value, hero_flop_actions_value, hero_turn_actions_value)

# main function - create the feature tensor (wrapper function)
def parse_and_create_features_tensor(street: int, is_blind_agent: bool, board_cards: tuple[Card], player_cards: tuple[Card], num_of_preflop_raises: int, hero_range, is_hero_pfr: bool, is_hero_flop_agressor: bool, is_hero_turn_agressor: bool, is_ip: bool, hero_usable_position: str, num_players_pre_flop: int, num_active_players: int, pot_size: float, effective_stack_size: float, hero_actions_value: float, villains_actions_value: float, hero_flop_actions_value: float, hero_turn_actions_value: float):
    is_hu_game = heuristics.normalize_boolean_value(num_players_pre_flop == 2)
    normalized_num_players = heuristics.normalize_value(num_active_players, MIN_PLAYERS, MAX_PLAYERS)
    pot_type = heuristics.normalize_value(num_of_preflop_raises, MIN_RAISES_PREFLOP, MAX_RAISES_PREFLOP)
       
    
    is_ip = heuristics.normalize_boolean_value(is_ip)
    position = create_position_one_hot(hero_usable_position, is_hu_game)
    is_pfr = heuristics.normalize_boolean_value(is_hero_pfr)
    normalized_pot_size = heuristics.normalize_log_value(pot_size, MAX_POT_SIZE)
    normalized_stack_size = heuristics.normalize_log_value(effective_stack_size, MAX_STACK_SIZE)
    spr = heuristics.normalize_value(pot_size, 0, effective_stack_size+pot_size)
    
    hand_strength = 0
    hand_class = 0
    flush_draw = 0
    straight_draw = 0
    if not is_blind_agent:
        hand_strength = calc_hand_strength(player_cards, board_cards, hero_range, street)
        hand_strength = heuristics.enhance_hand_strength(hand_strength)
        hand_class = create_hand_class_one_hot(heuristics.classify_hand_class_heuristic(player_cards, board_cards))
        flush_draw = calculate_flush_draw(player_cards, board_cards, street) if street < RIVER else 0
        straight_draw = calculate_straight_draw(player_cards, board_cards) if street < RIVER else 0
    
    is_flop_agressor = heuristics.normalize_boolean_value(is_hero_flop_agressor) if street > FLOP else 0
    is_turn_agressor = heuristics.normalize_boolean_value(is_hero_turn_agressor) if street == RIVER else 0
    
    board_tensor = create_board_tensor(board_cards) # board_connectivy, is_board_paired, board_flush, board_highest_card
    
    last_card_tensor = None
    if street > FLOP:
        last_card_tensor = create_last_card_tensor(board_cards) # is_over_card, is_complete_flush, is_connected_to_board
        
    feature_tensor = create_features_tensor(
        street, is_blind_agent, is_hu_game, 
        normalized_num_players, pot_type, is_ip, is_pfr, position,
        normalized_pot_size, normalized_stack_size, spr, 
        hand_strength, flush_draw, straight_draw,
        board_tensor, last_card_tensor,
        hero_actions_value, villains_actions_value, 
        hero_flop_actions_value, is_flop_agressor, hero_turn_actions_value, is_turn_agressor, hand_class
    )

    return feature_tensor

def create_board_tensor(board_cards: List[Card]) -> torch.Tensor:
    board_connectivy = heuristics.calc_board_connectivity_heuristic(board_cards)
    is_board_paired = heuristics.is_board_paired_heuristic(board_cards)
    board_flush = heuristics.calc_board_flush_heuristic(board_cards)
    board_highest_card = heuristics.calc_board_highest_card(board_cards)

    board_tensor = torch.tensor([board_connectivy, is_board_paired, board_flush, board_highest_card], dtype=torch.float32)
    return board_tensor

def create_last_card_tensor(board_cards): 
    over_card = heuristics.is_over_card(board_cards)
    complete_flush = heuristics.is_complete_flush(board_cards)
    connected_to_board = heuristics.is_connected_to_board(board_cards)
    return torch.tensor([over_card, complete_flush, connected_to_board], dtype=torch.float32)

def create_hand_class_one_hot(hand_class_idx: int) -> torch.tensor:
    one_hot_vector = np.zeros(4)
    if hand_class_idx == 10: # No Pair
        one_hot_vector[0] = 1.0
    elif hand_class_idx >= 8: # pair
        one_hot_vector[1] = 1.0
    elif hand_class_idx == 7: # top pair
        one_hot_vector[2] = 1.0
    else: # better
        one_hot_vector[3] = 1.0
    return one_hot_vector

def create_position_one_hot(position: str, is_hu_game: int) -> np.ndarray:
    # NOTE: this is a temoirary normalization of position - will be changed later
    if position == "SB" and is_hu_game:
        position = "BTN"
    
    index = 3 if position == "BTN" else 0 if position == "SB" else 1 if position == "BB" else 2; 
    one_hot_vector = np.zeros(4)
    one_hot_vector[index] = 1.0
    return one_hot_vector

def create_features_tensor(street: int, is_blind_agent: bool, is_hu_game: int, num_of_players: int, pot_type: float, is_ip: bool, is_pfr: bool, position: np.ndarray, pot_size: float, stack_size: float, spr: float, hand_strength: float, flush_draw: float, straight_draw: float, board_tensor: torch.Tensor, last_card_tensor: torch.Tensor, hero_actions_value: float, villains_actions_value: float, flop_actions_strength: float, is_flop_agressor: int, turn_actions_strength: float, is_turn_agressor: int, hand_class: torch.Tensor) -> torch.Tensor: 
    # FLOP: hand_strength_features (3) + hand_class (4) base_features (10)  + position_one_hot (4) + board_tensor (4) = 25      
    # TURN: hand_strength_features (3) + hand_class (4) + base_features (10) + added_features (2) + position_one_hot (4) + board_tensor (4) + flop_action(2) + last_card_tensor (3) = 30  
    # RIVER: hand_strength_features (1) + hand_class (4) + base_features (10) + added_features (4) + position_one_hot (4) + board_tensor (4) + flop_action(2) + turn_action(2) + last_card_tensor (3) = 30  
    # blind flop: base_features (10) + position_one_hot (4) + board_tensor (4) = 20
    # blind turn: base_features (10) + added_features (2) + position_one_hot (4) + board_tensor (4) + last_card_tensor (3) = 25
    # blind river: base_features (10) + added_features (4) + position_one_hot (4) + board_tensor (4) + last_card_tensor (3) = 27
  
    # NOTE: hand features mush ALWAYS be first! - break this order will break the game logic.
    
     # when the agent can't see the cards - doesn't have hand strength features
    hand_features = None
    if not is_blind_agent: 
        hand_strength_features = torch.tensor([hand_strength, flush_draw, straight_draw], dtype=torch.float32) if street < RIVER else torch.tensor([hand_strength], dtype=torch.float32)
        hand_features_names = ["hand_strength", "flush_draw", "straight_draw"] if street < RIVER else ["hand_strength"]

        hand_features_names += ['no_pair', 'pair', 'top_pair', 'over_pair+'];
        if not isinstance(hand_class, torch.Tensor):
            hand_class = torch.tensor(hand_class, dtype=torch.float32)

        hand_features = torch.cat([hand_strength_features, hand_class])

    base_features = torch.tensor([is_hu_game, num_of_players, pot_type, is_ip, is_pfr, pot_size, stack_size, spr, hero_actions_value, villains_actions_value], dtype=torch.float32)    
    base_features_names = ["is_hu_game", "num_of_players", "pot_type", "is_ip", "is_pfr", "pot_size", "stack_size", "spr", "hero_actions_value", "villains_actions_value"]
    
    if hand_features is not None:
        base_features = torch.cat([hand_features, base_features])
        base_features_names = hand_features_names + base_features_names
      
    # turn and river added features
    if street > FLOP:
        added_features = torch.tensor([flop_actions_strength, is_flop_agressor], dtype=torch.float32) if street == TURN else torch.tensor([flop_actions_strength, is_flop_agressor, turn_actions_strength, is_turn_agressor], dtype=torch.float32)
        added_features_names = ["flop_actions_strength", "is_flop_agressor"] if street == TURN else ["flop_actions_strength", "is_flop_agressor", "turn_actions_strength", "is_turn_agressor"]
    
    # Convert position to tensor if it's numpy array
    position_names = ["position_SB", "position_BB", "position_UTG", "position_BTN"]
    if not isinstance(position, torch.Tensor):
        position = torch.tensor(position, dtype=torch.float32)
    
    board_tensor_names = ["board_connectivity", "is_board_paired", "board_flush", "board_highest_card"]
    combined_features_tensor = torch.cat([base_features, position, board_tensor]) # base_features, position, board_tensor
    combined_features_names = base_features_names + position_names + board_tensor_names 
    
    
    if street > FLOP:
        last_card_tensor_names = ["over_card", "complete_flush", "connected_to_board"]
        combined_features_tensor = torch.cat([combined_features_tensor, added_features, last_card_tensor])
        combined_features_names += (added_features_names + last_card_tensor_names)
    #if not is_blind_agent:
    #    save_data_to_json(combined_features_tensor.tolist(), combined_features_names, "data/combined_features_tensor.json")
    return combined_features_tensor

def save_game_features_names_to_json_file(board_cards: List[Card], player_cards: List[Card], street: int, pot_size: float, hero_usable_position: str, effective_stack_size: float):
    dict_to_save = {
        "board_cards": cards_str(board_cards),
        "player_cards": cards_str(player_cards),
        "street": STREETS[street],
        "pot_size": pot_size,
        "hero_usable_position": hero_usable_position,
        "effective_stack_size": effective_stack_size
    }
    #save_data_to_json(list(dict_to_save.values()), list(dict_to_save.keys()), "data/combined_features_tensor.json")

def _format_json_compact_lists(obj, indent=0):
    """Format JSON with lists of simple values on single line"""
    if isinstance(obj, dict):
        if not obj:
            return '{}'
        items = []
        for key, value in obj.items():
            formatted_value = _format_json_compact_lists(value, indent + 4)
            items.append(' ' * (indent + 4) + json.dumps(key) + ': ' + formatted_value)
        return '{\n' + ',\n'.join(items) + '\n' + ' ' * indent + '}'
    elif isinstance(obj, list):
        if not obj:
            return '[]'
        # Check if list contains only simple types (not dict/list)
        has_complex_types = any(isinstance(item, (dict, list)) for item in obj)
        if has_complex_types:
            # Format lists of complex types with indentation
            items = []
            for item in obj:
                formatted_item = _format_json_compact_lists(item, indent + 4)
                items.append(' ' * (indent + 4) + formatted_item)
            return '[\n' + ',\n'.join(items) + '\n' + ' ' * indent + ']'
        else:
            # Format lists of simple values on single line
            items = [json.dumps(item) for item in obj]
            return '[' + ', '.join(items) + ']'
    else:
        return json.dumps(obj)

def save_data_to_json(data: Any, names: List[str], filename: str):
    new_entry = dict(zip(names, data))
    
    # Use JSONL format (JSON Lines) - each entry on a separate line
    # This allows appending without reading the entire file
    with open(filename, 'a', encoding='utf-8') as f:
        # Format the entry with compact lists
        formatted_entry = _format_json_compact_lists(new_entry, indent=0)
        f.write(formatted_entry + '\n')
