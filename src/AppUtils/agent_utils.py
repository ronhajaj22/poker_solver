import pdb
import string
import AppUtils.StringUtils as strings

### the follwing functions are for trying to turn actions into a numeric value
def update_action_strength(prev_sum, hero_cureent_action, hero_action_size, num_of_players, pot_size, street):
    action_strength = prev_sum
    if hero_cureent_action == strings.RAISE:
        if street == 0:
            action_strength = round(action_strength + hero_action_size/10, 4) # TODO - do it better
        else:
            action_strength = hero_action_size/pot_size # TODO - do it better
    elif hero_cureent_action == strings.CALL:
        action_strength += 0.1
    elif hero_cureent_action == strings.CHECK:
        action_strength += (-0.1 if num_of_players == 2 else -0.05)
    
    return round(action_strength, 2)

# good for all streets
agent_global_features = ['stage', 'hero_cards', 'board_cards', 'hero_pos', 'pot_size', 'stack_size', 'spr', 'hand_strength', 'num_of_players_pre_flop', 'num_of_players_flop', 'preflop_action', 'is_hero_pre_flop_aggressor', 'flop_action']
flop_features = agent_global_features + ['flush_draw', 'straight_draw', 'board_dynamic']
turn_features = agent_global_features + ['draw', 'is_flush_hit', 'is_hero_last_aggressor', 'num_of_players_turn', 'turn_action']
river_features = agent_global_features + ['is_flush_hit', 'is_hero_last_aggressor', 'num_of_players_turn', 'num_of_players_river', 'turn_action', 'river_action']

FEATURES_BY_STREET = {
    1: flop_features,
    2: turn_features,
    3: river_features
}

FOLD = 0
CHECK = 1
CALL = 2
RAISE = 3

action_map = {
    FOLD: strings.FOLD, 
    CHECK: strings.CHECK, 
    CALL: strings.CALL, 
    RAISE: strings.RAISE
}



reversed_action_map = {v: k for k, v in action_map.items()}

