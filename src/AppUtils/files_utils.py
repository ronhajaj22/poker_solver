#Files
AGENTS_FILES_DIR = 'agents'
PRE_FLOP_CHARTS_DIR = 'pre_flop_charts'
PRE_FLOP_CHARTS_FILE_NAME = 'preflop_charts.json'

street_names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}

def find_agent_path (street: int):
    return f'{AGENTS_FILES_DIR}/best_poker_agent_{street_names[street]}_v0.pth'

def find_parsed_hands_path(street: int):
    return f'parsed_hands/{street_names[street]}_parsed_hands.json'

def find_hand_history_path(club_name, month=None):
    return f'hand_history/{club_name}_hands/{month if month else ""}'

def find_players_stats_path(bb_size):
    return f'players_stats/villains_stats_{bb_size}'

def get_club_hands_dir(club_name):
    return 'hand_history/' + club_name + '_hand'

def load_preflop_chart():
    """Load preflop charts from JSON file"""
    import os
    import json
    chart_path = os.path.join(os.path.dirname(__file__), '..', '..', PRE_FLOP_CHARTS_DIR, PRE_FLOP_CHARTS_FILE_NAME)
    with open(chart_path, 'r') as f:
        return json.load(f)

def find_heads_up_json_name(num_raises, stacks_size, position):
    """Find the appropriate JSON chart name for heads-up situations"""
    stacks_size = 100  # TODO: check bug here
    is_sb = position == "SB"
    if num_raises == 0:
        situation = 'heads_up_' + ('sb' if is_sb else 'bb') + '_open_' + str(stacks_size)
    elif num_raises == 1:
        situation = 'heads_up_' + ('sb' if is_sb else 'bb') + '_vs_' + ('sb' if not is_sb else 'bb') + '_bet_' + str(stacks_size)
    else:
        situation = 'heads_up_' + ('sb' if is_sb else 'bb') + '_vs_' + ('sb' if not is_sb else 'bb') + '_' + str(num_raises+1) + 'bet_' + str(stacks_size)
    return situation