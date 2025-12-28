#Files
AGENTS_FILES_DIR = 'agents'
PRE_FLOP_CHARTS_DIR = 'data/pre_flop_charts'
PRE_FLOP_CHARTS_FILE_NAME = 'preflop_charts.json'

street_names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}

def find_agent_path (street: int, is_blind_agent: bool = False):
    return f'{AGENTS_FILES_DIR}/{"blind_" if is_blind_agent else ""}{street_names[street]}_agent.pth'

def find_hand_history_path(club_name, month=None):
    return f'data/hand_history/{club_name}_hands/{month if month else ""}'

def find_players_stats_path(bb_size):
    return f'data/players_stats/villains_stats_{bb_size}'

def get_club_hands_dir(club_name):
    return 'data/hand_history/' + club_name + '_hand'

def load_preflop_chart():
    """Load preflop charts from JSON file"""
    import os
    import json
    chart_path = os.path.join(os.path.dirname(__file__), '..', '..', PRE_FLOP_CHARTS_DIR, PRE_FLOP_CHARTS_FILE_NAME)
    with open(chart_path, 'r') as f:
        return json.load(f)
def load_hu_preflop_chart():
    import os
    import json
    chart_path = os.path.join(os.path.dirname(__file__), '..', '..', PRE_FLOP_CHARTS_DIR, 'hu_preflop_charts.json')
    with open(chart_path, 'r') as f:
        return json.load(f)

def find_hu_category(num_raises, stacks_size, position):
    return 'open' if num_raises == 0 else 'vs_open' if num_raises == 1 else 'vs_3bet' if num_raises == 2 else 'vs_4bet' if num_raises == 3 else 'vs_5bet'