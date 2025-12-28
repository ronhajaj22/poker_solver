from AppUtils.constants import FLOP, STREETS

FLOP_ACTIONS = 'flop_stats'
ALL_STATS = 'general'

NUM_OF_HANDS = 'hands_in_flop'
TWO_PLAYERS_POT = '2_players'
MULTIWAY_POT = 'multiway'

# ACTIONS KEYS
VS_CHECK = 'vs_check'
VS_BET = 'vs_bet'
VS_BET_SIZES = 'vs_different_bet_sizes'
BET_SIZES = 'open_sizes'
VS_RAISE = 'vs_raise'
IP = 'ip'
OOP = 'oop'

SRP = 'srp'
THREE_BET = '3BP' # TODO - fix in file
FOUR_BET = '4BP'

PRE_FLOP_RAISER = 'pfr'
PRE_FLOP_CALLER = 'pfc'

FLOP_AGGRESSOR = 'flop_aggressor'
FLOP_PASSIVE = 'flop_passive'

TURN_AGGRESSOR = 'turn_aggressor'
TURN_PASSIVE = 'turn_passive'

# BET SIZING CATEGORIES
SMALL = 'small'
MEDIUM = 'medium'
LARGE = 'large'
OVERBET = 'overbet'

HERO = 'Hero'
HERO_HU = 'Hero_hu'
GENERAL_PLAYER = 'player'
GENERAL_PLAYER_HU = 'player_hu'

def get_key(is_pfr, is_srp, is_ip):
    pfr_key = PRE_FLOP_RAISER if is_pfr else PRE_FLOP_CALLER
    pot_type_key = SRP if is_srp else THREE_BET
    ip_key = IP if is_ip else OOP
    print("pot_type_key", pot_type_key + '_' + pfr_key + '_' + ip_key)
    return pot_type_key + '_' + pfr_key + '_' + ip_key

def get_action_key(action):
    action_key = {0: VS_CHECK, 1: VS_BET, 2: VS_RAISE}
    return action_key.get(action, VS_RAISE)

def get_street_key(street):
    return STREETS[street].lower()+'_stats'

def get_num_of_players_key(num_of_players):
    num_of_players_key = {2: TWO_PLAYERS_POT, 3: MULTIWAY_POT}
    return num_of_players_key.get(num_of_players, MULTIWAY_POT)