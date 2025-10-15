from .game_stats import GameStats
from ..create_hero.player import Player
import re
import pdb
from . import printer
from ..parser_utils import find_bb_size
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, ALL_POSITIONS_REVERSED
import AppUtils.StringUtils as strings

bb_size = 2
is_printer_active = True

def fix_won_spacing(text):
    # Add space before 'won' if not already preceded by space
    return re.sub(r'(?<!\s)won', r' won', text)

def collect_data(text, players_map):
    text = fix_won_spacing(text)
    total_hands_count = 0
    lines = text.strip().split('\n')
    global bb_size
    bb_size = find_bb_size(lines[0]);

    # all the lines that start new hand
    hand_start_indices = []
    for idx, line in enumerate(lines):
        if 'Poker Hand' in line:
            hand_start_indices.append(idx)

    for index_in_array, hand_starting_line in enumerate(hand_start_indices):
        hand_lines = lines[hand_starting_line: hand_start_indices[index_in_array+1] if index_in_array < len(hand_start_indices)-1 else len(lines)]
        parse_hand(hand_lines, players_map)
        
        total_hands_count += 1
    
    #print([player.player_name for player in players_map.players.values()])
    #print([player for player in players_map.players.keys()])
    #pdb.set_trace()
    return total_hands_count
    
def parse_hand(hand_lines, players_map):
    flop_line, turn_line, river_line, showdown_line, summary_line = find_street_lines(hand_lines)
    
    players_names = create_players_names_list(hand_lines[summary_line:])
    current_players = [players_map.get_player(player_name, bb_size, len(players_names) == 2) for player_name in players_names]
    board_cards = find_board_cards(hand_lines[summary_line:])
    # Create CurrentHandStats for each player
    hand_stats = {}
    for player in current_players:
        player.hand_stats.start_new_hand()
        player_cards = find_player_cards(hand_lines[:flop_line], hand_lines[summary_line:], player.player_name)
        if player_cards != []:
            player.hand_stats.set_player_cards(player_cards)
    
    printer.print_new_hand(current_players)
    board_stats = GameStats(bb_size, len(current_players), 0, board_cards)
    # initialize players stacks sizes and pot size
    initialize_stack_sizes(current_players, hand_lines[:flop_line], board_stats)
    
    # find the winners in the hand and update their stats (hand_won, wsd, wnsd [#TODO - WTSD])
    update_winners(current_players, hand_lines, river_line, showdown_line, summary_line)
    
    calc_preflop_actions(current_players, hand_stats, board_stats, hand_lines[0:flop_line])
    lines = []
    streets = []
    if (flop_line != -1):
        lines.append(hand_lines[flop_line:turn_line])
        streets.append(FLOP)
    if (turn_line != -1):
        lines.append(hand_lines[turn_line:river_line])
        streets.append(TURN)
    if (river_line != -1):
        lines.append(hand_lines[river_line:showdown_line])
        streets.append(RIVER)
    
    for line, street in zip(lines, streets):
        #printer.print_street(board_stats, current_players, street)
        
        calc_actions(current_players, board_stats, line, street)
        

def update_winners(current_players, hand_lines, river_line, showdown_line, summary_line):
    if river_line == -1: # didn't reach river - players folded
        is_uncalled_bet = True
    else:
        is_uncalled_bet = search_for_uncalled_bet(hand_lines[river_line:showdown_line])
    
    winners_names = find_winners_names(hand_lines[showdown_line:summary_line])
    for player in current_players:
        if player.player_name in winners_names:
            player.set_winner(is_uncalled_bet)

def search_for_uncalled_bet(river_lines):
    for line in river_lines:
        if line.strip().lower().startswith('uncalled bet'):
            return True
    return False

def find_winners_names(showdown_lines):
    names = []
    for line in showdown_lines:
        if 'collected' in line.lower():
            names.append(line.split()[0])
    return names



def find_street_lines(hand_lines):
    river_pattern = r'\*+\s*[^*]*\bRIVER\b[^*]*\*+'
    turn_pattern = r'\*+\s*[^*]*\bTURN\b[^*]*\*+'
    flop_pattern = r'\*+\s*[^*]*\bFLOP\b[^*]*\*+'
    showdown_pattern = r'\*+\s*[^*]*\bSHOWDOWN\b[^*]*\*+'
    summary_pattern = r'\*+\s*[^*]*\bSUMMARY\b[^*]*\*+' 
    
    flop_line, turn_line, river_line, summary_line = -1, -1, -1, -1
    for index, line in enumerate(hand_lines):
        if re.search(flop_pattern, line, re.IGNORECASE) is not None:
            flop_line = index
        elif re.search(turn_pattern, line, re.IGNORECASE) is not None:
            turn_line = index
        elif re.search(river_pattern, line, re.IGNORECASE) is not None:
            river_line = index
        elif re.search(showdown_pattern, line, re.IGNORECASE) is not None:
            showdown_line = index
        elif '*** SUMMARY ***' in line:
            summary_line = index
            break;
        index += 1
    return flop_line, turn_line, river_line, showdown_line, summary_line
def find_board_cards(summary_lines):
    for line in summary_lines:
        match = re.search(r"\[(.*?)\]", line)
        if match:
            cards_str = match.group(1)
            return cards_str.split()

def create_players_names_list(summary_lines):
    names = []
    
    for line in summary_lines:
        match = re.search(r"Seat \d+: ([^\s()]+)", line)
        if match:
            name = match.group(1).strip()
            names.append(name)
    
    return names

def initialize_stack_sizes(current_players, pre_flop_lines, board_stats):
    # TODO - one for-loop (not a mandatory fix)
    # initialize players stacks sizes
    for line in pre_flop_lines:
        match = re.search(r"Seat \d+: ([^\s()]+)\s+\(\$([\d,]+)(?:\.\d+)?", line)
        if not match:
            continue

        player_name = match.group(1).strip()
        stack = float(match.group(2).replace(",", ""))
        player = [player for player in current_players if player.player_name == player_name][0]
        player.set_stack_size(float(stack)/bb_size)
    
    # initialize big blinds and antes
    posts_lines = [line for line in pre_flop_lines if 'posts' in line]
    for line in posts_lines:
        m = re.match(r'(\S+): posts [^$]*\$(\d+\.?\d*)', line)
        if not m:
            continue

        player_name = m.group(1)
        amount = float(m.group(2))
        player = [player for player in current_players if player.player_name == player_name][0]
        player.hand_stats.add_chips_to_pot(board_stats, amount/bb_size, PREFLOP)

def find_player_cards(pre_flop_lines, summary_lines, player_name):
    player_cards = []
    if player_name == 'Hero':
        dealt_line = [line for line in pre_flop_lines if 'Dealt to Hero' in line]
        m = re.match(r'Dealt to Hero \[(\S+) (\S+)\]', dealt_line[0])   
        if m:
            player_cards = [m.group(1), m.group(2)]
    else:
        for line in summary_lines:
            if f'{player_name}' in line and 'showed' in line:
                m = re.search(r'\[(\w{2}) (\w{2})\]', line)
                if m:
                    player_cards = [m.group(1), m.group(2)]

    return player_cards

def calc_preflop_actions(current_players, hand_stats, board_stats, pre_flop_lines):
    position_index = board_stats.num_players_pre_flop - 1 # calculate the players 

    for line in pre_flop_lines:
        match = re.search(r"Seat \d+: ([^\s()]+)\s+\(\$([\d,]+)(?:\.\d+)?", line)
        if match:
            name = match.group(1).strip()
            chips_str = match.group(2).replace(",", "")
            if name in hand_stats:
                hand_stats[name].set_stack_size(int(chips_str))

        m = re.match(r'(\S+): (folds|checks|bets|calls|raises)(?: \$?(\d+\.?\d*))?', line)
        if m:
            action = m.group(2)
            amount = m.group(3)
            name = m.group(1)

            player = [player for player in current_players if player.player_name == name][0]
            if position_index >= 0:
                player.hand_stats.set_position(ALL_POSITIONS_REVERSED[position_index])
                position_index -= 1           

            player.add_action(PREFLOP, board_stats, convert_action_string(action), amount)
            
                
def calc_actions(current_players, board_stats, street_lines, street):
    board_stats.save_pot_to_next_street(street) # TODO - this is a temporary fix

    for line in street_lines:
        m = re.match(r'(\S+): (folds|checks|bets|calls|raises)(?: \$?(\d+\.?\d*))?', line)
        if m:
            player_name = m.group(1)
            player = [player for player in current_players if player.player_name == player_name][0]
            player.add_action(street, board_stats, convert_action_string(m.group(2)), m.group(3))


def convert_action_string(action):
    if action == 'folds':
        return strings.FOLD
    elif action == 'checks':
        return strings.CHECK
    elif action == 'calls':
        return strings.CALL
    elif action == 'bets' or action == 'raises':
        return strings.RAISE
    else:
        return None