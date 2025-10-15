import re
import json
import pdb
import os
from . import hands_calculator
from .player import Player
from ..create_villians.player_stats.PlayerStatsManager import PlayerStatsManager
from ..create_villians import parse_user_villians as hand_history_parser
from AppUtils.hand_board_calculations import calc_hand_strength, calc_board_dynamic, calc_draw_on_turn, calculate_straight_draw, calculate_flush_draw, is_flush_hit
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, ALL_POSITIONS
from AppUtils.cards_utils import cards_str
from ..parser_utils import find_bb_size
from AppUtils.files_utils import find_hand_history_path, find_parsed_hands_path
import AppUtils.StringUtils as strings

positions = ALL_POSITIONS
DEBUG_MODE = False
MAX_HANDS = 10000
players_map = PlayerStatsManager()

# this functionn go through all the GG files and parse them
# if collect_data is True, it will collect data from the hands and save it to the players_map
# if collect_data is False, it will create an agent that mimic the hero's actions
def parse_sessions(folder_path = 'matrix', collect_data = False):
    all_flop_data = []
    all_turn_data = []
    all_river_data = []
    
    folder_path = find_hand_history_path('matrix')
    total_hands_count = 0
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                text = file.read()
                if (collect_data):
                    total_hands_count += hand_history_parser.collect_data(text, players_map)
                    print("file read successfully, Total hands count: ", total_hands_count)
                    if DEBUG_MODE and total_hands_count > MAX_HANDS:
                        break
                else:
                    ans = parse_session_hands(text)
                if (not collect_data and ans):
                    flop_data, turn_data, river_data = ans
                    if flop_data:
                        all_flop_data.extend(flop_data)
                    if turn_data:
                        all_turn_data.extend(turn_data)
                    if river_data:
                        all_river_data.extend(river_data)

                    save_hands_to_json(all_flop_data, find_parsed_hands_path(FLOP))
                    save_hands_to_json(all_turn_data, find_parsed_hands_path(TURN))
                    save_hands_to_json(all_river_data, find_parsed_hands_path(RIVER))
    
    if (collect_data):
        players_map.get_all_summaries()
        return players_map

# this function go through all the hands in the session and parse them one by one
def parse_session_hands(text):
    lines = text.strip().split('\n')
    bb_size = find_bb_size(lines[0]);
    
    all_flop_data = []
    all_turn_data = []
    all_river_data = []
    # all the lines that start new hand
    hand_start_indices = []
    for idx, line in enumerate(lines):
        if 'Poker Hand' in line:
            hand_start_indices.append(idx)

    for index_in_array, hand_starting_line in enumerate(hand_start_indices):
        hand_lines = lines[hand_starting_line: hand_start_indices[index_in_array+1] if index_in_array < len(hand_start_indices)-1 else len(lines)]
        
        result = parse_hand(hand_lines, bb_size) #this is the main function
        if result is None:
            all_flop_data.append(None)
            all_turn_data.append(None)
            all_river_data.append(None)
            continue
            
        flop_data, turn_data, river_data = result
        if flop_data:
            all_flop_data.extend(flop_data);
        else:
            all_flop_data.append(None)
        if turn_data:
            all_turn_data.extend(turn_data);
        else:
            all_turn_data.append(None)
        if river_data:
            all_river_data.extend(river_data);
        else:
            all_river_data.append(None)
    
    return all_flop_data, all_turn_data, all_river_data


#this is the final function - take the parsed hands and save them to a json file
def save_hands_to_json(hands_data, filename="parsed_hands.json"):
    """Save hands data to a formatted JSON file that the model can use"""
    # Filter out None values
    valid_hands = [hand for hand in hands_data if hand is not None]

    # Ensure the directory exists before saving
    import os
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(valid_hands, f, indent=2, ensure_ascii=False)

# parse the hand into flop, turn and river data
def parse_hand(hand_lines, bb_size):
    hero_found_in_flop, hero_found_in_turn, hero_found_in_river, flop_line, turn_line, river_line = find_hero_in_action(hand_lines)  
    if not hero_found_in_flop:
        return None

    hand_id = re.match(r'Poker Hand #(\S+):', hand_lines[0])
    turn_card = None
    
    data = {}
    data['hand_id'] = hand_id.group(1) if hand_id else 0
    data['hero_pos'] = None

    players, pot_size = initialize_pre_action_players(hand_lines[:flop_line], bb_size)

    posts_lines = [line for line in hand_lines[:flop_line] if 'posts' in line]
    for line in posts_lines:
        if 'ante' in line: # Special case - BOMB POT - ignore this hand
            return None;
        m = re.match(r'(\S+): posts [^$]*\$(\d+\.?\d*)', line)
        if m:
            player = m.group(1)
            amount = float(m.group(2))
            
            players[player]['chips_in_pot'] = amount/bb_size
            players[player]['stack'] -= (amount/bb_size)
            pot_size += (amount/bb_size)


    # Finding hero's cards
    flop_cards = find_board_cards(hand_lines[flop_line], 1)
    hero_cards = find_hero_cards(hand_lines, flop_line)
    
    my_hand_data = hands_calculator.HandData(data['hand_id'], bb_size, hero_cards, flop_cards, len(players))
    flop_cards = my_hand_data.flop_cards
    
    # Preflop action
    ans = parse_pre_flop_action(players, hand_lines[:flop_line], pot_size, bb_size)
    if ans == None:
        return (None, None, None)
    players, pot_size, preflop_raises, last_raiser = ans

    players_list = []
    for p in players:
        players_list.append(Player( players[p]['name'], players[p]['stack'], players[p]['pos'], hero_cards if p == 'Hero' else [], players[p]['is_folded'], players[p]['chips_in_pot'], 'FOLD' if players[p]['is_folded'] else 'CHECK'))
        
    hero = [player for player in players_list if player.name == 'Hero'][0]
   
    if last_raiser == 'Hero':
        hero.set_is_last_raiser()
    my_hand_data.hero = hero

    turn_data = None
    river_data = None

    # Flop action
    players = parse_actions(players_list, hand_lines[flop_line:turn_line], bb_size, FLOP);
    my_hand_data.calc_flop_action(players, pot_size, len(hero.actions[FLOP]))
    if (my_hand_data.sum_action[FLOP] == []):
        return (None, None, None)
    
    flop_data = create_flop_json(players, my_hand_data, hero, preflop_raises);
    if hero_found_in_turn:
        players = parse_actions(players_list, hand_lines[turn_line:river_line], bb_size, TURN);
        
        if (len(hero.actions[TURN]) == 0):
            pdb.set_trace()
        turn_card = find_board_cards(hand_lines[turn_line], 2)[0]
        my_hand_data.set_turn_card(turn_card)
        my_hand_data.calc_turn_action(players, len(hero.actions[TURN]))
        turn_data = create_turn_data(players, my_hand_data, preflop_raises)

    if hero_found_in_river:
        players = parse_actions(players_list, hand_lines[river_line:], bb_size, RIVER);
        if (len(hero.actions[RIVER]) == 0):
            pdb.set_trace()
        river_card = find_board_cards(hand_lines[river_line], 2)[0]
        my_hand_data.set_river_card(river_card)
        my_hand_data.calc_river_action(players, len(hero.actions[RIVER]))
        river_data = create_river_data(players, my_hand_data, preflop_raises)
    
    return flop_data, turn_data, river_data 

# check if hero is in flop (+ turn + river) action - if not, we skip the hand
def find_hero_in_action(hand_lines):
    flop_line, turn_line, river_line = 0, 0, 0;
    
    for line_index, line in enumerate(hand_lines):
        if '*** FLOP ***' in line:
            flop_line = line_index
            break

    if flop_line == 0:
        return (0, 0, 0, False, False, False)

    hero_found_in_flop, hero_found_in_turn, hero_found_in_river = False, False, False
    for line_index, line in enumerate(hand_lines[flop_line:]):
        if 'Hero' in line and re.match(r'(\S+): (folds|calls|checks|bets|raises)(?: \$?(\d+\.?\d*))?(?: to \$?(\d+\.?\d*))?', line) != None:
            if hero_found_in_flop == False:
                hero_found_in_flop = True
            else:
                if turn_line > 0:
                    hero_found_in_turn = True
                if river_line > 0:
                    hero_found_in_river = True
                    break
        if '*** TURN ***' in line:
            turn_line = flop_line + line_index
            if (not hero_found_in_flop):
                break;
        if '*** RIVER ***' in line:
            river_line = flop_line + line_index
            if (not hero_found_in_turn):
                break;
        if '*** SUMMARY ***' in line:
            if (turn_line == 0):
                turn_line = flop_line + line_index
            elif (river_line == 0):
                river_line = flop_line + line_index
            break
    
    if turn_line == 0:
        print("ERROR!!! something went wrong in finding the turn line")
        return (0,0,0, False, False, False)
    
    return (hero_found_in_flop, hero_found_in_turn, hero_found_in_river, flop_line, turn_line, river_line)

def find_hero_cards(hand_lines, flop_line):
    hero_cards = []
    dealt_line = [line for line in hand_lines[:flop_line] if 'Dealt to Hero' in line]
    m = re.match(r'Dealt to Hero \[(\S+) (\S+)\]', dealt_line[0])   
    if m:
        hero_cards = [m.group(1), m.group(2)]
    return hero_cards

def find_board_cards(line, stage):
    if stage == FLOP:
        m = re.search(r'\[(.*)\]', line)
    else:
        m = re.search(r'\[([^\]]+)\] \[([^\]]+)\]', line)
    if m:
        if stage == RIVER:
            print(m.group(stage))
        return m.group(stage).split()
        
    else:
        print("ERROR!!! something went wrong in finding the board cards")
        return []

def initialize_pre_action_players(pre_flop_line, bb_size):
    pot_size = 0
    players = {}
   
    seats_lines = [line for line in pre_flop_line if 'Seat' in line and 'chips' in line]
    for line in seats_lines:
        #m = re.match(r'Seat (\d+): (\S+) \(\$(\d+\.?\d*) in chips\)', line)
        m = re.match(r'Seat (\d+): (\S+) \(\$([\d,]+\.?\d*) in chips\)', line)

        if m != None:
            seat_num, player, stack = m.groups()
            stack = float(stack.replace(',', ''))
            
            players[player] = {
                'stack': round(float(stack)/bb_size, 2),
                'chips_in_pot': 0,
                'is_folded': False,
            }
        
    return players, pot_size

def parse_pre_flop_action(players, pre_flop_lines, pot_size, bb_size):
    back_position = len(players) # calculate the players 
    last_raiser = None
    preflop_raises = 0

    for line in pre_flop_lines:
        m = re.match(r'(\S+): (folds|calls|checks|bets|raises)(?: \$?(\d+\.?\d*))?(?: to \$?(\d+\.?\d*))?', line)
        if m:
            player, action, difference_from_last_bet, bet_total_size = m.groups()
            bet_total_size = float(bet_total_size)/bb_size if bet_total_size else 0
            difference_from_last_bet = float(difference_from_last_bet)/bb_size if difference_from_last_bet else 0

            # calculate the players positions
            if back_position > 0:
                players[player]['pos'] = positions[len(positions)-back_position]
                players[player]['name'] = player
                back_position -= 1
            if action == 'raises':
                preflop_raises += 1
                prev_chips_in_pot = 0 if players[player]['chips_in_pot'] == None else players[player]['chips_in_pot']
                players[player]['chips_in_pot'] = round(players[player]['chips_in_pot'] + bet_total_size, 2)
                players[player]['stack'] = round(players[player]['stack'] - (bet_total_size - prev_chips_in_pot), 2)
                pot_size = round(pot_size + (bet_total_size - prev_chips_in_pot), 2)
                last_raiser = player
            elif action == 'calls':
                players[player]['chips_in_pot'] = round(players[player]['chips_in_pot'] + difference_from_last_bet, 2)
                pot_size = round(pot_size + difference_from_last_bet, 2)
                players[player]['stack'] = round(players[player]['stack'] - difference_from_last_bet, 2)
            elif action == 'folds':
                players[player]['is_folded'] = True

    ### hero stats
    if 'Hero' not in players:
        print("Error! Hero not found in players")
        return None
    if 'pos' not in players['Hero']:
        print("Error! Hero position not found")
        return None
    
    villians = {player: players[player] for player in players if players[player]['is_folded'] == False}

    return villians, pot_size, preflop_raises, last_raiser

def parse_actions(players_list, street_lines, bb_size, street):
    hero_num_of_actions = get_hero_num_of_actions(street_lines)
    action_counter = 0
    for line in street_lines:
        m = re.match(r'(\S+): (folds|checks|bets|calls|raises)(?: \$?(\d+\.?\d*))?', line)
        if m:
            player_name = m.group(1)
            #    pdb.set_trace()
            if player_name in [player.name for player in players_list]:
                villian = [player for player in players_list if player.name == player_name][0]
                villian.add_action(street, convert_action(m.group(2)), m.group(3), bb_size)

                if player_name == 'Hero': 
                    action_counter += 1
                    if action_counter == hero_num_of_actions: # TODO - Check it, maybe remove it
                        break
            else:
                print(player_name, "is not found")
    
    
    return players_list

def get_hero_num_of_actions(street_lines):
    counter = 0
    for line in street_lines:
        m = re.match(r'(\S+): (folds|checks|bets|calls|raises)(?: \$?(\d+\.?\d*))?', line)
        if m and 'Hero' in line:
            counter += 1
        
    return counter

def convert_action(action):
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

def create_flop_json(players, hand_data, hero, preflop_raises,):
    index = 0
    flop_data = []
    players_in_flop = [player for player in players if strings.FOLD not in player.actions[PREFLOP]]
    for hero_action, hero_size in zip(hero.actions[FLOP], hero.action_sizes[FLOP]):
        flop_action_data = {
            "hand_id": hand_data.hand_id,
            
            "street": FLOP,
            "stage": index,

            "hero_cards": cards_str(hand_data.hero_cards),
            "hero_pos": hero.position,
            "stack_size": hero.stack[FLOP][index],

            "board_cards": cards_str(hand_data.flop_cards),
            "pot_size": hand_data.pot_size[FLOP][index],
            "spr": hand_data.calc_spr(players_in_flop, FLOP, index),
            
            "hand_strength": calc_hand_strength(hand_data.hero_cards, hand_data.flop_cards),
            "flush_draw": calculate_flush_draw(hand_data.hero_cards, hand_data.flop_cards),
            "straight_draw": calculate_straight_draw(hand_data.hero_cards, hand_data.flop_cards),
            "board_dynamic": calc_board_dynamic(hand_data.flop_cards),
            
            "num_of_players_pre_flop": len(players),
            "num_of_players_flop": len(players_in_flop),
           
            "is_hero_pre_flop_aggressor": hero.actions[PREFLOP][-1] == strings.RAISE,

            "preflop_action": preflop_raises,
            "flop_action": hand_data.sum_action[FLOP][index],

            # hero's decision
            "action": hero_action,
            "action_size": hero_size,
        }
        index +=1
        flop_data.append(flop_action_data)
    return flop_data

def create_turn_data(players, my_hand_data, preflop_raises):
    hero = my_hand_data.hero
    index = 0
    turn_data = []

    players_in_flop = [player for player in players if strings.FOLD not in player.actions[PREFLOP]]
    players_in_turn = [player for player in players_in_flop if strings.FOLD not in player.actions[FLOP]]

    for hero_action, hero_size in zip(hero.actions[TURN], hero.action_sizes[TURN]):
        turn_action_data = {
            "hand_id": my_hand_data.hand_id,
            
            "street": TURN,
            "stage": index,

            "hero_cards": cards_str(my_hand_data.hero_cards),
            "hero_pos": hero.position,
            "stack_size": hero.stack[TURN][index],

            "board_cards": cards_str(my_hand_data.flop_cards + [my_hand_data.turn_card]),
            "pot_size": my_hand_data.pot_size[TURN][index],
            "spr": my_hand_data.calc_spr(players_in_turn, TURN, index),

            "hand_strength": calc_hand_strength(my_hand_data.hero_cards, my_hand_data.board_cards),
            "draw": calc_draw_on_turn(my_hand_data.hero_cards, my_hand_data.board_cards),

            "board_dynamic": calc_board_dynamic(my_hand_data.board_cards),
            "is_flush_hit": is_flush_hit(TURN, my_hand_data.flop_cards, my_hand_data.turn_card),

            "is_hero_pre_flop_aggressor": hero.actions[PREFLOP][-1] == strings.RAISE,
            "is_hero_last_aggressor": hero.actions[FLOP][-1] == 'RAISE',

            "num_of_players_pre_flop": len(players),
            "num_of_players_flop": len(players_in_flop),
            "num_of_players_turn": len(players_in_turn),

            "preflop_action": preflop_raises,
            "flop_action": my_hand_data.sum_action[FLOP][-1],
            "turn_action": my_hand_data.sum_action[TURN][index],

            "action": hero_action,
            "action_size": hero_size,
        }
        index +=1
        turn_data.append(turn_action_data)
    return turn_data


def create_river_data(players, my_hand_data, preflop_raises): 
    hero = my_hand_data.hero
    index = 0
    river_data = []

    players_in_flop = [player for player in players if strings.FOLD not in player.actions[PREFLOP]]
    players_in_turn = [player for player in players_in_flop if strings.FOLD not in player.actions[FLOP]]
    players_in_river = [player for player in players_in_turn if strings.FOLD not in player.actions[TURN]]

    for hero_action, hero_size in zip(hero.actions[RIVER], hero.action_sizes[RIVER]):
        river_action_data = {
            "hand_id": my_hand_data.hand_id,
            
            "street": RIVER,
            "stage": index,

            "hero_cards": cards_str(my_hand_data.hero_cards),
            "hero_pos": hero.position,
            "stack_size": hero.stack[RIVER][index],

            "board_cards": cards_str(my_hand_data.flop_cards + [my_hand_data.turn_card] + [my_hand_data.river_card]),
            "pot_size": my_hand_data.pot_size[RIVER][index],
            "spr": my_hand_data.calc_spr(players_in_river, RIVER, index),
            
            "hand_strength": calc_hand_strength(my_hand_data.hero_cards, my_hand_data.board_cards),
            "is_flush_hit": is_flush_hit(RIVER, my_hand_data.flop_cards, my_hand_data.turn_card, my_hand_data.river_card),
            
            "is_hero_pre_flop_aggressor": hero.actions[PREFLOP][-1] == strings.RAISE,
            "is_hero_last_aggressor": hero.actions[TURN][-1] == strings.RAISE,

            "num_of_players_pre_flop": len(players),
            "num_of_players_flop": len(players_in_flop),
            "num_of_players_turn": len(players_in_turn),
            "num_of_players_river": len(players_in_river),

            "preflop_action": preflop_raises,
            "flop_action": my_hand_data.sum_action[FLOP][-1],
            "turn_action": my_hand_data.sum_action[TURN][-1],
            "river_action": my_hand_data.sum_action[RIVER][index],
            
            "action": hero_action,
            "action_size": hero_size,
        }
        index +=1
        river_data.append(river_action_data)
    return river_data