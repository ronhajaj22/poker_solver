import json
import os
from re import I
from turtle import pos
from AppUtils.constants import USED_POSITIONS
from AppUtils.cards_utils import ALL_CARD_COMBINATIONS
from AppUtils.files_utils import PRE_FLOP_CHARTS_DIR
import AppUtils.StringUtils as strings

utg_open_range = {
    'AA', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
    'KK', 'KQs', 'KJs', 'KTs', 'K9s',
    'QQ', 'QJs', 'QTs', 'Q9s',
    'JJ', 'JTs', 'J9s',
    'TT', 'T9s',
    '99', 
    '88',
    '77',
    'AKo', 'AQo', 'AJo', 'ATo', 'KQo', 'KJo'
}
hj_open_range = {
    'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44',
    'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
    'KQs', 'KJs', 'KTs', 'K9s', 'K8s',
    'QJs', 'QTs', 'Q9s', 'Q8s',
    'JTs', 'J9s', 'J8s',
    'T9s', 'T8s',
    '98s', '97s',
    '87s', '86s',
    '76s', 
    'AKo', 'AQo', 'AJo', 'ATo', 'KQo', 'KJo'
}
co_open_range = {
  'AA', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
  'KK', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s',
  'QQ', 'QJs', 'QTs', 'Q9s', 'Q8s',
  'JJ', 'JTs', 'J9s', 'J8s',
  'TT', 'T9s', 'T8s',
  '99', '98s', '97s',
  '88', '87s', '86s',
  '77', '76s', '75s',
  '66', '65s',
  '55', '54s',
  '44', 
  '33',
  '22',
  'AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'KQo', 'KJo', 'KTo', 'QJo', 'QTo', 'JTo'
}
btn_open_range = {
    # Pairs
    'AA','KK','QQ','JJ','TT','99','88','77','66','55','44','33','22',
    # Suited
    'AKs','AQs','AJs','ATs','A9s','A8s','A7s','A6s','A5s','A4s','A3s','A2s',
    'KQs','KJs','KTs','K9s','K8s','K7s','K6s','K5s','K4s','K3s','K2s',
    'QJs','QTs','Q9s','Q8s','Q7s',
    'JTs','J9s','J8s','J7s',
    'T9s','T8s','T7s',
    '98s','97s',
    '87s','86s',
    '76s','75s',
    '65s','64s',
    '54s','53s',
    '43s','42s',
    '32s',
    # Offsuit
    'AKo','AQo','AJo','ATo','A9o', 'A8o', 'A7o', 'A6o', 'A5o',
    'KQo','KJo','KTo', 'K9o', 'K8o',
    'QJo','QTo', 'Q9o', 'Q8o',
    'JTo', 'J9o', 'J8o',
    'T9o',
    '98o',
    '87o',
    '76o'
}
sb_open_range = {
  'AA', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
  'KK', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s',
  'QQ', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s',
  'JJ', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s',
  'TT', 'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s',
  '99', '98s', '97s', '96s', '95s', '94s', '93s', '92s',
  '88', '87s', '86s', '85s', '84s', '83s', '82s',
  '77', '76s', '75s', '74s', '73s', '72s',
  '66', '65s', '64s', '63s', '62s',
  '55', '54s', '53s', '52s',
  '44', '43s', '42s',
  '33', '32s',
  '22',
  'AKo','AQo','AJo','ATo','A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o',
  'KQo','KJo','KTo', 'K9o', 'K8o', 'K7o', 'K6o', 'K5o',
  'QJo','QTo', 'Q9o', 'Q8o', 'Q7o',
  'JTo', 'J9o', 'J8o', 'J7o',
  'T9o', 'T8o', 
  '98o', 
  '87o',
  '76o'
}
bb_open_range = {
    # Pairs
    'AA','KK','QQ','JJ','TT','99','88','77','66','55','44','33','22',
    # Suited
    'AKs','AQs','AJs','ATs','A9s','A8s','A7s','A6s','A5s','A4s','A3s','A2s',
    'KQs','KJs','KTs','K9s','K8s','K7s','K6s','K5s','K4s','K3s','K2s',
    'QJs','QTs','Q9s','Q8s','Q7s',
    'JTs','J9s','J8s','J7s',
    'T9s','T8s','T7s',
    '98s','97s',
    '87s','86s',
    '76s','75s',
    '65s','64s',
    '54s','53s',
    '43s','42s',
    '32s',
    # Offsuit
    'AKo','AQo','AJo','ATo','A9o', 'A8o', 'A7o', 'A6o', 'A5o',
    'KQo','KJo','KTo', 'K9o', 'K8o',
    'QJo','QTo', 'Q9o', 'Q8o',
    'JTo', 'J9o', 'J8o',
    'T9o',
    '98o',
    '87o',
    '76o'
}
bb_in_2_bet_pots ={
  'AA', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
  'KK', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s',
  'QQ', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s',
  'JJ', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s',
  'TT', 'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s',
  '99', '98s', '97s', '96s', '95s', '94s', '93s', '92s',
  '88', '87s', '86s', '85s', '84s', '83s', '82s',
  '77', '76s', '75s', '74s', '73s', '72s',
  '66', '65s', '64s', '63s', '62s',
  '55', '54s', '53s', '52s',
  '44', '43s', '42s',
  '33', '32s',
  '22',
  'AKo','AQo','AJo','ATo','A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o',
  'KQo','KJo','KTo', 'K9o', 'K8o', 'K7o', 'K6o', 'K5o',
  'QJo','QTo', 'Q9o', 'Q8o', 'Q7o',
  'JTo', 'J9o', 'J8o', 'J7o',
  'T9o', 'T8o', 
  '98o', '97o',
  '87o', '86o',
  '76o', '75o',
  '65o', '64o',
  '54o'
}


ultra_premium_hands = ['AA', 'KK', 'AKs', 'QQ', 'AKo']
premium_hands = ['AQs', 'JJ', 'AQo', 'TT', 'AJs', 'KQs', '99', 'ATs']

pocket_pairs = ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22']
a_high_suited = ['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A5s', 'A4s', 'A3s', 'A7s', 'A2s', 'A6s']
off_premium = ['AKo', 'AQo', 'KQo', 'AJo', 'ATo', 'KJo']
high_suited = ['KQs', 'KJs', 'KTs', 'K9s', 'QTs', 'QTs', 'J9s']
suited_connected = ['QJs', 'JTs', 'T9s', '98s', '87s', '65s', '54s'] 

open_positions = {
    'UTG': utg_open_range,
    'HJ': hj_open_range,
    'CO': co_open_range,
    'BTN': btn_open_range,
    'SB': sb_open_range,
    'BB': bb_open_range,
}

in_2_bet_pots = {
    'BB': bb_in_2_bet_pots
}

def create_pre_flop_charts():
    all_possible_hands = ALL_CARD_COMBINATIONS
    charts = {}
    
    raise_counter = 'open'
    write_first_raise_chart(raise_counter, charts, all_possible_hands)
    raise_counter = 'in_2_bet'
    write_chart(raise_counter, charts, all_possible_hands, add_plus=0)
    raise_counter = 'in_3_bet'
    write_chart(raise_counter, charts, all_possible_hands, add_plus=-2)
    raise_counter = 'in_4_bet'
    write_chart(raise_counter, charts, all_possible_hands, add_plus=4)

    # Ensure the directory exists before writing the file
    os.makedirs(PRE_FLOP_CHARTS_DIR, exist_ok=True)
    
    with open(PRE_FLOP_CHARTS_DIR + '/preflop_charts.json', 'w') as f:
        json.dump(charts, f, indent=2)

def write_first_raise_chart(raise_counter, charts, all_possible_hands):
    charts[raise_counter] = {}
    for position in USED_POSITIONS:
        charts[raise_counter][position] = {}
        
        for hand in all_possible_hands:
            # if hand is premium, raise 95%, call 5%, fold 0%
            if hand in premium_hands or hand in ultra_premium_hands:
                charts[raise_counter][position][hand] = {strings.RAISE.lower(): 100, strings.CALL.lower(): 0, strings.FOLD.lower(): 0}
            else:
                # if hand is in position's open range, calc raise frequncy by previous poisiton's open range
                if hand in open_positions[position]:
                    raise_odds = create_raise_chart(position, 85, hand)
                    if (raise_odds is None):
                        raise_odds = 85
                    
                    # call and fold gets the same odds
                    rest_odds = 100-raise_odds if position == 'BB' else (100 - raise_odds)/2
                    charts[raise_counter][position][hand] = {strings.RAISE.lower(): raise_odds, strings.CALL.lower(): rest_odds, strings.FOLD.lower(): rest_odds}
                else:
                    charts[raise_counter][position][hand] = {strings.RAISE.lower(): 0, strings.CALL.lower(): 0, strings.FOLD.lower(): 100}

# calc open range
def create_raise_chart(position, basic_raise_odds, hand):
    position_index =  USED_POSITIONS.index(position)
    index = -1;
    
    adjustment = 0
    while (position_index + index > 0):
        position_behind = USED_POSITIONS[position_index + index]
        if hand in open_positions[position_behind]:
            adjustment += 10
        else: # if it's not in previous position's open range, we will decrease the raise frequency
            adjustment = adjustment + (20/index)
            break;
        index -= 1

    return min(99, basic_raise_odds + adjustment)

# calc 3bet+ range
def write_chart(raise_counter, charts, all_possible_hands, add_plus=0):
    charts[raise_counter] = {}
    for position in USED_POSITIONS:
        charts[raise_counter][position] = {}
        for hand in all_possible_hands:
            high_cards_arrays = [pocket_pairs, a_high_suited, suited_connected, off_premium, high_suited]
            found = False;
            for (index, high_cards_array) in enumerate(high_cards_arrays):
                if hand in high_cards_array:
                    charts[raise_counter][position][hand] = calc_hand_odds(hand, high_cards_array, index, add_plus)
                    found = True;
                    break;
            if not found:
                charts[raise_counter][position][hand] = {strings.RAISE.lower(): 0, strings.CALL.lower(): 0, strings.FOLD.lower(): 100}


def calc_hand_odds(hand, high_cards_array, array_index, add_plus=0):
    start_index = 0 # before this index, all hands are raise
    gap = 0 # the gap between each hand (probability of raise)
    if (array_index == 0): # pocket pairs  ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22']
        start_index = 3
        add_plus += 3
        gap = 5
    elif (array_index == 1): # a_high_suited ['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A5s', 'A4s', 'A3s', 'A7s', 'A2s', 'A6s']
        start_index = 1
        gap = 10
    elif (array_index == 2): # suited_connected ['AKo', 'AQo', 'KQo', 'AJo', 'ATo', 'KJo']
        start_index = 0
        gap = 15
    elif (array_index == 3): # off_premium ['KQs', 'KJs', 'KTs', 'K9s', 'QTs', 'QTs', 'J9s']
        start_index = 0
        gap = 20
        add_plus -= 1
    elif (array_index == 4): # high_suited ['QJs', 'JTs', 'T9s', '98s', '87s', '65s', '54s'] 
        start_index = 0
        add_plus -= 1
        gap = 20
    
    index = high_cards_array.index(hand)
    fold_value = 0 if index < (start_index+add_plus) else gap*(index + add_plus)
    raise_odds = 100 - (fold_value / 3)
    call_odds = 100 - (2 * fold_value / 3)
    return {strings.RAISE.lower(): raise_odds, strings.CALL.lower(): call_odds, strings.FOLD.lower(): fold_value}

def create_fold_chart(position, basic_fold_odds, hand):
    position_index =  USED_POSITIONS.index(position)
    index = 1;
    
    adjustment = 0
    while (position_index + index < len(USED_POSITIONS)):
        position_after  = USED_POSITIONS[position_index + index]
        if hand not in open_positions[position_after]:
            adjustment += 10
        else:
            adjustment -= (20/index)
            break;
        index = index + 1

    return min(99, basic_fold_odds + adjustment)