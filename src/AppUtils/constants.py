#TODO - think about it
POST_FLOP_HAND_KEYS = {
    'Straight Flush': 0, 'Four of a Kind': 0, 'Full House': 0, 'Flush': 0, 'Straight': 0, 'Set': 0, 
    'Three of a Kind': 0, 'Two Pair': 0, 'Pair': 0, 'High Card': 0,
    'Over Pair': 0, 
    'Top Pair Top Kicker': 0, 'Top Pair Second Kicker': 0, 'Top Pair': 0, 
    'Mid Pair': 0, 'Low Pair': 0,
    'Combo Draw': 0, 'Nuts Flush Draw': 0, 'Flush Draw': 0, 'Straight Draw': 0, 
    'Top Pair Flush Draw': 0, 'Mid Pair Flush Draw': 0, 'Low Pair Flush Draw': 0,
    'other': 0
}
RIVER_HANDS_KEYS = {
    'Nuts',
    'Straight Flush',
    'Four of a Kind',
    'Full House',
    'Flush',
    'Set',
    'Trips',
    'Two Pair',
    'Over Pair'
    'Top Pair',
    'UnderPair'
    'Second Pair'
    'Third Pair',
    'Low Pair'
    'Ace High',
    'King High',
    'No Hand'
}

# ACTIONS
ACTIONS = ['FOLD', 'CHECK', 'CALL', 'RAISE']
# STREETS
PREFLOP = 0; 
FLOP = 1
TURN = 2; 
RIVER = 3
STREETS = ['PRE_FLOP', 'FLOP', 'TURN', 'RIVER']

# POSITIONS
ALL_POSITIONS = ["UTG", "UTG+1", "MP", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
ALL_POSITIONS_POST_FLOP = ["SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO", "BTN"]
USED_POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
HU_USED_POSITIONS = ["SB", "BB"]
USED_POSITIONS_POST_FLOP = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
HU_POSITIONS_POST_FLOP = ["BB", "SB"]
ALL_POSITIONS_REVERSED = ["BB", "SB", "BTN", "CO", "HJ", "LJ", "MP", "UTG+1", "UTG"]
ALL_POSITIONS_REVERSED_POST_FLOP = ["BTN", "CO", "HJ", "LJ", "MP", "UTG+1", "UTG", "BB", "SB"]


# ['UTG', 'MP', 'HJ', 'CO', 'BTN', 'SB', 'BB']
def get_general_position(position):
    if "UTG" in position or "MP" in position:
        return "UTG"
    elif "LJ" in position:
        return "HJ"
    return position

def pretty_suit(suit):
    cards_symbols = {'h' : '\u2665', 'd' : '\u2666', 'c' : '\u2663', 's' : '\u2660'}
    return cards_symbols[suit]





