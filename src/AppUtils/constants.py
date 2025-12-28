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






