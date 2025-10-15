from .player_stats import PlayerStats   
from AppUtils.constants import FLOP, TURN, RIVER
import AppUtils.StringUtils as strings

is_printer_active = False

def print_pre_flop_data(player_name, hand_stats, board_stats, action, size):
    if not is_printer_active:
        return
    value = hand_stats.position + ": "
    value += player_name + " "
    if (hand_stats.is_hand_shown):
        value += "(" + hand_stats.player_cards_str + ") "
    value += " | ";
    if action == strings.FOLD:
        value += "folds";
    elif action == strings.RAISE:
        value += " raises to " + str(size + hand_stats.chips_in_pot);
    elif action == strings.CALL:
        value += " calls " + str(size) + " BBs";
    else:
        value += " Checks";
    print(value)

def print_new_hand(current_players):
    if not is_printer_active:
        return
    my_str = ("*** NEW HAND ***\n")
    my_str += get_current_players(current_players, "Hand")
    print(my_str)

def print_street(board_stats, players, street):
    if street == FLOP:
        print_flop(board_stats, players)
    elif street == TURN:
        print_turn(board_stats, players)
    elif street == RIVER:
        print_river(board_stats, players)

def print_flop(board_stats, players):
    if not is_printer_active:
        return
    my_str = "*** FLOP ***" + " (pot: " + str(board_stats.pot_size) + ") " + get_board_cards(board_stats, 3) + "\n"
    my_str += get_current_players(players, "Flop")
    print(my_str)

### ADD HERE ###

def print_c_bet(player_name, hand_stats, board_stats, action, size, hand_type, street):
    if not is_printer_active:
        return
    
    value = player_name + " "
    if (hand_stats.is_hand_shown):
        value += "(" + hand_stats.player_cards_str + ") "
    value += " | C-bet " + street + " to " + str(size)
    if hand_type:
        value += " with " + hand_type
    print(value)

def print_open(player_name, hand_stats, board_stats, action, size, hand_type, street):
    if not is_printer_active:
        return
    
    value = player_name + " "
    if (hand_stats.is_hand_shown):
        value += "(" + hand_stats.player_cards_str + ") "
    value += " | Open " + street + " to " + str(size)
    if hand_type:
        value += " with " + hand_type
    print(value)

def print_c_check(player_name, street):
    if not is_printer_active:
        return
    
    value = player_name + " | C-check "
    print(value)

def print_check(player_name, street):
    if not is_printer_active:
        return
    
    value = player_name + " | Check "
    print(value)

def print_check_raise(player_name, hand_stats, board_stats, action, size, hand_type, street):
    if not is_printer_active:
        return
    
    value = player_name + " "
    if (hand_stats.is_hand_shown):
        value += "(" + hand_stats.player_cards_str + ") "
    value += " | Check-raise " + street + " to " + str(size + hand_stats.chips_in_pot)
    if hand_type:
        value += " with " + hand_type
    print(value)

def print_re_raise(player_name, hand_stats, board_stats, action, size, hand_type, street):
    if not is_printer_active:
        return
    
    value = player_name + " "
    if (hand_stats.is_hand_shown):
        value += "(" + hand_stats.player_cards_str + ") "
    value += " | Re-raise " + street + " to " + str(size)
    if hand_type:
        value += " with " + hand_type
    print(value)

def print_fold(player_name, street):
    if not is_printer_active:
        return
    
    value = player_name + " | Fold " + street
    print(value)

def print_call(player_name, street):
    if not is_printer_active:
        return
    
    value = player_name + " | Call " + street
    print(value)

###  END ###
def print_turn(board_stats, players):
    if not is_printer_active:
        return
    my_str = "*** TURN *** " + " (pot: " + str(board_stats.pot_size) + ") " + get_board_cards(board_stats, 4) + "\n"
    my_str += get_current_players(players, "Turn")
    print(my_str)

def print_river(board_stats, players):
    if not is_printer_active:
        return
    my_str = "*** RIVER *** " + "(pot: " + str(board_stats.pot_size) + ") " + get_board_cards(board_stats, 5) + "]" + "\n"
    my_str += get_current_players(players, "River")
    print(my_str)

def get_current_players(players, street):
    my_str = "Players in " + street + ": "
    for player in players:
        if not player.hand_stats.is_folded:
            my_str += player.player_name
            my_str += (" (" + player.hand_stats.player_cards_str + ")" if player.hand_stats.is_hand_shown else "")
            my_str += " (" + str(player.hand_stats.stack_size) + "), "
    return my_str

def get_board_cards(board_stats, street):
    my_str = '['
    index = 0
    for card in board_stats.board_cards_str:
        my_str += card + (", " if index < street-1 else "")
        index += 1
        if index == street:
            break
    
    my_str += ']'
    return my_str