from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, get_general_position
from AppUtils.files_utils import load_hu_preflop_chart, load_preflop_chart
from hands_classifier.hand_board_calculations import calc_strength_pre
import AppUtils.StringUtils as strings
from actions_logic.postflop_action_logic import AgentActions, Action
from actions_logic.preflop_reader import create_range
import pdb
import random;


class ActionLogic:
    _instance = None
    is_initialized = False
    
    # create new instance if not exists
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ActionLogic, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ActionLogic.is_initialized:
            self.is_initialized = False
            self.preflop_chart = load_preflop_chart()
            self.hu_preflop_chart = load_hu_preflop_chart()
            ActionLogic.is_initialized = True

    def get_preflop_action(self, game, player):
        num_raises = game.get_num_raises_in_current_street()
        
        raise_is_not_possible = game.last_bet_size > player.stack_size + player.chips_in_pot # if last bet size is greater than player's stack size, the player can't raise
        fold_is_not_possible = game.last_bet_size <= player.chips_in_pot

        hand_str = player.hand.to_string()
        position_general = get_general_position(player.position)
        
        if game.is_heads_up:
            situation = ('open' if position_general == 'SB' else 'vs_limp') if num_raises == 0 else 'vs_open' if num_raises == 1 else 'vs_3bet' if num_raises == 2 else 'vs_4bet' if num_raises == 3 else 'vs_5bet'
            if (hand_str in self.hu_preflop_chart[situation]):
                action_freqs = self.hu_preflop_chart[situation][hand_str]
            else:
                print("hand_str", hand_str, "in position", player.position, "in stage", game.street, "not found")
                action_freqs = {strings.RAISE.lower(): 0, strings.CALL.lower(): 0, strings.FOLD.lower(): 100}
            possible_actions = self.create_action_preflop(action_freqs, raise_is_not_possible, game.last_bet_size, fold_is_not_possible)
            print("possible_actions", possible_actions)

        else: 
            situation = ('open' if player.position != 'BB' else 'vs_limp') if num_raises == 0 else 'vs_open' if num_raises == 1 else 'vs_3bet' if num_raises == 2 else 'vs_4bet' if num_raises == 3 else 'vs_5bet'
            if situation == 'open':
                action_freqs = self.preflop_chart[situation][position_general][hand_str]
            elif situation == 'vs_limp':
                action_freqs = self.preflop_chart[situation][hand_str]
            else: 
                if situation == 'vs_open':
                    is_ip = player.position != 'BB'
                else:
                   is_ip = player.is_ip(game.get_positions_by_order(), game.get_active_players_in_hand())
                if hand_str not in self.preflop_chart[situation + ('_ip' if is_ip else '_oop')]:
                    print("hand_str", hand_str, "in position", player.position, "in stage", game.street, "not found")
                    action_freqs = {strings.RAISE.lower(): 0, strings.CALL.lower(): 0, strings.FOLD.lower(): 100}
                else:
                    action_freqs = self.preflop_chart[situation + ('_ip' if is_ip else '_oop')][hand_str]
            
            for action in action_freqs:
                action_freqs[action] = round(action_freqs[action], 2)
            possible_actions = self.create_action_preflop(action_freqs, raise_is_not_possible, game.last_bet_size, fold_is_not_possible)
        
        # Choose and return the selected action
        selected_action = self.choose_action(possible_actions, raise_is_not_possible, player.stack_size)

        return selected_action

    #  Create action objects from frequency dictionary
    def create_action_preflop(self, action_freqs, is_no_raise, last_bet_size, fold_is_not_possible):
        raise_freq = action_freqs[strings.RAISE.lower()]
        if (strings.CALL.lower() in action_freqs):
            call_check_action = Action(strings.CALL, action_freqs[strings.CALL.lower()], last_bet_size)
        else:
            call_check_action = Action(strings.CHECK, action_freqs[strings.CHECK.lower()], 0)
        if (strings.FOLD.lower() in action_freqs):
            fold_freq = action_freqs[strings.FOLD.lower()]
        else:
            fold_freq = 0
        print("raise_freq", raise_freq, "call/check freq", call_check_action.freq, "fold_freq", fold_freq)
        
        if is_no_raise:
            call_check_action.freq += raise_freq
            raise_freq = 0
        
        possible_actions = []
        if ('raise' in action_freqs or 'raise_big' in action_freqs):
            if ('raise_big' in action_freqs):
                possible_actions = [Action(strings.RAISE, action_freqs['raise_big'], last_bet_size*5),
                                    Action(strings.RAISE, action_freqs['raise'], last_bet_size*2.5)]
            else:
                possible_actions = [Action(strings.RAISE, raise_freq, last_bet_size*2.5)]
        
            possible_actions.append(Action(strings.FOLD, fold_freq, 0))
            possible_actions.append(call_check_action)
        else:
            possible_actions = [call_check_action, Action(strings.FOLD, fold_freq, 0)]

        return possible_actions

    def choose_action(self, possible_actions, no_raise=False, stack_size=100):
        if (no_raise):
            possible_actions = [a for a in possible_actions if a.action != strings.RAISE]
        
        # Safety check: if no actions available, create a default fold action
        if not possible_actions:
            possible_actions = [Action(strings.CHECK, 100, 0)]
        
        weights = [a.freq for a in possible_actions]
        selected_action = random.choices(possible_actions, weights=weights, k=1)[0]

        return selected_action

    #  Create a message for the main player to check his action
    def check_preflop_action(self, num_of_players, num_raises, player):
        is_heads_up = num_of_players == 2
        hand_str = player.hand.to_string() 

        action_freqs = {}
    
        if is_heads_up:
            situation = ('open' if player.position == 'SB' else 'vs_limp') if num_raises == 0 else 'vs_open' if num_raises == 1 else 'vs_3bet' if num_raises == 2 else 'vs_4bet' if num_raises == 3 else 'vs_5bet'
            action_freqs = self.hu_preflop_chart[situation][hand_str]
            is_there_valid_json = True;
        else:
            action_freqs = self.get_preflop_actions_freqs(num_raises, player, hand_str)
            is_there_valid_json = True;

        if is_there_valid_json:
            parts = [] # Build a readable string without a trailing comma
            for (key, value) in action_freqs.items():
                if value == 0:
                    continue
                parts.append(f"{key.capitalize().replace('_', ' ')}: {round(value, 2)}%")
            return "GTO solution: " + ", ".join(parts)
        else:
            print("hand_str", hand_str, "in position", player.position, "not found")
    
    def get_preflop_actions_freqs(self, num_raises, player, hand_str):
        situation = ('open' if player.position != 'BB' else 'vs_limp') if num_raises == 0 else 'vs_open' if num_raises == 1 else 'vs_3bet' if num_raises == 2 else 'vs_4bet' if num_raises == 3 else 'vs_5bet'
        if situation == 'open':
            position_general = get_general_position(player.position)
            action_freqs = self.preflop_chart[situation][position_general][hand_str]
        elif situation == 'vs_limp':
            action_freqs = self.preflop_chart[situation][hand_str]
        else: 
            # TODO - this is incorrect - need to check if the player is in position
            is_ip = player.position != 'BB' and player.position != 'SB'
            if hand_str not in self.preflop_chart[situation + ('_ip' if is_ip else '_oop')]:
                print("hand_str", hand_str,  "not found in " + situation)
                action_freqs = {strings.RAISE.lower(): 0, strings.CALL.lower(): 0, strings.FOLD.lower(): 100}
            else:
                action_freqs = self.preflop_chart[situation + ('_ip' if is_ip else '_oop')][hand_str]

        return action_freqs

# Export functions for backward compatibility
def get_preflop_action(game, player):
    return ActionLogic().get_preflop_action(game, player)

def check_preflop_action(num_of_players, num_raises, player):
    return ActionLogic().check_preflop_action(num_of_players, num_raises, player)
