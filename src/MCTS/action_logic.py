from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, get_general_position
from AppUtils.files_utils import find_heads_up_json_name, load_preflop_chart
from AppUtils.hand_board_calculations import calc_strength_pre
import AppUtils.StringUtils as strings
from MCTS.agent_actions import AgentActions, Action, State
from itertools import chain
import random

class ActionLogic:
    """Main action logic coordinator - Singleton pattern"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ActionLogic, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not ActionLogic._initialized:
            print("Initializing ActionLogic singleton")
            self.agent_actions = AgentActions()
            self.preflop_chart = load_preflop_chart()
            ActionLogic._initialized = True

    def get_action(self, game, player):
        """ Main entry point for getting player action. Returns: Action: chosen action """
        
        if game.street == PREFLOP: # pre flop action using preflop charts
            action = self.handle_preflop_action(game, player)
        else: # post flop action using supervised agent
            action = self.handle_postflop_action(game, player)
        
        # TODO - choose amount here
        return action

    def handle_preflop_action(self, game, player):
        """Handle preflop actions using preflop charts"""
        # Calculate number of raises from previous actions
        num_raises = game.get_num_raises_in_current_street()
        
        raise_is_not_possible = game.last_bet_size > player.stack_size + player.chips_in_pot # if last bet size is greater than player's stack size, the player can't raise
        hand_str = player.hand.to_string()
        position_general = get_general_position(player.position)
        
        if game.is_heads_up:
            situation = find_heads_up_json_name(num_raises, game.stacks_size, player.position)
            if (hand_str in self.preflop_chart[situation]):
                action_freqs = self.preflop_chart[situation][hand_str]
            else:
                print("hand_str", hand_str, "in position", player.position, "in stage", game.street, "not found")
                action_freqs = {strings.RAISE.lower(): 0, strings.CALL.lower(): 0, strings.FOLD.lower(): 100}
            possible_actions = self.create_action_preflop(action_freqs, raise_is_not_possible, game.last_bet_size)
            print("possible_actions", possible_actions)

        else: # Multi-way logic
            if num_raises == 0 or num_raises == 1:
                fold_is_not_possible = game.last_bet_size <= player.chips_in_pot
                situation = 'open' if num_raises == 0 else 'in_2_bet'
                action_freqs = self.preflop_chart[situation][position_general][hand_str]
                possible_actions = self.create_action_preflop(action_freqs, raise_is_not_possible, game.last_bet_size, fold_is_not_possible)
            elif num_raises == 2:  # Two raises - vs 3-bet - not a valid json yet
                hand_strength = calc_strength_pre(player.hand.cards)
                if hand_strength < 60:
                    possible_actions = [Action(strings.FOLD, 100, 0)]
                else:
                    action_1 = Action(strings.RAISE, 50, 3*game.last_bet_size)
                    action_2 = Action(strings.CALL, 50, game.last_bet_size)
                    possible_actions = [action_1, action_2] 
            else:  # Three raises - vs 4-bet - not a valid json yet
                hand_strength = calc_strength_pre(player.hand.cards)
                if hand_strength > 80:
                    # Simple shove/fold for now, can be improved with stack size logic
                    action_1 = Action(strings.CALL if raise_is_not_possible else strings.RAISE, 100, player.stack_size + player.chips_in_pot)
                    possible_actions = [action_1]
                else:
                    action_1 = Action(strings.FOLD, 100, 0)
                    possible_actions = [action_1]
        
        # Choose and return the selected action
        selected_action = self.choose_action(possible_actions, raise_is_not_possible, game.street, player.stack_size)
        return selected_action

    def handle_postflop_action(self, game, player):
        # Check if MCTS is available
        try:
            # Use MCTS to find best action
            selected_action = self.agent_actions.mcts_act(game, player)
            
            # If MCTS failed, use fallback
            if selected_action is False:
                print(f"MCTS failed. Using fallback logic.")
                return self._get_fallback_postflop_actions(game, player)
            
            return selected_action
            
        except Exception as e:
            print(f"MCTS failed: {e}. Using fallback logic.")
            return self._get_fallback_postflop_actions(game, player)
        

    def _get_fallback_postflop_actions(self, game, player):
        """Fallback postflop logic when ML is not available"""
        num_raises = game.get_num_raises_in_current_street();
        
        if (num_raises == 0):
            possible_actions = [Action(strings.RAISE, 25, game.pot_size*0.55), 
            Action(strings.RAISE, 25, game.pot_size*0.5), Action(strings.CHECK, 50, 0)]
        else:
            if (game.last_bet_size >= player.stack_size):
                possible_actions = [Action(strings.CALL, 50, player.stack_size), 
                Action(strings.FOLD, 50, 0)]
            else:
                possible_actions = [Action(strings.RAISE, 10, game.last_bet_size*3),
                 Action(strings.RAISE, 10, game.last_bet_size*2), Action(strings.CALL, 40, game.last_bet_size), Action(strings.FOLD, 40, 0)]
        
        # Choose and return the selected action
        selected_action = self.choose_action(possible_actions, False, game.street, player.stack_size)
        return selected_action

    def create_action_preflop(self, action_freqs, is_no_raise, last_bet_size, fold_is_not_possible):
        """Create action objects from frequency dictionary"""
        print("action_freqs", action_freqs)
        raise_freq = action_freqs[strings.RAISE.lower()]
        if (strings.CALL.lower() in action_freqs):
            call_check_action = Action(strings.CALL, action_freqs[strings.CALL.lower()], last_bet_size)
        else:
            call_check_action = Action(strings.CHECK, action_freqs[strings.CHECK.lower()], 0)
        fold_freq = action_freqs[strings.FOLD.lower()]
        print("raise_freq", raise_freq, "call/check freq", call_check_action.freq, "fold_freq", fold_freq)
        
        if is_no_raise:
            call_check_action.freq += raise_freq
            raise_freq = 0
        
        # TODO - change the preflop charts and remove this
        if (fold_is_not_possible):
            call_check_action.action = strings.CHECK
            call_check_action.size = 0
            call_check_action.freq += fold_freq
            fold_freq = 0
            
        
        possible_actions = []
        if (raise_freq > 0):
            possible_actions = [Action(strings.RAISE, raise_freq/2, last_bet_size*2.5),
                Action(strings.RAISE, raise_freq/2, last_bet_size*3),
                Action(strings.FOLD, fold_freq, 0),
                call_check_action]
        else:
            possible_actions = [call_check_action, Action(strings.FOLD, fold_freq, 0)]

        return possible_actions

    def choose_action(self, possible_actions, no_raise=False, street=0, stack_size=100):
        """Choose action from possible actions based on weights"""
        
        if (no_raise):
            possible_actions = [a for a in possible_actions if a.action != strings.RAISE]
        
        # Safety check: if no actions available, create a default fold action
        if not possible_actions:
            print(f"Warning: No possible actions available for street {street}, defaulting to CHECK")
            possible_actions = [Action(strings.CHECK, 100, 0)]
        
        weights = [a.freq for a in possible_actions]
        selected_action = random.choices(possible_actions, weights=weights, k=1)[0]

        return selected_action

    def check_action(self, action, bet_size, num_of_players, street, num_raises, player):
        """
        Create a message for the main player to check his action
        
        Args:
            action: The action to check
            bet_size: Size of the bet
            num_of_players: Number of players in the game
            street: Current street (PRE_FLOP, FLOP, etc.)
            prev_actions: Previous actions in the hand
            player: Player object
            
        Returns:
            str: Message about the action quality
        """
        is_heads_up = num_of_players == 2
        hand_str = player.hand.to_string() 

        action_freqs = {}
    
        is_there_valid_json = False;
        if is_heads_up and street == PREFLOP:
            situation = find_heads_up_json_name(num_raises, player.stack_size, player.position)
            action_freqs = self.preflop_chart[situation][hand_str]
            is_there_valid_json = True;
        elif street == PREFLOP:
            situation = 'open' if num_raises == 0 else 'in_2_bet'
                    
            position = get_general_position(player.position)
            action_freqs = self.preflop_chart[situation][position][hand_str]
            is_there_valid_json = True;
        else:
            print("not found because street is", street, "and is heads up", is_heads_up)
        
        if is_there_valid_json:
            for (key, value) in action_freqs.items():
                if key.upper() == action.upper():
                    if value >= 50:
                        return f" This is the best action with {hand_str} in this spot, the solver {key}s here {value}% of the time"
                    elif value > 20:
                        return f"This is a solid action with {hand_str} in this spot. the solver {key}s here {value}% of the time"
                    else:
                        return f"This is a bad action with {hand_str} in this spot. the solver {key}s here {value}% of the time"

            return "In this position, the solver has no idea what to do"
        else:
            print("hand_str", hand_str, "in position", player.position,"in stage", street, "not found")
