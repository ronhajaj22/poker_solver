from hand import Hand
import AppUtils.StringUtils as strings
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, ALL_POSITIONS, ALL_POSITIONS_POST_FLOP
from agent.features_heuristics import update_street_actions_value
from hands_classifier.hand_range_updater import update_player_range_after_action

from actions_logic.preflop_action_logic import get_preflop_action, check_preflop_action
from actions_logic.postflop_action_logic import AgentActions, Action

class Player:

    def __init__(self,index: int, is_player: bool = False, stack_size: float = 100, chips_in_pot: float = 0):
        self.index = index
        self.position_index = index # TODO - remove this. bad implementation
        self.is_main_player = is_player
        self.chips_in_pot = chips_in_pot
        self.stack_size = stack_size
        self.is_folded = False
        self.is_all_in = False
        self.actions = [[] for _ in range(4)]
        self.is_pre_flop_agressor = False
        self.is_flop_agressor = False
        self.is_turn_agressor = False
        self.mcts_chips_in_pot = 0
        self.action_strength = [0 for _ in range(4)]
        # Initialize action logic for decision making
        self.range = None # range - map of hand: {prob, strength}
        self.range_strength = 0.5 # NOTE: this is unused - can be removed
        self.is_range_strength_updated = False

    def set_position(self, num_of_players):
        self.position = ALL_POSITIONS[len(ALL_POSITIONS) - num_of_players + self.position_index]
        
    def set_name(self, index, num_of_players, active_player_name):
        self.name = active_player_name if index == num_of_players - 1 else f"Player {index + 1}"
    
    def set_hand(self, cards):
        self.hand = Hand(cards)

    def set_post_flop_player_position_index(self):
        self.position_index = ALL_POSITIONS_POST_FLOP.index(self.position)

    # this function create a meassage for the main player to check his action
    def check_action(self, action, bet_size, num_of_players, street, num_raises):
        return check_preflop_action(num_of_players, num_raises, self)

    def update_action_strength(self, street, action, action_size, last_bet_size, pot_size):
        if street == PREFLOP:
            return
        
        self.action_strength[street] += update_street_actions_value(action, action_size, pot_size, last_bet_size)
        if action == strings.CALL:
            self.action_strength[street] = min(self.action_strength[street], 0.9)
        elif action == strings.RAISE:
            self.action_strength[street] = min(self.action_strength[street], 1)

        if street == FLOP:
            self.is_flop_agressor = action == strings.RAISE
        elif street == TURN:
            self.is_turn_agressor = action == strings.RAISE

    def find_action(self, game):
        selected_action = get_preflop_action(game, self) if game.street == PREFLOP else AgentActions().mcts_act(game, self)
        self.actions[game.street].append(selected_action.action)
        print(f"player {self.name} actions: {self.actions}")
        return selected_action

    # NOTE: this is unused - can be removed
    def set_range_strength(self, action, last_bet_size, is_ip, street, num_of_raises_in_street, game):
        if street == PREFLOP:
            return
        M_CALL = 1 + (last_bet_size)

        if action == strings.RAISE:
            if num_of_raises_in_street > 1: # second raise is already added when we change the strength
                self.range_strength = self.range_strength / (num_of_raises_in_street + last_bet_size)
                return;
            self.range_strength = self.range_strength / M_CALL
        elif action == strings.CALL:
            M_CALL = 1 + (last_bet_size)
            self.range_strength = self.range_strength / M_CALL
        elif action == strings.CHECK:
            # TODO - check how often is check in this spot
            if is_ip:
                self.range_strength = min(0.6, self.range_strength*1.15)
            else:
                self.range_strength = self.range_strength + 0.05 if street == FLOP or street == TURN else self.range_strength * 1.15
        
    
    def is_ip(self, positions_by_order, players_in_hand):
        is_ip = True;
        player_position_index = positions_by_order.index(self.position)
        for player in players_in_hand:
            if positions_by_order.index(player.position) > player_position_index:
                is_ip = False
                break;
        return is_ip

    def update_range_after_action(self, game, action):
        if action == strings.FOLD or game.street == PREFLOP or not hasattr(self, 'range') or self.range is None:
            return
        
        update_player_range_after_action(game, self, action)
        self.is_range_strength_updated = True
    '''
    def add_action_to_json_file(self, game, action, action_amount):
        
        num_of_players = len(game.get_players_in_hand())
        bet_size_category = ''
        if game.get_num_raises_in_current_street() == 1:
            bet_size_category = get_bet_size_category(game.last_bet_size/(game.pot_size-game.last_bet_size))
        elif game.get_num_raises_in_current_street() == 0:
            bet_size_category = get_bet_size_category(action_amount/(game.pot_size-game.last_bet_size))

        #if game.street == FLOP:
          #  add_to_json_file(game.is_heads_up, game.get_num_raises_preflop(), game.get_num_raises_in_current_street(), game.street,
         #       num_of_players, is_ip, self.is_pre_flop_agressor, bet_size_category, action)
    '''       
    def reset_player(self, game, is_new_game):
        
        self.position_index = (self.index + game.num_of_players -  (0 if is_new_game else 1)) % game.num_of_players
        self.index = self.position_index
        self.set_position(game.num_of_players)
        self.chips_in_pot = 1 if self.position == "BB" else 0.5 if self.position == "SB" else 0
        self.stack_size -= self.chips_in_pot
        self.mcts_chips_in_pot = 0
        if self.stack_size <= 1:
            self.stack_size = game.stacks_size - self.chips_in_pot
        self.actions = [[] for _ in range(4)]
        self.range = None
        self.range_strength = 0.5
        self.action_strength = [0 for _ in range(4)]
        self.is_need_to_act = False if self.position == "BB" else True

        self.is_folded = False;
        self.is_all_in = False
        self.is_pre_flop_agressor = False
        self.is_flop_agressor = False
        self.is_turn_agressor = False    
        self.is_range_strength_updated = False
    def __repr__(self):
        return f"{self.name} ({self.position}, {self.stack_size}bb) {self.hand}"
