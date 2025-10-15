from hand import Hand

from AppUtils.constants import ALL_POSITIONS, ALL_POSITIONS_POST_FLOP
from MCTS.action_logic import ActionLogic
import AppUtils.StringUtils as strings

class Player:

    def __init__(self,index: int, is_player: bool = False, stack_size: float = 100, chips_in_pot: float = 0):
        self.index = index
        self.position_index = index
        self.is_main_player = is_player
        self.chips_in_pot = chips_in_pot
        self.stack_size = stack_size
        self.is_folded = False
        self.is_all_in = False
        self.actions = [[] for _ in range(4)]
        self.is_pre_flop_agressor = False
        self.is_last_agressor = False
        self.mcts_chips_in_pot = 0
        
        # Initialize action logic for decision making
        self.action_logic = ActionLogic()


    def set_position(self, num_of_players):
        self.position = ALL_POSITIONS[len(ALL_POSITIONS) - num_of_players + self.position_index]
        self.is_need_to_act = False if self.position == "BB" else True
        self.chips_in_pot = 1 if self.position == "BB" else 0.5 if self.position == "SB" else 0 # bb_size

    def set_name(self, index, num_of_players, active_player_name):
        self.name = active_player_name if index == num_of_players - 1 else f"Player {index + 1}"
    
    def set_hand(self, cards):
        self.hand = Hand(cards)

    def set_post_flop_player_position_index(self):
        self.position_index = ALL_POSITIONS_POST_FLOP.index(self.position)

    # this function create a meassage for the main player to check his action
    def check_action(self, action, bet_size, num_of_players, street, num_raises):
        return self.action_logic.check_action(action, bet_size, num_of_players, street, num_raises, self)
                
    def act(self, game):
        """Main action method - delegates to action logic for decision making"""
        selected_action = self.action_logic.get_action(game, self)
        self.actions[game.street].append(selected_action.action)
        # Track action in player's action history
        print(f"player {self.name} actions: {self.actions}")
        return selected_action

    def get_possible_actions(self, game):
        """Get possible actions from game state"""

        actions = []
        
        if game.last_bet_size > 0:
            actions.append(strings.FOLD)
            actions.append(strings.CALL)
        else:
            actions.append(strings.CHECK)
        # Only allow raise if stack is sufficient
        if self.get_max_size(game, strings.RAISE) > game.last_bet_size and len(game.get_active_players_in_hand()) > 1:
            actions.append(strings.RAISE)
        
        return actions

    def get_max_size(self, game, action):
        stack_size = self.stack_size + self.chips_in_pot
        largest_stack_size = 0;
        players = game.get_active_players_in_hand() if action == strings.RAISE else game.get_players_in_hand()
        for player in players:
            if player.name != self.name:
                largest_stack_size = max(largest_stack_size, player.stack_size + player.chips_in_pot)
        
        if game.last_bet_size > largest_stack_size:
            largest_stack_size = game.last_bet_size
        return min(stack_size, largest_stack_size)

    def reset_player(self, game, is_new_game):
        self.is_folded = False;
        self.position_index = (self.index + game.num_of_players -  (0 if is_new_game else 1)) % game.num_of_players
        self.index = self.position_index
        self.set_position(game.num_of_players)
        self.is_all_in = False
        self.stack_size -= self.chips_in_pot
        self.chips_in_pot = 0.5 if self.position == "SB" else 1 if self.position == "BB" else 0
        self.mcts_chips_in_pot = 0
        if self.stack_size <= 1:
            self.stack_size = game.stacks_size - self.chips_in_pot
        self.actions = [[] for _ in range(4)]
        
    def __repr__(self):
        return f"{self.name} ({self.position}, {self.stack_size}bb) {self.hand}"
