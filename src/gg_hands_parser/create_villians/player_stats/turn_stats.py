from AppUtils.hand_board_calculations import calc_hand_type, calc_hand_strength
from ..game_stats import GameStats
from AppUtils.constants import TURN, POST_FLOP_HAND_KEYS, ALL_POSITIONS_POST_FLOP, HU_POSITIONS_POST_FLOP, FLOP
from .. import printer
import AppUtils.StringUtils as strings
import pdb

bet_sizing_categories = {
    'small': (0.0, 0.35),
    'medium': (0.35, 0.67),
    'large': (0.67, 1.0),
    'overbet': (1.0, float('inf')),
}


class TurnStats:
    def __init__(self, is_heads_up_player = False):
        self.name = 'General'
        self.is_heads_up_player = is_heads_up_player
        self.post_flop_positions = HU_POSITIONS_POST_FLOP if is_heads_up_player else ALL_POSITIONS_POST_FLOP
        self.turn_stats_2_player_pot = TurnPlayer()
        if not self.is_heads_up_player:
            self.turn_stats_multi_player_pot = TurnPlayer()

        self.init_keys()
        self.POST_FLOP_KEYS = POST_FLOP_HAND_KEYS
    
    def init_keys(self):
        self.hands_in_turn = 0
        self.checked_to_him_actions = {strings.RAISE: 0, strings.CHECK: 0}
        self.open_sizes = {"small": 0, "medium": 0, "large": 0, "overbet": 0}
        self.open_strength = {"small": 0, "medium": 0, "large": 0, "overbet": 0}
        
        self.got_raised_actions = {strings.RAISE: 0, strings.FOLD: 0, strings.CALL: 0}
        self.got_re_raised_actions = {strings.RAISE: 0, strings.FOLD: 0, strings.CALL: 0}
        self.got_4_bet_plus_actions = {strings.RAISE: 0, strings.FOLD: 0, strings.CALL: 0}

    def calc_size(self, pot_size, bet_size):
        """Calculate bet size as percentage of pot"""
        if pot_size > 0:
            return bet_size / pot_size
        else:
            print("Turn Error!! pot size is 0")
        return 0

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        if player_stats.hand_stats.actions[TURN] == []:
            self.initialize_is_in_position(player_stats, board_stats)
        self.add_new_action(player_stats, board_stats, action, size)
        if board_stats.num_players == 2:
            self.turn_stats_2_player_pot.add_action(player_stats, board_stats, action, size)
        else:
            self.turn_stats_multi_player_pot.add_action(player_stats, board_stats, action, size)

    def initialize_is_in_position(self, player_stats, board_stats: GameStats):
        # Safety check for empty position
        if not player_stats.hand_stats.position or player_stats.hand_stats.position not in self.post_flop_positions:
            return
            
        position_index = self.post_flop_positions.index(player_stats.hand_stats.position)
        is_ip = True
        for player in board_stats.players_in_hand():
            if player.hand_stats.position and player.hand_stats.position in self.post_flop_positions:
                if self.post_flop_positions.index(player.hand_stats.position) > position_index:
                    is_ip = False
                    break

        player_stats.hand_stats.ip = is_ip

    def add_new_action(self, player_stats, board_stats: GameStats, action, size = None):
        if player_stats.hand_stats.actions[TURN] == []:
            self.hands_in_turn += 1
        
        if action == strings.RAISE:
            if size != None:
                size = self.calc_size(board_stats.pot_size[FLOP], size)

        num_raises_in_turn = board_stats.num_raises[TURN]
        if num_raises_in_turn == 0:
            if action in self.checked_to_him_actions:
                self.checked_to_him_actions[action] += 1
            else:
                print("Error!!! no raises in turn - choosing action - " + action)
            if size != None:
                for category, (min_size, max_size) in bet_sizing_categories.items():
                    if min_size <= size < max_size:
                        self.open_sizes[category] += 1
                        break
        else: 
            stats_to_update = self.got_raised_actions if num_raises_in_turn == 1 else self.got_re_raised_actions if num_raises_in_turn == 2 else self.got_4_bet_plus_actions
        
            if action in stats_to_update:
                stats_to_update[action] += 1
            else:
                print("Error!!! some raises in turn - choosing action - " + action)
    
    def print_all_stats(self, f):
        self.print_stats(f)
        f.write("\n\n")
        f.write("2 Players:\n")
        self.turn_stats_2_player_pot.print_stats(f)
        f.write("\n\n")

        if not self.is_heads_up_player:
            f.write("MultiPlayer:\n")
            self.turn_stats_multi_player_pot.print_stats(f)
            f.write("\n\n")

    def print_action_stats(self, f, actions_dict, label):
        """Generic function to print action statistics with percentages"""
        if len(actions_dict) > 0:
            total_actions = sum(actions_dict.values())
            if total_actions > 0:
                f.write(f"  {label}: ")
                actions_list = list(actions_dict.items())
                for i, (action, count) in enumerate(actions_list):
                    percentage = round(float(count * 100 / total_actions), 3)
                    f.write(f"{action.capitalize()}: {count} ({percentage}%)")
                    if i < len(actions_list) - 1:
                        f.write(f", ")
                f.write(f"\n")
        
    def print_stats(self, f):
        f.write(f"\n\n{self.name} Turn STATS:\n")
        f.write(f"  Hands in Turn: {self.hands_in_turn}\n")
        
        self.print_action_stats(f, self.checked_to_him_actions, "When checked to him")
        self.print_action_stats(f, self.open_sizes, "Open Sizes")
        self.print_action_stats(f, self.got_raised_actions, "When got raised")
        self.print_action_stats(f, self.got_re_raised_actions, "When got re-raised")
        self.print_action_stats(f, self.got_4_bet_plus_actions, "When got 4-bet or more")

         
class TurnPlayer(TurnStats):
    def __init__(self):
        self.name = ''
        self.init_keys()
        self.turn_stats_srp = TurnStatsSRP();
        self.turn_stats_3bet_plus = TurnStats3BetPlus()

        self.turn_stats_ip = TurnStatsIP()
        self.turn_stats_oop = TurnStatsOOP()

        self.turn_stats_fr = TurnStatsFR()
        self.turn_stats_fc = TurnStatsFC()
    
    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)

        self.add_action_to_children(player_stats, board_stats, action, size)
    
    def add_action_to_children(self, player_stats, board_stats: GameStats, action, size = None):
        turn_stats_to_apply = []
        
        # pot type: SRP or 3Bet+
        num_raises =  board_stats.num_raises[0]
        turn_stats_to_apply.append(self.turn_stats_srp if num_raises < 2 else self.turn_stats_3bet_plus)
       
        # ip or oop - need to determine based on hand_stats
        turn_stats_to_apply.append(self.turn_stats_ip if player_stats.hand_stats.ip else self.turn_stats_oop)

        # pre-flop last aggressor - need to determine based on hand_stats
        turn_stats_to_apply.append(self.turn_stats_fr if player_stats.hand_stats.fr else self.turn_stats_fc)
       
        for stat in turn_stats_to_apply:
            stat.add_action(player_stats, board_stats, action, size)

    def print_stats(self, f):
        super().print_stats(f)  # Call parent's print_stats first
        self.turn_stats_srp.print_stats(f)
        self.turn_stats_3bet_plus.print_stats(f)
        self.turn_stats_ip.print_stats(f)
        self.turn_stats_oop.print_stats(f)
        self.turn_stats_fr.print_stats(f)
        self.turn_stats_fc.print_stats(f)

class TurnStatsSRP(TurnStats):
    def __init__(self):
        self.name = 'SRP'
        self.init_keys()
    
    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class TurnStats3BetPlus(TurnStats):
    def __init__(self):
        self.name = '3Bet+'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class TurnStatsIP(TurnStats):
    def __init__(self):
        self.name = 'IP'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class TurnStatsOOP(TurnStats):
    def __init__(self):
        self.name = 'OOP'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class TurnStatsFR(TurnStats):
    def __init__(self):
        self.name = 'Flop Raiser'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class TurnStatsFC(TurnStats):
    def __init__(self):
        self.name = 'Flop Caller'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)


    def calc_hand(self, hand_stats, board_stats: GameStats, action, size = None):
        hand_type = None
        if hand_stats.is_hand_shown:
            hand_result = calc_hand_type(hand_stats.player_cards, board_stats.board_cards[:4])
            hand_type = hand_result[0]  # Extract just the hand type string from the tuple
            hand_strength = calc_hand_strength(hand_stats.player_cards, board_stats.board_cards[:4])