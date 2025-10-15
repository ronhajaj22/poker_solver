from AppUtils.hand_board_calculations import calc_hand_type, calc_hand_strength
from ..game_stats import GameStats
from AppUtils.constants import RIVER, POST_FLOP_HAND_KEYS, ALL_POSITIONS_POST_FLOP, HU_POSITIONS_POST_FLOP, TURN
from AppUtils.cards_utils import co_open_range
import AppUtils.StringUtils as strings
import pdb

bet_sizing_categories = {
    'small': (0.0, 0.35),
    'medium': (0.35, 0.67),
    'large': (0.67, 1.0),
    'overbet': (1.0, float('inf')),
}

class RiverStats:
    def __init__(self, is_heads_up_player = False):
        self.POST_FLOP_HAND_KEYS = POST_FLOP_HAND_KEYS
        self.post_flop_positions = HU_POSITIONS_POST_FLOP if is_heads_up_player else ALL_POSITIONS_POST_FLOP
        self.name = 'General'
        self.is_heads_up_player = is_heads_up_player

        # create two river categories - 2Playes pot ans multi player pot (then divide them to sub-classes)
        self.river_stats = RiverPlayer()
        if not self.is_heads_up_player:
            self.river_stats_multi_player = RiverPlayer()
        self.init_keys()

    # this function initialize class's keys with all the details we need to save
    def init_keys(self):
        possible_actions_vs_bet = {strings.RAISE: 0, strings.FOLD: 0, strings.CALL: 0}
        possible_outcomes = {"won": 0, "lost": 0}
        self.hands_in_river = 0
        self.checked_to_him_actions = {strings.RAISE: 0, strings.CHECK: 0}
        self.bet_sizes = {"small": 0, "medium": 0, "large": 0, "overbet": 0}
        self.bet_strength = {"small": 0, "medium": 0, "large": 0, "overbet": 0}
        self.vs_different_bet_sizes = {key: possible_actions_vs_bet.copy() for key in self.bet_sizes}

        self.general_calls_counter = possible_outcomes.copy()
        self.calls_counter = {'small': possible_outcomes.copy(), 'medium': possible_outcomes.copy(), 'large': possible_outcomes.copy(), 'overbet': possible_outcomes.copy(), 're_raise': possible_outcomes.copy()}
        self.got_raised_actions = possible_actions_vs_bet.copy()
        self.got_re_raised_actions = possible_actions_vs_bet.copy()
        self.got_4_bet_plus_actions = possible_actions_vs_bet.copy()

        self.overbet_split = {'Bluff': 0, 'Value': 0}
        self.overbet_hand_strength = []
        self.re_raise_split = {'Bluff': 0, 'Value': 0}
        self.re_raise_hand_strength = []
        self.three_bet_split = {'Bluff': 0, 'Value': 0}
        self.three_bet_hand_strength = []

    def calc_size(self, pot_size, bet_size):
            """Calculate bet size as percentage of pot"""
            if pot_size > 0:
                return bet_size / pot_size
            else:
                print("River Error!! pot size is 0")
            return 0

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        if player_stats.hand_stats.actions[RIVER] == []:
            self.initialize_is_in_position(player_stats, board_stats)
        
        self.add_new_action(player_stats, board_stats, action, size)
        river_player = self.river_stats if board_stats.num_players == 2 else self.river_stats_multi_player
        river_player.add_action(player_stats, board_stats, action, size)

    # check if current player is OOP or IP
    # TODO - if the player was IP in flop/turn - he should still be IP in river
    # same if there are only 2 players in the hand OOP
    def initialize_is_in_position(self, player_stats, board_stats: GameStats):
        # Safety check for empty position
        if not player_stats.hand_stats.position or player_stats.hand_stats.position not in self.post_flop_positions:
            print("Error!!! player position is not set")
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
        if player_stats.hand_stats.actions[RIVER] == []:
            self.hands_in_river += 1
        
        if action == strings.RAISE:
            if size != None:
                size = self.calc_size(board_stats.pot_size[TURN], size)

        num_raises = board_stats.num_raises[RIVER]
        if num_raises == 0:
            if action in self.checked_to_him_actions:
                self.checked_to_him_actions[action] += 1
            else:
                print("Error!!! action not found " + action)
            if size != None:
                for category, (min_size, max_size) in bet_sizing_categories.items():
                    if min_size <= size < max_size:
                        self.bet_sizes[category] += 1

                        # TODO - this code shoulde be more organized
                        # save the last bet size to see how opponent react base on bet sizes
                        board_stats.last_bet_size_category = category
                        # if it's overbet, we add the hand to the overbet range (bluff or value)
                        if category == 'overbet':
                            self.add_to_player_range(0, player_stats.hand_stats, board_stats)
                        break
        else: 
            if action == strings.CALL:
                self.update_calls_counter(player_stats.hand_stats.is_hand_won, num_raises, board_stats.last_bet_size_category)
            
            stats_to_update = self.got_raised_actions if num_raises == 1 else self.got_re_raised_actions if num_raises == 2 else self.got_4_bet_plus_actions
            
            if num_raises == 1:
                self.vs_different_bet_sizes[board_stats.last_bet_size_category][action] += 1
            if action in stats_to_update:
                stats_to_update[action] += 1
                if (action == strings.RAISE):
                    self.add_to_player_range(num_raises, player_stats.hand_stats, board_stats)
            else:
                print("Error!!! action not found " + action)

    def update_calls_counter(self, is_hand_won, num_raises, bet_size_category):
        if num_raises > 1:
            bet_size_category = 're_raise'
        self.calls_counter[bet_size_category]['won' if is_hand_won else 'lost'] += 1
        self.general_calls_counter['won' if is_hand_won else 'lost'] += 1

    def add_to_player_range(self, num_raises, hand, board):
        if not hand.is_hand_shown:
            return;

        if hand.player_cards == [] or hand.player_cards == None:
            print("Error!!! player cards are not set");
            return;
        
        hand_strength = calc_hand_strength(hand.player_cards, board.board_cards, co_open_range)
        if num_raises == 0:
            split = self.overbet_split
            hand_strength_list = self.overbet_hand_strength
        elif num_raises == 1:
            split = self.re_raise_split
            hand_strength_list = self.re_raise_hand_strength
        else: # 2+
            split = self.three_bet_split
            hand_strength_list = self.three_bet_hand_strength
        
        # TODO - this is super-basic way to decide if player is bluffing or not
        if hand_strength < 80:
            split['Bluff'] += 1
            if hand_strength > 50:
                print("Semi-bluff! Board cards: ", board.board_cards_str, " Player cards: ", hand.player_cards_str)
        else:
            hand_strength_list.append(hand_strength)
            split['Value'] += 1

    def print_all_stats(self, f):
        self.print_stats(f)
        if not self.is_heads_up_player:
            f.write("2 Players ")
        self.river_stats.print_stats(f)
        
        if not self.is_heads_up_player:
            f.write("MultiPlayer ")
            self.river_stats_multi_player.print_stats(f)

    def print_stats(self, f):
        f.write(f"{self.name} River STATS:\n")
        f.write(f"  Hands number: {self.hands_in_river}\n")
        
        self.print_action_stats(f, self.checked_to_him_actions, "When checked to him")
        self.print_action_stats(f, self.bet_sizes, "Bet Sizes")
        self.print_action_stats(f, self.got_raised_actions, "When got raised")
        for bet_size in self.bet_sizes.keys():
            self.print_action_stats(f, self.vs_different_bet_sizes[bet_size], f"vs {bet_size} bet size")
        self.print_action_stats(f, self.got_re_raised_actions, "When got re-raised")
        self.print_action_stats(f, self.got_4_bet_plus_actions, "When got 4-bet or more")
        self.print_hand_strength_stats(f, self.re_raise_hand_strength, self.re_raise_split, "Re-raise")
        self.print_hand_strength_stats(f, self.three_bet_hand_strength, self.three_bet_split, "Three-bet+")
        self.print_hand_strength_stats(f, self.overbet_hand_strength, self.overbet_split, "Overbet")
        
        self.print_calls_counter(f)
        
        f.write("\n\n")
    
    def print_calls_counter(self, f):
        f.write(f"  Calling outcomes: {self.general_calls_counter}\n")
        f.write(f"  Calling outcomes by bet size: {self.calls_counter}\n")

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

    def print_hand_strength_stats(self, f, hand_strength_list, hand_strength_split, label):
        if len(hand_strength_list) > 0:
            avg_strength = round(sum(hand_strength_list) / len(hand_strength_list), 2)
            f.write(f"  {label} split: {hand_strength_split}\n")
            f.write(f"  Value {label} Avg Strength: {avg_strength}\n")

class RiverPlayer(RiverStats):
    def __init__(self):
        self.name = ''
        self.init_keys()
        self.river_stats_small_pot = RiverStatsSmallPot(); # max - 16bb
        self.river_stats_med_pot = RiverStatsMedPot() # max - 50bb
        self.river_stats_large_pot = RiverStatsLargePot() # bigger than 50bb

        self.river_stats_ip = RiverStatsIP()
        self.river_stats_oop = RiverStatsOOP()

    
    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)

        self.add_action_to_children(player_stats, board_stats, action, size)
    
    def add_action_to_children(self, player_stats, board_stats: GameStats, action, size = None):
        river_stats_to_apply = []
        
        # pot type: SPR, 3Bet or 4Bet
        river_stats_to_apply.append(self.river_stats_small_pot if board_stats.pot_size[TURN] < 16
            else self.river_stats_med_pot if board_stats.pot_size[TURN] < 50 else self.river_stats_large_pot)       
        
        # ip or oop - need to determine based on hand_stats
        river_stats_to_apply.append(self.river_stats_ip if player_stats.hand_stats.ip else self.river_stats_oop)

        for stat in river_stats_to_apply:
            stat.add_action(player_stats, board_stats, action, size)

    def print_stats(self, f):
        super().print_stats(f)  # Call parent's print_stats first
        self.river_stats_small_pot.print_stats(f)
        self.river_stats_med_pot.print_stats(f)
        self.river_stats_large_pot.print_stats(f)
        self.river_stats_ip.print_stats(f)
        self.river_stats_oop.print_stats(f)

class RiverStatsSmallPot(RiverStats):
    def __init__(self):
        self.name = 'Small Pot'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class RiverStatsMedPot(RiverStats):
    def __init__(self):
        self.name = 'Med Pot'   
        self.init_keys()
    
    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class RiverStatsLargePot(RiverStats):
    def __init__(self):
        self.name = 'Large Pot'
        self.init_keys()
    
    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class RiverStatsIP(RiverStats):
    def __init__(self):
        self.name = 'IP'
        self.init_keys()
    
    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f) 
    

class RiverStatsOOP(RiverStats):
    def __init__(self):
        self.name = 'OOP'
        self.init_keys()
    
    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f) 