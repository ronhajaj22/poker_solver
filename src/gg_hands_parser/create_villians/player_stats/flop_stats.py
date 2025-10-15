from AppUtils.hand_board_calculations import calc_hand_type, calc_hand_strength, calculate_flush_draw
from ..game_stats import GameStats
from AppUtils.constants import FLOP, POST_FLOP_HAND_KEYS, ALL_POSITIONS_POST_FLOP, HU_POSITIONS_POST_FLOP, PREFLOP
from AppUtils.cards_utils import co_open_range
import AppUtils.StringUtils as strings
from .. import printer
import pdb

bet_sizing_categories = {
    'small': (0.0, 0.35),
    'medium': (0.35, 0.67),
    'large': (0.67, 1.0),
    'overbet': (1.0, float('inf')),
}

possible_actions_vs_bet = [strings.RAISE, strings.FOLD, strings.CALL]
possible_actions = [strings.RAISE, strings.CHECK]


class FlopStats:
    def __init__(self, is_heads_up_player = False):
        self.name = 'General'
        self.is_heads_up_player = is_heads_up_player
        self.post_flop_positions = HU_POSITIONS_POST_FLOP if is_heads_up_player else ALL_POSITIONS_POST_FLOP
        self.flop_stats_2_player_pot = FlopPlayer()
        if not self.is_heads_up_player:
            self.flop_stats_multi_player_pot = FlopPlayer()

        self.init_keys()
        self.POST_FLOP_KEYS = POST_FLOP_HAND_KEYS
    
    def init_keys(self):
        self.hands_in_flop = 0
        self.checked_to_him_actions = {strings.RAISE: 0, strings.CHECK: 0}
        self.open_sizes = {key: 0 for key in bet_sizing_categories.keys()}
        self.open_strength = {key: 0 for key in bet_sizing_categories.keys()}
        
        self.vs_raise_actions = {action: 0 for action in possible_actions_vs_bet}
        self.vs_different_bet_sizes = {bet_size: {action: 0 for action in possible_actions_vs_bet} for bet_size in bet_sizing_categories.keys()}
        self.vs_re_raise_actions = {action: 0 for action in possible_actions_vs_bet}
        self.vs_4_bet_plus_actions = {action: 0 for action in possible_actions_vs_bet}

        # how the player plays his big hands
        self.big_hand_play = {action: 0 for action in possible_actions}
        self.big_hand_play_vs_raise = {action: 0 for action in possible_actions_vs_bet}
        # how the player plays his big hands on flush board (#TODO - change it to a dynamic board)
        self.big_hand_on_flush_board = {action: 0 for action in possible_actions}
        self.big_hand_on_flush_board_vs_raise = {action: 0 for action in possible_actions_vs_bet}
        # how the player plays his flush draws
        self.flush_draw_play = {action: 0 for action in possible_actions}
        self.flush_draw_play_vs_raise = {action: 0 for action in possible_actions_vs_bet}

        # TODO - not implemented yet
        self.re_raise_strength = [];
        self.overbet_strength = [];

    
    def calc_size(self, pot_size, bet_size):
        """Calculate bet size as percentage of pot"""
        if pot_size > 0:
            return bet_size / pot_size
        else:
            print("Error!! pot size is 0")
        return 0

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        if player_stats.hand_stats.actions[FLOP] == []:
            self.initialize_is_in_position(player_stats, board_stats)
        self.add_new_action(player_stats, board_stats, action, size)
        if board_stats.num_players == 2:
            self.flop_stats_2_player_pot.add_action(player_stats, board_stats, action, size)
        else:
            self.flop_stats_multi_player_pot.add_action(player_stats, board_stats, action, size)

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
        if player_stats.hand_stats.actions[FLOP] == []:
            self.hands_in_flop += 1
        
        if action == 'RAISE':
            if size != None:
                size = self.calc_size(board_stats.pot_size[PREFLOP], size)
        
        self.fr = action == 'RAISE'
        
        num_raises_in_flop = board_stats.num_raises[FLOP]
        if num_raises_in_flop == 0:
            if action in self.checked_to_him_actions:
                self.checked_to_him_actions[action] += 1

                if player_stats.hand_stats.is_hand_shown:
                    # TODO - this is currently called only on checked to him action
                    self.add_big_hand_play(player_stats, board_stats, action, False)
            else:
                print("Flop Error!!! action not found " + action)
            if size != None:
                for category, (min_size, max_size) in bet_sizing_categories.items():
                    if min_size <= size < max_size:
                        self.open_sizes[category] += 1
                    
                        # TODO - this code should be more organized
                        # save the last bet size to see how opponent react base on bet sizes
                        board_stats.last_bet_size_category = category
                        break
        else:
            if player_stats.hand_stats.is_hand_shown and num_raises_in_flop == 1:
                # TODO - this is currently called only on checked to him action
                self.add_big_hand_play(player_stats, board_stats, action, True)

            stats_to_update = self.vs_raise_actions if num_raises_in_flop == 1 else self.vs_re_raise_actions if num_raises_in_flop == 2 else self.vs_4_bet_plus_actions
        
            if num_raises_in_flop == 1 and board_stats.last_bet_size_category != None and action in self.vs_different_bet_sizes[board_stats.last_bet_size_category]:
                self.vs_different_bet_sizes[board_stats.last_bet_size_category][action] += 1
            elif num_raises_in_flop == 1 and board_stats.last_bet_size_category == None:
                print("Error!!! last bet size category is None")
            if action in stats_to_update:
                stats_to_update[action] += 1
            else:
                print("Error!!! action not found " + action)
    
    def add_big_hand_play(self, player_stats, board_stats: GameStats, action, vs_bet = False):
        hand_strength = calc_hand_strength(player_stats.hand_stats.player_cards, board_stats.board_cards[:3])
        if board_stats.board_config.to_flush_board_flop:
            # TODO - this is a made up way to decide the opponent range, need to improve it
            if hand_strength > 85:
                if vs_bet:
                    self.big_hand_on_flush_board_vs_raise[action] += 1
                else:
                    self.big_hand_on_flush_board[action] += 1
            elif calculate_flush_draw(player_stats.hand_stats.player_cards, board_stats.board_cards[:3]) > 0.65:
                if vs_bet:
                    self.flush_draw_play_vs_raise[action] += 1
                else:
                    self.flush_draw_play[action] += 1
        elif hand_strength > 90:
            if vs_bet:
                self.big_hand_play_vs_raise[action] += 1
            else:
                self.big_hand_play[action] += 1

    def print_all_stats(self, f):
        self.print_stats(f)
        f.write("\n\n")
        f.write("2 Players:\n")
        self.flop_stats_2_player_pot.print_stats(f)
        f.write("\n\n")

        if not self.is_heads_up_player:
            f.write("MultiPlayer:\n")
            self.flop_stats_multi_player_pot.print_stats(f)
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
        f.write(f"\n\n{self.name} Flop STATS:\n")
        f.write(f"  Hands in Flop: {self.hands_in_flop}\n")
        
        self.print_action_stats(f, self.checked_to_him_actions, "When checked to him")
        self.print_action_stats(f, self.open_sizes, "Open Sizes")
        self.print_action_stats(f, self.vs_raise_actions, "When got raised")
        for bet_size in self.vs_different_bet_sizes.keys():
            self.print_action_stats(f, self.vs_different_bet_sizes[bet_size], f"vs {bet_size} bet size")

        self.print_action_stats(f, self.vs_re_raise_actions, "When got re-raised")
        self.print_action_stats(f, self.vs_4_bet_plus_actions, "When got 4-bet or more")

        self.print_action_stats(f, self.big_hand_on_flush_board, "Big hand on flush board (when checked to him)")
        self.print_action_stats(f, self.big_hand_on_flush_board_vs_raise, "Big hand on flush board (when got raised)")
        self.print_action_stats(f, self.flush_draw_play, "Flush draw play (when checked to him)")
        self.print_action_stats(f, self.flush_draw_play_vs_raise, "Flush draw play (when got raised)")
        self.print_action_stats(f, self.big_hand_play, "Big hand play (when checked to him)")
        self.print_action_stats(f, self.big_hand_play_vs_raise, "Big hand play (when got raised)")

class FlopPlayer(FlopStats):
    def __init__(self):
        self.name = ''
        self.init_keys()
        self.flop_stats_srp = FlopStatsSRP();
        self.flop_stats_3bet = FlopStats3Bet()
        self.flop_stats_4bet = FlopStats4Bet();

        self.flop_stats_ip = FlopStatsIP()
        self.flop_stats_oop = FlopStatsOOP()

        self.flop_stats_pfr = FlopStatsPFR()
        self.flop_stats_pfc = FlopStatsPFC()
    
    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)

        self.add_action_to_children(player_stats, board_stats, action, size)
    
    def add_action_to_children(self, player_stats, board_stats: GameStats, action, size = None):
        flop_stats_to_apply = []
        
        # pot type: SRP, 3Bet or 4Bet
        num_raises =  board_stats.num_raises[0]
        flop_stats_to_apply.append(self.flop_stats_srp if num_raises <= 1
            else self.flop_stats_3bet if num_raises == 2 else self.flop_stats_4bet)
       
        # ip or oop - need to determine based on hand_stats
        flop_stats_to_apply.append(self.flop_stats_ip if player_stats.hand_stats.ip else self.flop_stats_oop)

        # pre-flop last aggressor - need to determine based on hand_stats
        flop_stats_to_apply.append(self.flop_stats_pfr if player_stats.hand_stats.pfr else self.flop_stats_pfc)
       
        for stat in flop_stats_to_apply:
            stat.add_action(player_stats, board_stats, action, size)

    def print_stats(self, f):
        super().print_stats(f)  # Call parent's print_stats first
        self.flop_stats_srp.print_stats(f)
        self.flop_stats_3bet.print_stats(f)
        self.flop_stats_4bet.print_stats(f)
        self.flop_stats_ip.print_stats(f)
        self.flop_stats_oop.print_stats(f)
        self.flop_stats_pfr.print_stats(f)
        self.flop_stats_pfc.print_stats(f)

class FlopStatsSRP(FlopStats):
    def __init__(self):
        self.name = 'SRP'
        self.init_keys()
    
    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class FlopStats3Bet(FlopStats):
    def __init__(self):
        self.name = '3Bet'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class FlopStats4Bet(FlopStats):
    def __init__(self):
        self.name = '4Bet'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class FlopStatsIP(FlopStats):
    def __init__(self):
        self.name = 'IP'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class FlopStatsOOP(FlopStats):
    def __init__(self):
        self.name = 'OOP'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class FlopStatsPFR(FlopStats):
    def __init__(self):
        self.name = 'Pre-Flop Raiser'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)

class FlopStatsPFC(FlopStats):
    def __init__(self):
        self.name = 'Pre-Flop Caller'
        self.init_keys()

    def add_action(self, player_stats, board_stats: GameStats, action, size = None):
        super().add_new_action(player_stats, board_stats, action, size)
    
    def print_stats(self, f):
        super().print_stats(f)


    def calc_hand(self, hand_stats, board_stats: GameStats, action, size = None):
        hand_type = None
        if hand_stats.is_hand_shown:
            hand_result = calc_hand_type(hand_stats.player_cards, board_stats.board_cards[:3])
            hand_type = hand_result[0]  # Extract just the hand type string from the tuple
            hand_strength = calc_hand_strength(hand_stats.player_cards, board_stats.board_cards[:3])