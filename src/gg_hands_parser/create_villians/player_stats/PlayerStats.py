
from .. import printer
from ..game_stats import GameStats
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER
from AppUtils.cards_utils import ALL_CARD_COMBINATIONS
import pdb
from . import pre_flop_stats, flop_stats, turn_stats, river_stats, player_hand_stats

class PlayerStats:
    HAND_KEYS = {combination: 0 for combination in ALL_CARD_COMBINATIONS}
    HAND_KEYS['other'] = 0

    # hand values (two pair, over pair, etc..) - TODO  think about it
    
    
    def __init__(self, player_name: str, player_general_name: str, is_general_player: bool, general_player: 'PlayerStats', is_heads_up: bool):
        """Initialize a new PlayerStats object for a player."""
        self.player_name = player_name
        self.player_general_name = player_general_name
        self.is_general_player = is_general_player
        self.is_heads_up_player = is_heads_up
        self.hand_stats = player_hand_stats.CurrentHandStats()
        self.general_player = general_player
        self.pre_flop_range = None
        
        self.hands_won = 0
        self.wsd = 0 # won with showdown
        self.wnsd = 0 # won with no showdown
        self.wtsd = 0 # went to showdown

        if self.is_general_player:
            self.pre_flop_stats = general_player.pre_flop_stats
            self.flop_stats = general_player.flop_stats
            self.turn_stats = general_player.turn_stats
            self.river_stats = general_player.river_stats
        else:
            self.pre_flop_stats = pre_flop_stats.PreFlopStats(self.HAND_KEYS, is_heads_up);
            self.flop_stats = flop_stats.FlopStats(is_heads_up);
            self.turn_stats = turn_stats.TurnStats(is_heads_up);
            self.river_stats = river_stats.RiverStats(is_heads_up);
            

        self.stats = {
            PREFLOP: self.pre_flop_stats,
            FLOP: self.flop_stats,
            TURN: self.turn_stats,
            RIVER: self.river_stats
        }

    def set_winner(self, is_uncalled_bet):
        self.hand_stats.set_hand_won()

        if self.is_general_player:
            current_player = self.general_player
        else:
            current_player = self
        current_player.hands_won += 1;
        if is_uncalled_bet:
            current_player.wnsd += 1
        else:
            current_player.wsd += 1
    
    def set_stack_size(self, stack_size):
        self.hand_stats.set_stack_size(stack_size)

    def set_player_cards(self, player_cards):
        self.hand_stats.set_player_cards(player_cards)

    def add_action(self, street, board_stats: GameStats, action, size = None):  
        if (size != None):
            size = board_stats.convert_size(size)
        
     #   print("adding action", action, "to street", street)
        self.stats[street].add_action(self, board_stats, action, size)
        self.hand_stats.add_chips_to_pot(board_stats, size if size != None else 0 if action == 'CHECK' else -1, street)
            
        if street == PREFLOP:
            printer.print_pre_flop_data(self.player_name, self.hand_stats, board_stats, action, size)
            self.hand_stats.pfr = action == 'RAISE'
        elif street == RIVER:
            board_stats.river_actions += 1

        # UPDATE GENERAL STATS
        self.hand_stats.actions[street].append(action)
        self.hand_stats.action_sizes[street].append(size)
        if action == 'RAISE':
            board_stats.num_raises[street] += 1
        elif action == 'FOLD':
            board_stats.num_players -= 1
            self.hand_stats.is_folded = True
    
    def build_pre_flop_range(self, board_cards, hero_cards, num_raises):
        if self.pre_flop_range != None:
            return self.pre_flop_range
        
        #self.pre_flop_range = pre_flop_range.get_range(num_raises, self.hand_stats.usable_position, self.hand_stats.pfr)

