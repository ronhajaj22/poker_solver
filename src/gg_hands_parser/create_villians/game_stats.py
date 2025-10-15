
from collections import Counter
from AppUtils.constants import RIVER
import deck

class GameStats:
    """
    Class to track board-level statistics for poker hands
    """
    
    def __init__(self, bb_size=2, num_players=6, pot_size=0, board_cards=None):
        # Board statistics
        self.bb_size = bb_size
        self.num_players_pre_flop = num_players
        self.num_players = num_players
        self.pot_size = [float(pot_size), 0, 0, 0]
        self.river_actions = 0;
        self.board_cards_str = ""
        if board_cards != None:
            self.board_cards_str = board_cards
            self.board_cards = [deck.Card(card[0], card[1]) for card in board_cards]
        else:
            self.board_cards = []
            self.board_cards_str = ""
        self.last_raiser = None
        self.num_raises = [0, 0, 0, 0]  # [preflop, flop, turn, river]
        self.bet_history = [[], [], [], []] # [preflop, flop, turn, river]
        self.active_players = []
        self.board_config = BoardConfig(self.board_cards, self.board_cards_str)
        self.last_bet_size_category = None

    def add_to_pot_size(self, street, added_chips):
        if added_chips == 0 or street > RIVER:
            print(f"Warning: Invalid street index {street}. Valid range is 0-3.")
            return;
        # Ensure street is a valid index (0-3 for preflop, flop, turn, river)
        self.pot_size[street] = round(self.pot_size[street] + added_chips, 2)
            
    def set_last_raiser(self, last_raiser):
        self.last_raiser = last_raiser

    def add_to_bet_history(self, street, bet_size):
        self.bet_history[street].append(bet_size)
        if bet_size > 0:
            self.add_to_pot_size(street, bet_size)

    def convert_size(self, size):
        bbs = round(float(size)/self.bb_size, 2)
        return bbs

    def players_in_hand(self):
        return [player for player in self.active_players if not player.hand_stats.is_folded]

    def save_pot_to_next_street(self, street):
        self.pot_size[street] = self.pot_size[street - 1]

class BoardConfig:
    def __init__(self, board_cards, board_cards_str):
        self.board_cards_str = board_cards_str
        self.board_cards = board_cards
        self.to_flush_board_flop = self.to_flush_board_flop(board_cards[:3])
        self.to_flush_board_turn = self.to_flush_board_turn(board_cards[:4])
    
    def to_flush_board_flop(self, board_cards):
        suit_counts = Counter([c.suit for c in board_cards])
        return 2 in suit_counts.values()
    
    def to_flush_board_turn(self, board_cards):
        suit_counts = Counter([c.suit for c in board_cards])
        return 2 in suit_counts.values()
