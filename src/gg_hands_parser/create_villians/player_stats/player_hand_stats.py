import hand
import deck
from AppUtils.constants import get_general_position
import pdb
class CurrentHandStats:
    def __init__(self):
        self.player_cards = []
        self.player_cards_str = ''
        
        self.stack_size = 0
        self.chips_in_pot = 0
        self.position = ''
        
        self.actions = [[], [], [], []] # [preflop, flop, turn, river]
        self.action_sizes = [[], [], [], []]
        
        self.is_folded = False
        self.is_hand_shown = False
        self.is_hand_won = False

        # TODO - implement
        self.ip = False
        self.pfr = False # pre-flop raiser
        self.fr = False # flop raiser
        self.tr = False # turn raiser
        self.last_bet_size_category = None

    def start_new_hand(self):
        self.actions = [[], [], [], []] # TODO - implement
        self.action_sizes = [[], [], [], []] # TODO - implement
        self.player_cards = []
        self.player_cards_str = ''
        self.chips_in_pot = 0
        self.ip = False
        self.pfr = False # TODO - need to delete
        self.fr = False # TODO - need to delete
        self.tr = False # TODO - need to delete
        self.is_hand_shown = False
        self.position = ''
        self.is_folded = False
        self.last_bet_size_category = None
        self.is_hand_won = False

    def set_player_cards(self, player_cards):
        self.is_hand_shown = True
        self.player_cards = [deck.Card(card[0], card[1]) for card in player_cards]
        
        # Use the Hand class to_string method - TODO: move this function to UTILS 
        hand_instance = hand.Hand(self.player_cards)
        self.player_cards_str = hand_instance.to_string(self.player_cards)

    def set_stack_size(self, stack_size):
        self.stack_size = round(stack_size, 2)

    def set_position(self, position):
        self.position = position
        self.usable_position = get_general_position(position)
    
    def add_chips_to_pot(self, board_stats, amount, street):
        # Convert amount to int if it's a string
        if isinstance(amount, str):
            amount = float(amount)
        if amount > 0:
            self.chips_in_pot = round(self.chips_in_pot + amount, 2)
            self.stack_size = round(self.stack_size - amount, 2)
        board_stats.add_to_bet_history(street, amount)
            

    def set_hand_won(self):
        self.is_hand_won = True