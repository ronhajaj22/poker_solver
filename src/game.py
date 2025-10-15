from player import Player
from deck import Deck
from AppUtils import agent_utils
from AppUtils.constants import ALL_POSITIONS_POST_FLOP, ALL_POSITIONS, PREFLOP, FLOP, TURN, RIVER, STREETS
import copy
import pdb
from itertools import chain
import AppUtils.StringUtils as strings

class Game:
    def __init__(self, num_players, main_player_seat, stacks_size = 100, active_player_name = "Ron"):
        self.active_player_position = main_player_seat
        self.num_of_players = num_players
        self.is_heads_up = num_players == 2
        self.players = [
            Player(
            i, 
            self.active_player_position == i + 1,
            stacks_size, 
            0)
            for i in range(num_players)
            
        ]

        for i in range(num_players):
            self.players[i].set_position(num_players)
            self.players[i].set_name(i, num_players, active_player_name)

        self.bb_size = 1;
        self.pot_size = round(sum(player.chips_in_pot for player in self.players), 2);
        self.deck = Deck()
        self.community_cards = []
        self.stacks_size = stacks_size;
        self.current_hand_actions_strength = [0, 0, 0, 0]
        self.prev_actions_in_street = {}
        self.all_prev_actions = {}
        # this save the number of players in the START of each street
        self.num_of_players_in_streets = [num_players, num_players, num_players, num_players]
        self.street = 0
        self.start_new_hand(True);
    
    def print_players(self):
        for player in self.players:
            print(player)

    def get_main_player(self):
        ans = next((p for p in self.players if p.is_main_player), self.get_bb())
        return ans
    
    def get_sb(self):
        return self.players[self.num_of_players - 2]
    
    def get_bb(self):
        return next((p for p in self.players if p.index == self.num_of_players - 1), self.players[self.num_of_players - 1])

    def get_players_in_hand(self):
        return [p for p in self.players if not p.is_folded]
    
    def get_active_players_in_hand(self):
        return [p for p in self.players if not p.is_folded and not p.is_all_in]
    
    def get_players_to_act(self, is_post_flop = False):
        array = ALL_POSITIONS_POST_FLOP if self.street > PREFLOP else ALL_POSITIONS
        return sorted([p for p in self.players if p.is_need_to_act and not p.is_all_in],
            key=lambda p: array.index(p.position))
    
    def get_players_to_act_by_index(self, player_name, is_post_flop = False):
        positions_by_order = ALL_POSITIONS_POST_FLOP if self.street > PREFLOP else ALL_POSITIONS
        position_index = positions_by_order.index(self.get_player_by_name(player_name).position)
        players_to_act_next_round = []
        players_to_act_this_round = []
        for player in self.get_players_to_act():    
            if (positions_by_order.index(player.position) < position_index):
                players_to_act_next_round.append(player)
            else:
                players_to_act_this_round.append(player)
        
        players_to_act_this_round.extend(players_to_act_next_round)
        
        return players_to_act_this_round;
    
    def get_players_to_act_after_deal_cards(self):
        players_to_act = self.get_players_to_act(True)
        if (len(players_to_act) <= 1):
            return [];
        else:
            return players_to_act;
        
    def start_new_hand(self, is_new_game = False):
        self.deck = Deck()
        self.street = 0
        self.community_cards = []
        self.deck.deal_players_cards(self.players)
        self.pot_size = self.bb_size * 1.5
        self.current_hand_actions_strength = [0, 0, 0, 0]
        for player in self.players:
            player.reset_player(self, is_new_game)
        self.prev_actions_in_street = {}
        self.all_prev_actions = {}
        self.current_player_index = 0
        self.last_bet_size = self.bb_size
    
    # change the seat of the players by order, put sb and bb money
    def get_ready_for_next_street(self, street):
        positions_by_order = ALL_POSITIONS_POST_FLOP if street > 0 else ALL_POSITIONS

        for player in self.get_players_in_hand():
            player.is_need_to_act = True;
            player.chips_in_pot = 0
            player.position_index = positions_by_order.index(player.position)

        self.community_cards.extend(self.deck.deal_next_cards(street))
        self.all_prev_actions = {STREETS[self.street]: self.prev_actions_in_street}
        self.prev_actions_in_street = {} # map of position -> list of actions in current street
        self.last_bet_size = 0
        self.street += 1;
        self.num_of_players_in_streets[self.street] = len(self.get_players_in_hand())
        
    def check_action(self, action, bet_size):
        message = self.get_main_player().check_action(action, bet_size, self.num_of_players, self.street, self.get_num_raises_in_current_street());
        return message;
    
    def actPlayer(self, p):
        player = next((pl for pl in self.players if pl.name == p), None)
        
        # calculate the player action
        action = player.act(self);
        print("action amount: ", action.amount)
        action_amount = min(action.amount, player.get_max_size(self, action.action))
        print("new action amount: ", action_amount)
        return self.after_player_action(player, action.action, action_amount)

    def act_after_mcts_choice(self, p, action, amount):
        action_amount = min(amount, p.get_max_size(self, action))
        action, chips_in_pot, added_amount, stack_size = self.after_player_action(p, action, action_amount);
        p.mcts_chips_in_pot += added_amount;
        for index, player in enumerate(self.players):
            if player.name == p.name:
                self.players[index] = p;
                break;
        return self;

    def act_after_human_choice(self, p, action, amount):
        action_amount = min(amount, p.get_max_size(self, action))
        return self.after_player_action(p, action, action_amount);

    def after_player_action(self, player, action, amount):
        player.is_need_to_act = False

        self.current_hand_actions_strength[self.street] = agent_utils.update_action_strength(self.current_hand_actions_strength[self.street], action, amount, self.num_of_players, self.pot_size, self.street)
        
        if amount >= self.last_bet_size:
            self.last_bet_size = amount
            
        if self.prev_actions_in_street.get(player.position) is None:
            self.prev_actions_in_street[player.position] = [action]
        else:
            self.prev_actions_in_street[player.position].append(action)
        added_amount = 0
        
        if action == strings.FOLD:
            player.is_folded = True

        elif action == strings.RAISE or action == strings.CALL:    
            added_amount = round(amount - player.chips_in_pot, 2)

            if added_amount >= player.stack_size:
                player.is_all_in = True
                
            player.stack_size = round(player.stack_size - added_amount, 2)
            player.chips_in_pot = round(player.chips_in_pot + added_amount, 2)

            self.pot_size = round(self.pot_size + added_amount, 2)
            if self.street == PREFLOP:
                player.is_pre_flop_agressor = True
            else:
                player.is_last_agressor = True

            if (action == strings.RAISE):
                for p in self.get_players_in_hand():
                    if p != player:
                        p.is_need_to_act = True
                        if self.street == PREFLOP:
                            p.is_pre_flop_agressor = False
                        else:
                            p.is_last_agressor = False
                    
            elif (self.street == PREFLOP and action == strings.CALL and self.get_num_raises_in_current_street() == 0):
                self.get_bb().is_need_to_act = True

        return action, player.chips_in_pot, added_amount, player.stack_size

    def is_over(self):
        betting_finished = self.street == RIVER and self.get_players_to_act() == []
        one_player_left = len(self.get_players_in_hand()) == 1
        all_players_all_in = True
        for player in self.get_players_in_hand():
            if not player.is_all_in:
                all_players_all_in = False
                break
        one_player_left_new_street = len(self.get_active_players_in_hand()) == 1 and self.prev_actions_in_street == {}

        return betting_finished or one_player_left or all_players_all_in or one_player_left_new_street

    def find_winners(self):
        players_in_hand = self.get_players_in_hand()
        if (len(players_in_hand) == 1):
            players_in_hand[0].stack_size += self.pot_size
            return players_in_hand, []
        else:
            winners, winning_hands = self.deck.find_winners(self.get_players_in_hand(), self.community_cards)
            for winner in winners:
                winner.stack_size += (self.pot_size / len(winners))
            return winners, winning_hands;

    def get_player_by_name(self, name):
        return next((p for p in self.players if p.name == name), None)
    
    def get_num_raises_in_current_street(self): 
        if self.prev_actions_in_street == {}:
            return 0
        return list(chain.from_iterable(self.prev_actions_in_street.values())).count(strings.RAISE)
    
    def __deepcopy__(self, memo):
        """Create a deep copy of the Game object using copy.deepcopy()"""
        # Create a new Game instance with same basic parameters
        new_game = Game(self.num_of_players, self.active_player_position, self.stacks_size)
        
        # Copy all attributes
        new_game.active_player_position = self.active_player_position
        new_game.num_of_players = self.num_of_players
        new_game.is_heads_up = self.is_heads_up
        new_game.bb_size = self.bb_size
        new_game.pot_size = self.pot_size
        new_game.stacks_size = self.stacks_size
        new_game.current_hand_actions_strength = self.current_hand_actions_strength.copy()
        new_game.num_of_players_in_streets = self.num_of_players_in_streets.copy()
        new_game.street = self.street
        new_game.community_cards = self.community_cards.copy()
        new_game.prev_actions_in_street = {k: v.copy() for k, v in self.prev_actions_in_street.items()}
       # new_game.all_prev_actions = {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in self.all_prev_actions.items()}
        new_game.last_bet_size = self.last_bet_size
        
        # Deep copy players
        new_game.players = []
        for player in self.players:
            new_player = Player(player.index, player.is_main_player, player.stack_size, player.chips_in_pot)
            new_player.name = player.name
            new_player.position = player.position
            new_player.position_index = player.position_index
            new_player.is_folded = player.is_folded
            new_player.is_need_to_act = player.is_need_to_act
            new_player.is_all_in = player.is_all_in
            new_player.is_pre_flop_agressor = player.is_pre_flop_agressor
            new_player.is_last_agressor = player.is_last_agressor
            new_player.actions = [action_list.copy() for action_list in player.actions]
            new_player.mcts_chips_in_pot = player.mcts_chips_in_pot
            new_player.hand = copy.deepcopy(player.hand)
            new_game.players.append(new_player)
        
        # Deep copy deck
        new_game.deck = copy.deepcopy(self.deck)
        
        return new_game
    
    