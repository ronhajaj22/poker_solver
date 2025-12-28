from itertools import chain
import copy, pdb
from player import Player
from deck import Deck
import AppUtils.StringUtils as strings
from AppUtils.constants import ALL_POSITIONS_POST_FLOP, ALL_POSITIONS, PREFLOP, FLOP, TURN, RIVER, STREETS, HU_POSITIONS_POST_FLOP, HU_USED_POSITIONS, ALL_POSITIONS, ALL_POSITIONS_POST_FLOP
from AppUtils.actions_utils import find_max_size_in_live_play
from hands_classifier.hands_strength_classifier import find_winners
from actions_logic.preflop_reader import create_range
from MCTS.players_stats_reader import Reader

class Game:
    def __init__(self, num_players, main_player_seat, stacks_size = 100, club_mode = False, active_player_name = "Hero"):
        self.main_player_seat = main_player_seat # Hero's seat
        self.num_of_players = num_players # num of players in the table
        self.is_heads_up = num_players == 2
        self.players = [Player(i, self.main_player_seat == i + 1, stacks_size,  0) for i in range(num_players)]
        self.bb_size = 1;
        self.stacks_size = stacks_size; # starting stacks size
        
        for i in range(num_players):
            self.players[i].set_name(i, num_players, active_player_name)

        self.start_new_hand(True);
        self.club_mode = club_mode

    def start_new_hand(self, is_new_game = False):
        for player in self.players:
            player.reset_player(self, is_new_game)

        self.pot_size = round(sum(player.chips_in_pot for player in self.players), 2);
        self.last_bet_size = self.bb_size

        self.deck = Deck()
        self.deck.deal_players_cards(self.players)
        self.community_cards = []
        
        self.street = PREFLOP
        self.num_of_players_in_streets = [self.num_of_players for _ in range(4)] # num of players in the START of each street
        self.num_raises_preflop = 0; # num of raises in preflop
        
        self.prev_actions_in_street = {} # map of street -> {position -> list of actions in current street}
        self.all_prev_actions = {} # map of position -> list of actions in current street

    # change the seat of the players by order, put sb and bb money
    def get_ready_for_next_street(self):
        self.community_cards.extend(self.deck.deal_next_cards(self.street))
        self.all_prev_actions[STREETS[self.street]] = self.prev_actions_in_street
        self.street += 1;
        positions_by_order = self.get_positions_by_order()
        for player in self.get_active_players_in_hand():
            player.is_need_to_act = True;
            player.chips_in_pot = 0
            player.is_range_strength_updated = False
            player.position_index = positions_by_order.index(player.position)
        
        self.prev_actions_in_street = {}
        self.last_bet_size = 0
       
        self.num_of_players_in_streets[self.street] = len(self.get_players_in_hand())
    
    def get_players_in_hand(self):
        return [p for p in self.players if not p.is_folded]
    
    # all the players in the hand that are not folded and not all in
    def get_active_players_in_hand(self):
        all_players_in_hand = [p for p in self.players if not p.is_folded and not p.is_all_in]
        all_players_in_hand.sort(key=lambda x: x.position_index)
        return all_players_in_hand

    # all the players in the hand that are need to act
    def get_players_to_act(self):
        return [p for p in self.get_active_players_in_hand() if p.is_need_to_act]

    # when player_name made a raise, this function will return the players to act after him
    def get_players_to_act_by_index(self, player_name):
        positions_by_order = self.get_positions_by_order()
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
        players_to_act = self.get_players_to_act()
        if (len(players_to_act) <= 1):
            return [];
        else:
            return players_to_act;
    
    def get_positions_by_order(self):
        if self.is_heads_up:
            return HU_POSITIONS_POST_FLOP if self.street > PREFLOP else HU_USED_POSITIONS
        else:
            return ALL_POSITIONS_POST_FLOP if self.street > PREFLOP else ALL_POSITIONS

    # when player's turn to act, this function will find the right action and calculate the game state
    def actPlayer(self, p):
        player = next((pl for pl in self.players if pl.name == p), None)
        
        action = player.find_action(self); # search for the best action
        action_amount = min(action.amount, find_max_size_in_live_play(self, player))
        if action.action == strings.RAISE:
            action_amount = self.go_all_in_if_needed(action_amount, player)
        
        print(f"player {player.name} action: {action.action}, amount: {action_amount}")
        player.update_range_after_action(self, action.action)
        return self.after_player_action(player, action.action, action_amount) # update the game state

    def get_mcts_advice(self):
        player = [p for p in self.players if p.is_main_player][0]
        
        action = player.find_action(self); # search for the best action
        action_amount = min(action.amount, find_max_size_in_live_play(self, player))
        if action.action == strings.RAISE:
            action_amount = self.go_all_in_if_needed(action_amount, player)
        
        return action.action, action_amount
    
    # this the same as actPlayer, but for mcts algorithm - just for the simulation
    def act_after_mcts_choice(self, p, action, amount):
        action_amount = min(amount, find_max_size_in_live_play(self, p))
        if action == strings.RAISE:
            action_amount = self.go_all_in_if_needed(action_amount, p)

        action, chips_in_pot, added_amount, stack_size = self.after_player_action(p, action, action_amount);

        p.mcts_chips_in_pot += added_amount;
        for index, player in enumerate(self.players):
            if player.name == p.name:
                self.players[index] = p;
                break;
        
       # player.update_range_after_action(self, action)
        return self;

    # when the user made a choice, this function will update the game state
    def act_after_human_choice(self, p, action, amount):
        action_amount = min(amount, find_max_size_in_live_play(self, p))
        #p.add_action_to_json_file(self, action, action_amount) # TODO - need to implement it
        
        p.update_range_after_action(self, action)
        return self.after_player_action(p, action, action_amount);

    def go_all_in_if_needed(self, amount, player):
        # TODO - this check only for the player's stack, but not for the other players' stacks so it can bet over their stack size
        # or to maybe amount*1.2 is the second player's stack size
        if amount*1.2 > player.stack_size + player.chips_in_pot:
            return round(player.stack_size + player.chips_in_pot, 2)
        return amount
    
    # this function will update the game state after a player (user/mcts/agent) made an action
    def after_player_action(self, player, action, amount):
        player.is_need_to_act = False
        player.update_action_strength(self.street, action, amount-player.chips_in_pot, self.last_bet_size, self.pot_size)
        # update action strength and set is_last_agressor flag
        
        if amount >= self.last_bet_size:
            self.last_bet_size = amount
            
        self.prev_actions_in_street.setdefault(player.position, []).append(action)

        added_amount = 0
        
        is_ip = player.is_ip(self.get_positions_by_order(), self.get_active_players_in_hand())
        if self.street == PREFLOP and action != strings.FOLD:
            player.range = create_range(action, self.is_heads_up, is_ip, player.position, self.get_num_raises_preflop(), False, amount=amount)
        
        player.set_range_strength(action, amount/self.pot_size, is_ip, self.street, self.get_num_raises_in_current_street(), self.community_cards)

        if action == strings.FOLD:
            player.is_folded = True

        elif action == strings.RAISE or action == strings.CALL:    
            added_amount = round(amount - player.chips_in_pot, 2)

            if added_amount >= player.stack_size:
                player.is_all_in = True
                
            player.stack_size = round(player.stack_size - added_amount, 2)
            player.chips_in_pot = round(player.chips_in_pot + added_amount, 2)
            
            self.pot_size = round(self.pot_size + added_amount, 2)
            if action == strings.RAISE:
                if self.street == PREFLOP:
                    player.is_pre_flop_agressor = True
                    self.num_raises_preflop += 1

            self.activate_players_after_raise(player, action) # make players to act after the raise

        return action, player.chips_in_pot, added_amount, player.stack_size


    def activate_players_after_raise(self, player, action):
        if (action == strings.RAISE):
            for p in self.get_active_players_in_hand():
                if p != player:
                    p.is_need_to_act = True
                    if self.street == PREFLOP:
                        p.is_pre_flop_agressor = False
                    
        elif (self.street == PREFLOP and action == strings.CALL and self.last_bet_size == self.bb_size):
            self.get_bb().is_need_to_act = True

    # situations where the betting round is over
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

    # find the winners of the hand and distribute the pot
    def find_winners(self):
        players_in_hand = self.get_players_in_hand()
        if (len(players_in_hand) == 1):
            players_in_hand[0].stack_size += self.pot_size
            return players_in_hand, []
        else:
            winners = find_winners(self.get_players_in_hand(), self.community_cards)
            for winner in winners:
                winner.stack_size += (self.pot_size / len(winners))
            winning_hands = [winner.hand for winner in winners]
            return winners, winning_hands;

    # check action - compare to agent's action to user's action
    def check_action(self, action, bet_size):
        message = ""
        if self.street == PREFLOP:
            message = self.get_main_player().check_action(action, bet_size, self.num_of_players, self.street, self.get_num_raises_in_current_street());
        '''
        else:
            
            #action, amount = self.get_mcts_advice()
            if action == strings.RAISE:
                message = f"MCTS advice: {action}, {amount}"
            else:
                message = f"MCTS advice: {action}"
        '''
        return message;
    
    # check num of raises in current street
    def get_num_raises_in_current_street(self): 
        if self.prev_actions_in_street == {}:
            return 0
        return list(chain.from_iterable(self.prev_actions_in_street.values())).count(strings.RAISE)
    
    # same, but just for preflop action
    def get_num_raises_preflop(self):
        if self.num_raises_preflop != 0:
            return self.num_raises_preflop;
        if self.all_prev_actions != {}:
            actions_preflop = self.all_prev_actions[STREETS[PREFLOP]]
            self.num_raises_preflop = list(chain.from_iterable(actions_preflop.values())).count(strings.RAISE)
        return self.num_raises_preflop;

    def get_player_by_name(self, name):
        player = next((p for p in self.players if p.name == name), None)
        if player is None:
            print(f"ERROR: Player '{name}' not found in game!")
            print(f"Available players: {[p.name for p in self.players]}")
            raise ValueError(f"Player '{name}' not found in game")
        return player

    def get_player_by_position(self, position):
        player = next((p for p in self.players if p.position == position), None)
        if player is None:
            print(f"ERROR: Player '{position}' not found in game!")
            print(f"Available positions: {[p.position for p in self.players]}")
            raise ValueError(f"Player '{position}' not found in game")
        return player

    # get the main player (user)
    def get_main_player(self):
        player = next((p for p in self.players if p.is_main_player), self.get_bb())
        if player is None:
            print(f"ERROR: Main player not found in game!")
            print(f"Available players: {[p.name for p in self.players]}")
            raise ValueError(f"Main player not found in game")
        return player

    def get_action_category_range(self, action, bet_size):
        if self.street == RIVER:
            action_reader = Reader()
            categories_probs = action_reader.predict_action_category(self.get_main_player(), self, action)
            if categories_probs is not None:
                return action_reader.categories_to_string(categories_probs, action)
        else:
            return None

    def get_sb(self):
        return self.players[self.num_of_players - 2]
    
    def get_bb(self):
        return next((p for p in self.players if p.index == self.num_of_players - 1), self.players[self.num_of_players - 1])
    
    def print_players(self):
        for player in self.players:
            print(player)

    # this will be called in the mcts algorithm to create a deep copy of the game
    def __deepcopy__(self, memo):
        # Create a deep copy of the Game object using copy.deepcopy()
        # Create a new Game instance with same basic parameters
        new_game = Game(self.num_of_players, self.main_player_seat, self.stacks_size)
        
        # Copy all attributes
        new_game.is_heads_up = self.is_heads_up
        new_game.bb_size = self.bb_size
        new_game.pot_size = self.pot_size
        new_game.num_of_players_in_streets = self.num_of_players_in_streets.copy()
        new_game.street = self.street
        new_game.community_cards = self.community_cards.copy()
        new_game.prev_actions_in_street = {k: v.copy() for k, v in self.prev_actions_in_street.items()}
        new_game.all_prev_actions = {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in self.all_prev_actions.items()}
        new_game.last_bet_size = self.last_bet_size
        
        # Deep copy players
        new_game.players = []
        all_added_players_positions = []
        for player in self.get_players_in_hand():
            new_player = Player(player.index, player.is_main_player, player.stack_size, player.chips_in_pot)
            new_player.name = player.name
            new_player.position = player.position
            new_player.position_index = player.position_index
            new_player.is_folded = player.is_folded
            new_player.is_need_to_act = player.is_need_to_act
            new_player.is_all_in = player.is_all_in
            new_player.is_pre_flop_agressor = player.is_pre_flop_agressor
            new_player.is_flop_agressor = player.is_flop_agressor
            new_player.is_turn_agressor = player.is_turn_agressor
            new_player.actions = [street.copy() for street in player.actions]
            new_player.mcts_chips_in_pot = player.mcts_chips_in_pot
            new_player.hand = copy.deepcopy(player.hand)
            new_player.action_strength = list(player.action_strength)  # Create new list to avoid reference sharing
            
            new_game.players.append(new_player)
            new_player.range_strength = player.range_strength
            new_player.range = player.range.copy()
            all_added_players_positions.append(new_player.position)
        
        for player in self.players:
            if player.position not in all_added_players_positions:
                new_game.players.append(player)
        # Deep copy deck
        new_game.deck = copy.deepcopy(self.deck)
        
        return new_game
    
    