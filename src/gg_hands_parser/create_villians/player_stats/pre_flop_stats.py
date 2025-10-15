from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, ACTIONS, get_general_position, USED_POSITIONS, ALL_POSITIONS_POST_FLOP, HU_USED_POSITIONS, HU_POSITIONS_POST_FLOP
from AppUtils.cards_utils import ALL_CARD_COMBINATIONS
from ..game_stats import GameStats
import AppUtils.StringUtils as strings
import pdb


class PreFlopStats:
    def __init__(self, hand_keys, is_heads_up):
        self.HAND_KEYS = hand_keys
        self.IN_POSITION = 'SB' if is_heads_up else 'ip'
        self.OUT_OF_POSITION = 'BB' if is_heads_up else 'oop'
        self.total_hands = 0;
        
        #### Preflop stats ####
        self.vpip = 0 # VPIP percentage
        self.pfr = 0 # PFR percentage
        
        self.vpip_count = 0
        self.pre_flop_raises = 0
        self.rfi = 0

        self.used_positions = HU_USED_POSITIONS if is_heads_up else USED_POSITIONS
        self.is_heads_up_player = is_heads_up

        self.preflop_actions = {position: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0, strings.CHECK: 0} for position in self.used_positions}
        self.preflop_actions_vs_aggression = {position: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0} for position in self.used_positions}
        self.preflop_actions_in_3_bet_pot = {self.IN_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}, self.OUT_OF_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}}
        if not is_heads_up:
            self.preflop_actions_in_cold_3bet_pot = {self.IN_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}, self.OUT_OF_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}}
        self.preflop_actions_in_4_bet_pot = {self.IN_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}, self.OUT_OF_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}}
        self.preflop_actions_in_5_bet_pot = {self.IN_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}, self.OUT_OF_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}}
        self.preflop_actions_in_6_bet_pot = {self.IN_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}, self.OUT_OF_POSITION: {strings.RAISE: 0, strings.CALL: 0, strings.FOLD: 0}}
        self.raise_range = {position: self.HAND_KEYS.copy() for position in USED_POSITIONS}
        self.call_3bet_range = self.HAND_KEYS.copy()
        
        self.three_bet_range = {position: self.HAND_KEYS.copy() for position in (['BB', 'other'] if not is_heads_up else ['BB', 'SB'])}
        self.four_bet_range = self.HAND_KEYS.copy()
        self.call_4bet_range = self.HAND_KEYS.copy()
        self.five_bet_range = self.HAND_KEYS.copy()
        self.call_5bet_range = self.HAND_KEYS.copy()
        self.six_bet_range = self.HAND_KEYS.copy()
        
        # Mapping between number of raises and the relevant action dictionary
        self.raises_to_actions_map = {
            1: self.preflop_actions_vs_aggression,  # vs_Bet
            2: self.preflop_actions_in_3_bet_pot,   # vs_3Bet
            3: self.preflop_actions_in_4_bet_pot,   # vs_4Bet
            4: self.preflop_actions_in_5_bet_pot,   # vs_5Bet
            5: self.preflop_actions_in_6_bet_pot    # vs_6Bet+
        }

    def calc_avgs(self):
        self.vpip = self.vpip_calc()
        self.pfr = self.pfr_calc()
    
   
    def vpip_calc(self):
        if self.total_hands == 0:
            print("Error! player has no hands")
            return 0
        return round(float(self.vpip_count*100)/self.total_hands, 2)

    def pfr_calc(self):
        if self.total_hands == 0:
            print("Error! player has no hands")
            return 0
        return round(float(self.pre_flop_raises*100)/self.total_hands, 2)
    
    def add_action(self, player, board_stats: GameStats, action, size = None):
        if player.hand_stats.actions[PREFLOP] == []:
            self.total_hands += 1
            if action != strings.FOLD:
                board_stats.active_players.append(player)
            if action == strings.CALL or self.is_raise(action):
                self.vpip_count += 1
                if self.is_raise(action):
                    self.pre_flop_raises += 1
                
        position = self.calc_position(player.hand_stats, board_stats)
        if self.is_raise(action):
            board_stats.set_last_raiser(player)
        
        player.hand_stats.pfr = self.is_raise(action)

        num_raises = board_stats.num_raises[PREFLOP]
        if num_raises == 0:
            self.add_to_pre_flop_open_actions(player.hand_stats, action)
        elif num_raises == 1:
            self.add_to_pre_flop_vs_aggression_actions(player.hand_stats, action)
        else:
            self.add_to_pre_flop_multi_aggression_actions(player.hand_stats.actions[PREFLOP], position, action, board_stats)

        if (action != strings.FOLD and player.hand_stats.is_hand_shown):
            self.add_to_preflop_range(player.hand_stats, num_raises, action, player.hand_stats.player_cards_str)

    def add_to_pre_flop_open_actions(self, hand_stats, action):
        if hand_stats.usable_position in self.preflop_actions:
            self.preflop_actions[hand_stats.usable_position][action] += 1
        else:
            print("Error! No such position in preflop actions!")

    def add_to_pre_flop_vs_aggression_actions(self, hand_stats, action):
        if hand_stats.usable_position in self.preflop_actions:
           # if action == 'CHECK':
           #     pdb.set_trace()
            self.preflop_actions_vs_aggression[hand_stats.usable_position][action] += 1

        else:
            print("Error! something went wrong")

    def add_to_pre_flop_multi_aggression_actions(self, player_prev_actions, position, action, board_stats):
        if (action not in ACTIONS):
            print ("Error! Problem with action!")
            return        
        
        try:
            num_raises = board_stats.num_raises[PREFLOP]

            if player_prev_actions == [] and not self.is_heads_up_player:
                if num_raises == 2:
                    self.preflop_actions_in_cold_3bet_pot[position][action] += 1
                elif action != strings.FOLD:
                    print("Note! player is facing 4bet+ with no play - not folding")
                return;

            self.raises_to_actions_map[5 if num_raises > 5 else num_raises][position][action] += 1
            
        except ValueError:
            print("Error! Position not found in ALL_POSITIONS!")
            return
    
    # TODO - Not working in limped pots and in multi-player pots
    def calc_position(self, hand_stats, board_stats):
        post_flop_positions = HU_POSITIONS_POST_FLOP if self.is_heads_up_player else ALL_POSITIONS_POST_FLOP
        my_position_index = post_flop_positions.index(hand_stats.position)
        last_raiser = board_stats.last_raiser
        if last_raiser != None and last_raiser.hand_stats.position in post_flop_positions:
            last_raiser_position_index = post_flop_positions.index(last_raiser.hand_stats.position)
            position = self.IN_POSITION if my_position_index > last_raiser_position_index else self.OUT_OF_POSITION
            if position == self.IN_POSITION:
                hand_stats.ip = True
                last_raiser.hand_stats.ip = False
        else:
            if hand_stats.position == 'SB' or hand_stats.position == 'BB':
                position = self.OUT_OF_POSITION
            else:
                position = self.IN_POSITION
        
        return position

    def add_to_preflop_range(self, hand_stats, num_raises, action, player_cards):
        if num_raises == 0 and self.is_raise(action):
            self.raise_range[hand_stats.usable_position][player_cards] += 1
        elif num_raises == 1 and self.is_raise(action):
            if self.is_heads_up_player:
                three_bet_position = hand_stats.position 
            else:
                three_bet_position = 'BB' if hand_stats.position == 'BB' else 'other'
            self.three_bet_range[three_bet_position][player_cards] += 1
        elif num_raises == 2:
            if self.is_raise(action):
                self.four_bet_range[player_cards] += 1
            else:
                self.call_3bet_range[player_cards] += 1
        elif num_raises == 3:
            if self.is_raise(action):
                self.five_bet_range[player_cards] += 1
            else:
                self.call_4bet_range[player_cards] += 1
        elif num_raises >= 4:
            if self.is_raise(action):
                self.six_bet_range[player_cards] += 1
            else:
                self.call_5bet_range[player_cards] += 1

    def is_raise(self, action):
        return action == strings.RAISE
    
    def print_stats(self, f, zero_values_filter):
        self.calc_avgs()
        f.write("\nPREFLOP STATS:\n")
        f.write(f"  PFR: {self.pfr}%\n")

        f.write(f"  When checked to him:\n")
        for position in self.used_positions:
            sum_actions = sum(self.preflop_actions[position].values())
            f.write(f"  {position}: ")
            for action in [strings.RAISE, strings.CALL, strings.CHECK]:
                # TODO - What is that
                if action == 'CHECK' and position != 'BB':
                    print("Error! check should not be possible here")
                    continue
                elif action == 'CALL' and position == 'BB':
                    print("Error! call should not be possible here")
                    continue
                f.write(f"{action.capitalize() if action != strings.CALL else 'Limp'}: {self.preflop_actions[position][action]}")
                if sum_actions > 0:
                    f.write(f" ({round(float(self.preflop_actions[position][action] * 100) / sum_actions, 2)}%), ")
            f.write("\n")
        # Use the generic function for all VS Bet stats
        self.print_vs_bet_stats(f, 1)
        
        if not self.is_heads_up_player:
            f.write(f"  \nFacing 3Bet with no play:\n")
            positions = [self.IN_POSITION, self.OUT_OF_POSITION]
            for position in positions:
                sum_actions = sum(self.preflop_actions_in_cold_3bet_pot[position].values())
                if sum_actions > 0:
                    f.write(f"  {position.upper()}: ")
                    actions_list = list(self.preflop_actions_in_cold_3bet_pot[position].items())
                    for i, (action, count) in enumerate(actions_list):
                        percentage = round(float(count * 100 / sum_actions), 2)
                        f.write(f"{action.capitalize()}: {count} ({percentage}%)")
                        if i < len(actions_list) - 1:
                            f.write(f", ")
                    f.write(f"\n")

        for i in range(2, 5):
            self.print_vs_bet_stats(f, i)

        # Preflop Ranges
        f.write("\nPREFLOP RANGES:\n")
        for position in self.used_positions:
            f.write(f"  {position} Open Range: {zero_values_filter(self.raise_range[position])}\n")
        
        #f.write(f"  BB 3-bet Range: {zero_values_filter(self.three_bet_range['BB'])}\n")
        if not self.is_heads_up_player:
            f.write(f"  {"General" if not self.is_heads_up_player else "SB"} 3-bet Range: {zero_values_filter(self.three_bet_range['other' if not self.is_heads_up_player else 'SB'])}\n")
        
        f.write(f"  Call 3bet Range: {zero_values_filter(self.call_3bet_range)}\n")
        f.write(f"  4bet Range: {zero_values_filter(self.four_bet_range)}\n")
        f.write(f"  Call 4bet Range: {zero_values_filter(self.call_4bet_range)}\n")
        f.write(f"  5bet Range: {zero_values_filter(self.five_bet_range)}\n")
        f.write(f"  Call 5bet Range: {zero_values_filter(self.call_5bet_range)}\n")

    
    def print_vs_bet_stats(self, f, num_raises):
        """Generic function to print VS Bet statistics"""
        
        bet_name = ('' if num_raises == 1 else str(num_raises+1))+"Bet"
        re_raise_name = str(num_raises+2)+"Bet"

        f.write(f"  \nVS {bet_name}:\n")
        
        total_raises, total_actions = self.calc_re_raise_freq(num_raises)
        # Print next bet percentage if provided
        if total_actions > 0:    
            percentage = round(float(total_raises * 100) / total_actions, 2)
            f.write(f"  {re_raise_name}: {percentage}%\n")
        
        actions_dict = self.raises_to_actions_map[num_raises]

        # Print position stats
        
        for position in actions_dict.keys():
            sum_actions = sum(actions_dict[position].values())
            if sum_actions > 0:
                f.write(f"  {position.upper()}: ")
                actions_list = list(actions_dict[position].items())
                for i, (action, count) in enumerate(actions_list):
                    percentage = round(float(count * 100 / sum_actions), 2)
                    f.write(f"{action.capitalize()}: {count} ({percentage}%)")
                    if i < len(actions_list) - 1:
                        f.write(f", ")
                f.write(f"\n")

    def count_raises_in_actions(self, actions_dict):
        """Count total raises across all positions in the given actions dictionary"""
        total_raises = 0
        for position_actions in actions_dict.values():
            total_raises += position_actions.get(strings.RAISE, 0)
        return total_raises
    
    def calc_re_raise_freq(self, num_raises):    
        actions_dict = self.raises_to_actions_map[num_raises]
        total_raises = self.count_raises_in_actions(actions_dict)
        total_actions = sum(sum(position_actions.values()) for position_actions in actions_dict.values())
        return total_raises, total_actions