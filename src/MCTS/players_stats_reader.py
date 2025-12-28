import json, os
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, STREETS
import AppUtils.StringUtils as strings
import MCTS.players_stats_keys as actions

'''
This file is used by the mcts
player_stats.json data is about how the player a act in some situations:
SRP Pot, 3Bet Pot, when he's the pre-flop agressor, etc
we have 4 players types: hero, player, player_hu, hero_hu
Now in the mcts algorithm, to anticipate what the player will do - 
get the data from similar situations and combine them to create a probability distribution for the actions
'''

class Reader:
    MIN_HANDS_THRESHOLD = 50 # Minimum number of hands required to use a dataset
    CATEGORY_THRESHOLD = 20 # Minimum number of hands required for a category to affect the probability
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - ensures only one instance is created"""
        if cls._instance is None:
            cls._instance = super(Reader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self): 
        # Only initialize if not already initialized
        if hasattr(self, '_initialized'):
            return

        self.load_json_file() # load the player_stats.json file into self.player_stats
        self.initialize_players_stats() # create a mapping for each player into self.players
        self._initialized = True
    
    def load_json_file(self):
        """Load player statistics from single JSON file"""
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        stats_path = os.path.join(base_dir, 'data', 'player_stats.json')
        
        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.player_stats = json.load(f)
                print(f"Loaded player stats from {stats_path}")
        except FileNotFoundError:
            print(f"Warning: player_stats.json not found at {stats_path}")
            self.player_stats = {}
        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing player_stats.json: {e}")
            self.player_stats = {}
    
    """Create mapping for each player type (player, player_hu, hero, hero_hu)"""
    def initialize_players_stats(self):
        # Map action keys to actual JSON keys
        player_key_mapping = {
            actions.GENERAL_PLAYER: 'player_5',
            actions.GENERAL_PLAYER_HU: 'player_hu_5',
            actions.HERO: 'Hero_5',
            actions.HERO_HU: 'Hero_hu_5'
        }
        
        self.players = {}
        for action_key, json_key in player_key_mapping.items():
            if json_key in self.player_stats:
                self.players[action_key] = self.player_stats[json_key]
            else:
                self.players[action_key] = {}
                print(f"Warning: Player {json_key} not found in JSON file")
    
    def get_hand_categories(self, player, game):
        vs_action_type = actions.get_action_key(game.get_num_raises_in_current_street())
        pot_type_by_raises = game.get_num_raises_preflop()
        is_multi_player_pot = game.num_of_players_in_streets[game.street] > 2
        is_ip = player.is_ip(game.get_positions_by_order(), game.get_active_players_in_hand())
        street_key = actions.get_street_key(game.street)

        is_fa = player.is_flop_agressor
        is_ta = player.is_turn_agressor

        return vs_action_type, pot_type_by_raises, is_multi_player_pot, is_ip, street_key, is_fa, is_ta

    def initialize_street_data(self, is_hero, is_hu_game, street_key):
        """
        Get data from JSON file using a specific key combination
        """
        # Build player name
        if is_hu_game:
            player_name = actions.HERO_HU if is_hero else actions.GENERAL_PLAYER_HU
        else:
            player_name = actions.HERO if is_hero else actions.GENERAL_PLAYER
        
        # Check if data exists
        if player_name not in self.players:
            print(f"Player {player_name} not found in players mapping")
            return None
        
        if street_key not in self.players[player_name]:
            print(f"Street {street_key} not found for player {player_name}")
            return None
        
        player_street_data = self.players[player_name][street_key]
        return player_street_data

    def get_categories_keys(self, street, is_pfr, raises_preflop, is_multi_player_pot, is_ip, is_fa, is_ta):
        # Build key parts based on actual parameter values
        pfr_key = actions.PRE_FLOP_RAISER if is_pfr else actions.PRE_FLOP_CALLER
        pot_type_key = actions.SRP if raises_preflop <= 1 else actions.THREE_BET if raises_preflop == 2 else actions.FOUR_BET
        pot_key = actions.TWO_PLAYERS_POT if not is_multi_player_pot else actions.MULTIWAY_POT
        ip_key = actions.IP if is_ip else actions.OOP
        
        categories = [pot_key, pot_type_key, pfr_key, ip_key]
        
        if street >= TURN:
            categories.append(actions.FLOP_AGGRESSOR if is_fa else actions.FLOP_PASSIVE)
        if street == RIVER:
            categories.append(actions.TURN_AGGRESSOR if is_ta else actions.TURN_PASSIVE)
        return categories

    # Main function - predict the action category based on the game history
    def predict_action_category(self, player, game, action, size=0):
        if action == strings.FOLD:
            return {}
        
        action_type, pot_type_by_raises, is_multi_player_pot, is_ip, street_key, is_fa, is_ta = self.get_hand_categories(player, game)
        
        bet_size = None
        if action == strings.RAISE:
            if action_type == actions.VS_BET:
                # NOTE: in this case the pot is already increased by the last bet size
                pot_size = game.pot_size - game.last_bet_size
                from AppUtils.actions_utils import get_bet_size_category
                bet_size = get_bet_size_category(game.last_bet_size/pot_size)
                if bet_size is not None:
                    bet_size += '_split'
            elif action_type == actions.VS_RAISE:
                bet_size = 'raise' + '_split'
            else:
                bet_size = None

        # Get all relevant categories for this game state
        relevant_categories = self.get_categories_keys(game.street, player.is_pre_flop_agressor, pot_type_by_raises, is_multi_player_pot, is_ip, is_fa, is_ta)
        
        category_data_map = {}  # Map: category -> {action: count}
        category_size_data_map = {}

        for category in relevant_categories:
            data, size_data = self.get_river_categories_freq_data(player.is_main_player, game.is_heads_up, street_key, category, action, bet_size)
            if data is not {}:
                category_data_map[category] = data
            if size_data is not {}:
                category_size_data_map[category] = size_data
        
        # Also get data from all_data
        all_data, size_data = self.get_river_categories_freq_data(player.is_main_player, game.is_heads_up, street_key, 'all_data', action, bet_size)
        if all_data is not {}:
            category_data_map['all_data'] = all_data
        if size_data is not {}:
            category_size_data_map['data'] = size_data
        
        # Send all category data to probability calculation function
        if category_data_map:
            general_probabilities = self.calculate_action_probabilities(category_data_map, category_size_data_map)
        
        return general_probabilities
    
    # calculate the frequency of each category for a given action by previous data in the json file
    def get_river_categories_freq_data(self, is_hero, is_hu_game, street_key, key_combo, action, size_key):
        player_street_data = self.initialize_street_data(is_hero, is_hu_game, street_key)
        summary_data = {}
        size_summary_data = {}
        if player_street_data is None:
            return summary_data, summary_data
        
        
        river_action_categories = 'river_action_categories'
        if key_combo in player_street_data and river_action_categories in player_street_data[key_combo]:
            action_categories = player_street_data[key_combo][river_action_categories]
            if action in action_categories:
                action_category = action_categories[action]
                
                for action_category, count in action_category.items():
                    summary_data[action_category] = summary_data.get(action_category, 0) + count
        if size_key is not None and key_combo in player_street_data and size_key in player_street_data[key_combo]:
            action_categories = player_street_data[key_combo][size_key]
            for action_category, count in action_categories.items():
                size_summary_data[action_category] = size_summary_data.get(action_category, 0) + count
            
        return summary_data, size_summary_data

    def calculate_action_probabilities(self, category_data_map, category_size_data_map = None):
        """
        Calculate action probabilities using Odds Ratios combination method
        Algorithm (based on Logistic Regression approach):
        """
        if not category_data_map or 'all_data' not in category_data_map:
            return {}
        
        # Step 1: Calculate P_Baseline from all_data
        all_data = category_data_map['all_data']
        if all_data is None:
            return {}
        
        total_hands = sum(all_data.values())
        base_probabilities = self.calculate_probabilities_from_counts(all_data)

        # Step 2: For each action, calculate final probability using Odds Ratios
        final_probabilities = {}
        all_actions = all_data.keys()

        for action in all_actions:
            P_baseline = base_probabilities[action]
            
            # Calculate Odds_Baseline
            if P_baseline <= 0:  
                final_probabilities[action] = 0
                continue
            
            # Step 3: Collect P_conditions from all categories
            P_conditions = []  # List of P_X for each category
            total_hands_in_category = []
            
            self.calculate_P_conditions(category_data_map, action, total_hands_in_category, P_conditions)
            if category_size_data_map != None:
                self.calculate_P_conditions(category_size_data_map, action, total_hands_in_category, P_conditions)
            
            # Step 4: Calculate combined probability using the formula
            P_Combined = self.calculate_combined_probability(P_baseline, P_conditions, total_hands, total_hands_in_category)
            
            final_probabilities[action] = P_Combined
        
        # Normalize probabilities to sum to 1
        total_prob = sum(final_probabilities.values())
        if total_prob > 0:
            normalized_probabilities = {}
            for action, prob in final_probabilities.items():
                normalized_probabilities[action] = prob / total_prob
            return normalized_probabilities
        
        return final_probabilities
    
    def calculate_P_conditions(self, category_data_map, action, total_hands_in_category, P_conditions):
        for category_name, category_data in category_data_map.items():
            if category_name == 'all_data':
                continue
            
            # Check if category has this action and enough hands
            if action in category_data:
                action_count = category_data[action]
                total_category_hands = sum(category_data.values())
                
                P_X = action_count / total_category_hands
                if P_X >= 0 and P_X <= 1:  # Only add valid probabilities
                    total_hands_in_category.append(total_category_hands)
                    P_conditions.append(P_X)
    
    def calculate_combined_probability(self, P_base, list_of_P_conditions, total_hands, list_of_hands_in_category):
        """
        Calculate combined probability using Odds Ratios multiplication
        """
        # Handle edge cases
        if P_base <= 0 or P_base >= 1:
            # If base is invalid, return 0 or use average of conditions
            if list_of_P_conditions:
                return sum(list_of_P_conditions) / len(list_of_P_conditions)
            return 0.0
        
        if not list_of_P_conditions:
            # If no conditions, return base probability
            return P_base
        
        # Step 1: Convert P_base to Odds_Base
        Odds_Base = P_base / (1 - P_base)
        
        # Step 2: Calculate Odds Ratios for each condition
        OR_combined = 1.0  # Start with 1 for multiplication
        
        for P_condition, hand_in_category in zip(list_of_P_conditions, list_of_hands_in_category):
            # Skip invalid probabilities
            if P_condition <= 0 or P_condition >= 1:
                continue
            
            # Calculate Odds_condition = P_condition / (1 - P_condition)
            Odds_condition = P_condition / (1 - P_condition)
            
            # Calculate Odds Ratio: OR = Odds_condition / Odds_Base
            OR = Odds_condition / Odds_Base
            ALPHA = hand_in_category / total_hands
            # Multiply to get combined OR
            OR_combined *= (OR ** ALPHA)
        
        # Step 3: Combine Odds
        Odds_Combined = Odds_Base * OR_combined
        
        # Step 4: Convert back to probability
        # P_Combined = Odds_Combined / (1 + Odds_Combined)
        P_Combined = Odds_Combined / (1 + Odds_Combined)
        
        return P_Combined

    def calculate_probabilities_from_counts(self, action_counts):
        """Convert action counts to probabilities"""
        total = sum(action_counts.values())
        if total == 0:
            return {}
        
        probabilities = {}
        for action, count in action_counts.items():
            probabilities[action] = count / total
        
        return probabilities

    # this is a simple average of the size probabilities and the average probabilities
    def combine_probabilities(self, general_probabilities, size_probabilities):
        """Combine general and size probabilities"""
        combined_probabilities = {}
        for action in general_probabilities.items():
            combined_probabilities[action] = (general_probabilities[action] + size_probabilities[action])/2
        return combined_probabilities
    
    def categories_to_string(self, categories_probs, action):
        categories_string = "Your " + action.capitalize() + " range: \n"
        for category, prob in categories_probs.items():
            categories_string += f"{category.capitalize().replace('_', ' ')}: {round(prob*100, 2)}%\n"
        return categories_string

    
    def find_hand_type_action_prob_in_category(self, game, player, action):
        category = self.is_categorized_action(game, player, action)
        if category is not None:
            # find the category and return a map of hand-category -> probability to take this action
            pass;
        return None

    def is_categorized_action(self, game, player, action):
        category = None
        if game.street == FLOP:
            category = self.is_flop_categorized_action(game, player, action)
        elif game.street == TURN:
            category = self.is_turn_categorized_action(game, player, action)
        
        return category

    def is_flop_categorized_action(self, game, player, action):
        # Flop categorized actions: srp_raise, srp_overbet, mbp_raise, mbp_overbet
        raises_preflop = game.get_num_raises_preflop()
        hand_type = 'srp' if raises_preflop <= 1 else 'mbp'
        # Only check for RAISE actions (bets/raises)
        if action != strings.RAISE:
            return False
        # Check if this is a raise action
        is_raise = action == strings.RAISE and game.last_bet_size > 0

        is_overbet = False;
        if not is_raise:
            # Check if overbet (bet size >= pot size)
            bet_size_ratio = game.last_bet_size / game.pot_size if game.pot_size > 0 else 0
            from AppUtils.actions_utils import get_bet_size_category
            is_overbet = get_bet_size_category(bet_size_ratio) == 'overbet'
        
        if is_raise or is_overbet:
            return hand_type + '_' + ('raise' if is_raise else 'overbet')
        return None
        
    def is_turn_categorized_action(self, game, player, action):
        # Turn categorized actions: srp_pfr_ip_bet_bet, srp_pfr_oop_bet_bet, srp_overbet, srp_raise, mbp_pfr_ip_bet_bet, mbp_pfr_oop_bet_bet, mbp_overbet, mbp_raise
        raises_preflop = game.get_num_raises_preflop()
        hand_type = 'srp' if raises_preflop <= 1 else 'mbp'
        
        # Only check for RAISE actions (bets/raises)
        if action != strings.RAISE:
            return None
        
        # Check if this is a raise action
        is_raise = game.last_bet_size > 0
        is_overbet = False;
        if not is_raise:
            # Check if overbet (bet size >= pot size)
            bet_size_ratio = game.last_bet_size / game.pot_size if game.pot_size > 0 else 0
            from AppUtils.actions_utils import get_bet_size_category
            is_overbet = get_bet_size_category(bet_size_ratio) == 'overbet'
        
        if is_raise or is_overbet:
            return hand_type + '_' + ('raise' if is_raise else 'overbet')
        
        if player.is_pre_flop_agressor and player.is_flop_agressor:
            if player.is_ip(game.get_positions_by_order(), game.get_active_players_in_hand()):
                return hand_type + '_' + 'pfr_ip_bet_bet'
            else:
                return hand_type + '_' + 'pfr_oop_bet_bet'
        
        return None 
    '''
    # this function predict the player's  action based on the game history
    def predict_action_by_freq(self, player, game):
        """Main function with backoff mechanism to get action frequencies"""
        
        vs_action_type, pot_type_by_raises, is_multi_player_pot, is_ip, street_key, is_fa, is_ta = self.get_hand_categories(player, game)
        
        # Get all relevant categories for this game state
        relevant_categories = self.get_categories_keys(game.street, player.is_pre_flop_agressor, pot_type_by_raises, is_multi_player_pot, is_ip, is_fa, is_ta)
        category_data_map = {}  # Map: category -> {action: count}
        category_size_data_map = {}

        # get the bet size category - TODO - this data should be sent
        bet_size = None
        if vs_action_type == actions.VS_BET:
            # NOTE: in this case the pot is already increased by the last bet size
            pot_size = game.pot_size - game.last_bet_size
            bet_size = get_bet_size_category(game.last_bet_size/pot_size)
        
        for category in relevant_categories:
            data, size_data = self.get_player_actions_freq_data(player.is_main_player, game.is_heads_up, street_key, category, vs_action_type, bet_size)
            if data is not None:
                category_data_map[category] = data
            if size_data is not None:
                category_size_data_map[category] = size_data

        # Also get data from all_data
        all_data, size_data = self.get_player_actions_freq_data(player.is_main_player, game.is_heads_up, street_key, 'all_data', vs_action_type, bet_size)
        if all_data is not None:
            category_data_map['all_data'] = all_data
        
        # Send all category data to probability calculation function
        if category_data_map:
            general_probabilities = self.calculate_action_probabilities(category_data_map, category_size_data_map)
        
        return  general_probabilities

    def get_player_actions_freq_data(self, is_hero, is_hu_game, street_key, key_combo, vs_action_type, size_category = None):
        player_street_data = self.initialize_street_data(is_hero, is_hu_game, street_key)
        if player_street_data is None:
            return None, None

        # JSON Structure: player -> street -> by_category -> key -> vs_action_type -> {action: count}
        summary_data = {}
        summary_size_data = {}
        
        if key_combo in player_street_data:
            combo_data = player_street_data[key_combo]
            if size_category is not None and vs_action_type == actions.VS_BET:
                vs_bet_sizes_key = actions.VS_BET_SIZES
                if isinstance(combo_data, dict) and vs_bet_sizes_key in combo_data and size_category in combo_data[vs_bet_sizes_key]:
                    for action, count in combo_data[vs_bet_sizes_key][size_category].items():
                        summary_size_data[action] = summary_size_data.get(action, 0) + count
            if isinstance(combo_data, dict) and vs_action_type in combo_data:
                for action, count in combo_data[vs_action_type].items():
                    summary_data[action] = summary_data.get(action, 0) + count
        
        return (summary_data if summary_data else None), (summary_size_data if summary_size_data else None)
    
    def bet_classifier_get_action_freq_with_backoff(self, hand_stats, board_stats, street, is_main_player):
        vs_action_type = actions.get_action_key(board_stats.num_raises[street]) 
        pot_type_by_raises = board_stats.num_raises[PREFLOP]
        is_multi_player_pot = board_stats.num_players
        is_ip = hand_stats.ip
        street_key = actions.get_street_key(street)
        is_fa = hand_stats.fr
        is_ta = hand_stats.tr

        # Get all relevant categories for this game state
        relevant_categories = self.get_relevant_categories_keys(street, hand_stats.pfr, pot_type_by_raises, is_multi_player_pot, is_ip, is_fa, is_ta)
        
        category_data_map = {}  # Map: category -> {action: count}
        
        for category in relevant_categories:
            data = self.get_player_data_from_json_file(is_main_player, board_stats.num_players_pre_flop == 2, street_key, category, vs_action_type)
            if data is not None:
                category_data_map[category] = data
        
        # Also get data from all_data
        all_data = self.get_player_data_from_json_file(is_main_player, board_stats.num_players_pre_flop == 2, street_key, 'all_data', vs_action_type)
        if all_data is not None:
            category_data_map['all_data'] = all_data
        
        # Send all category data to probability calculation function
        if category_data_map:
            return self.calculate_action_probabilities(category_data_map)
        
        # No data found - return empty dict
        print("No sufficient data found for any category")
        return {}

    
    def action_counts_to_freq(self, action_counts):
        """Convert action counts dictionary to frequency distribution"""
        if action_counts is None or not action_counts:
            return None
        
        total_hands = sum(action_counts.values())
        
        if total_hands == 0:
            return None
            
        action_freq = {}
        for action, count in action_counts.items():
            action_freq[action] = count / total_hands
            
        return action_freq
    '''