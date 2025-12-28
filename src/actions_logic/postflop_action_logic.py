import random
import AppUtils.StringUtils as strings
from agent.agents_registry import get_hero_agents, get_villain_agents, get_mcts
from agent.parse_data_to_agent import parse_live_play_features
# Handles all agent-related decision making logic
class AgentActions:
    def __init__(self):
        # Use agents and MCTS from the registry (initialized once globally)
        self.supervised_agents = get_hero_agents()
        self.villain_supervised_agents = get_villain_agents()
        self.MCTS_model = get_mcts()

    def mcts_act(self, game, player):
        if not self.MCTS_model:
            print("MCTS is not initialized")
            return False
        
        best_action = None
        try:
            best_action, bet_amount = self.MCTS_model.find_best_action(game, player)
        except Exception as e:
            print(f"MCTS failed: {e}")
            return False
        
        if best_action is None:
            print("MCTS returned no action, using supervised agent as fallback")
            # try to use the supervised agent to find the best action if mcts failed
            state_features = parse_live_play_features(game, player, False)
            from agent.agent_utils import mask_possible_actions_in_live_play
            action_mask = mask_possible_actions_in_live_play(game, player)
            action_index, _ = self.supervised_agents[game.street].predict_action(state_features, action_mask)
            
            from agent.agent_utils import INT_VALUE_TO_STRING_ACTION_MAP as action_map
            best_action = action_map.get(action_index)
            if best_action is None:
                print("Supervised agent returned no action")
                return False

        if best_action == strings.RAISE:
            if bet_amount == 0:
                if game.get_num_raises_in_current_street() > 0:
                    bet_amount = game.pot_size * 1.5;
                else:
                    bet_amount = self.find_bet_amount(game.pot_size, game.get_num_raises_preflop())
        
        elif best_action == strings.CALL:
            bet_amount = game.last_bet_size
        else:
            bet_amount = 0
        
        print(f"MCTS decided to {best_action}")
        return Action(best_action, 100, bet_amount)
                
    def find_bet_amount(self, pot_size, num_raises_preflop):
        srp_raise_size_probabilities = {0.33: 17.9, 0.5: 7.2, 0.75: 5, 1.5: 3.6}
        three_bet_size_probabilities = {0.3: 29.2, 0.5: 13.4, 0.75: 5.2, 1.5: 0.1}
        size_probabilities = srp_raise_size_probabilities if num_raises_preflop < 2 else three_bet_size_probabilities
        bet_size = random.choices(list(size_probabilities.keys()),
        weights=[size_probabilities[size] for size in size_probabilities.keys()])[0]
        bet_amount = pot_size * bet_size
        return bet_amount

class Action:
    def __init__(self, action, freq, amount): 
        self.action = action
        self.freq = freq
        self.amount = amount
    
    def __str__(self):
        return f"{self.action} ({self.freq}%) {self.amount}"