import torch
import random
import copy
from typing import Dict, List, Tuple, Optional
from collections import Counter
from itertools import combinations
import math

# Import poker-specific modules
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, ALL_POSITIONS_POST_FLOP, STREETS
import pdb
import AppUtils.StringUtils as strings
from AppUtils.agent_utils import action_map


# 1. for each player (if not hero):
# c. choose action by neural network
# d. create new node with the new game state, and move to the next player
# e. check if we are in showdown/moving to the next street - if yes, end it if too much moves
# f. evaluate the player hand against the other players hands and add the value to the node
# g. backpropagate the value to the root node

# TODO:
# 1. build range of cards for each player
# 2. add bet sizes
# 3. update the agent accordingly to the MCTS

k = 1.0
alpha = 0.5

def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    Uses PUCT (Predictor + UCT) formula from AlphaZero.
    """
    
    if child.visit_count == 0:
        return float('inf')  # Prioritize unvisited children
    
    # Q(s,a) - Average value (exploitation)
    
    exploitation = child_expected_value_for_parent(parent, child)    
    
    # U(s,a) - Exploration bonus (UCB1 style - without prior influence)
    # U(s,a) = c * sqrt(ln(N(s)) / N(s,a))
    # This ensures that less-visited children get higher exploration bonus

    c_puct = 2.5 if parent.game.street == FLOP else 1.4 # exploration constant
    exploration = c_puct * math.sqrt(math.log(parent.visit_count + 1) / child.visit_count)
    
    if parent.parent is None:
        print('value:', exploitation, 'exploration:', exploration, 'action:', child.action, 'visit count:', child.visit_count)
    ucb_score = exploitation + exploration
    return ucb_score

def child_expected_value_for_parent(parent, child):
        parent_position = parent.player.position if parent.player else 0
        if child.visit_count == 0 or parent_position not in child.value_sums:
            return 0.0
        value = child.value_sums[parent_position] / child.visit_count
        return value / parent.game.pot_size

class MCTSNode:
    def __init__(self, prior_prob, state = None, game = None, player = None, moves_left = 5, is_terminal = False, action = None, parent = None):
        self.visit_count = 0 # The number of times this node has been visited
        self.player_index = player.position_index if player is not None else -1 # The player index whose turn it is to act 
        self.children = {} # A look up pf legal child nodes, basically a map of action -> the node it creates
        # self.children_values = {} # A look up of the value of the child nodes

        ''' game properties '''
        self.state = state # game state for this node
        self.game = game # game for this node
        self.player = player # player for this node
        
        self.prior_prob = prior_prob  # The prior probability to selecting this state from it's parent
        self.moves_left = moves_left  # this the depth we can go down in the tree
        self.is_terminal = self.is_terminal_state();

        self.action = action # the action that led to this node
        self.parent = parent # the parent node

        ''' value properties '''
        self.value_sums = {} # The total value of this node for each player from all visits

    def has_children(self):
        return len(self.children) > 0

    def is_terminal_state(self):
        if self.game is None:
            pdb.set_trace()
            return True
        return self.game.is_over()

    def select_child(self):
        """Select child using UCB1 formula"""
        # Collect all children with their scores
        children_scores = [(action, child, ucb_score(self, child)) for action, child in self.children.items()]
        
        # Find max score
        max_score = max(score for _, _, score in children_scores)
        
        # Get all children with max score
        best_children = [(action, child) for action, child, score in children_scores if score == max_score]
        
        # If multiple with same score, choose by prior (or randomly if same prior)
        if len(best_children) > 1:
            best_action, best_child = max(best_children, key=lambda x: x[1].prior_prob)
        else:
            best_action, best_child = best_children[0]

        return best_child, best_action

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior_prob)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())

class MCTS:
    def __init__(self, models):
        self.models = models
        self.num_simulations =  100
        self.root_game = None

    def find_best_action(self, state, game, player):
        players_to_act = game.get_players_to_act(True)
        if not players_to_act:
            print("Error: No players to act")
            pdb.set_trace()
            return None
        self.reset_mcts_chips_in_pot(game)
        root = MCTSNode(1, state, game, player)
        self.root_game = game;
        relevant_positions = [player.position for player in game.get_players_in_hand()]

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # === 1. SELECTION ===
            # go through the tree until we reach a node with no children or a terminal node
            index = 0
            while node.has_children() and not node.is_terminal:
                node, action = node.select_child()
                '''
                print("depth:", index, 'street:', STREETS[node.game.street], node.player.name, 'action:', node.action, ', left in his stack: ', node.player.stack_size)
                '''
                search_path.append(node)
                index += 1
            

            # === 2. EXPANSION ===
            if not node.is_terminal:
                if node.state is None:
                    break
                action_dict = self.supervised_agent_act(node.state, self.models[node.game.street])
                valid_moves = self.mask_invalid_moves(action_dict, node.player, node.game, node.moves_left)

                for action, prob in valid_moves.items():
                    if action not in node.children:
                        game_copy = copy.deepcopy(node.game)
                        game_copy, next_player = self.simulate_action(action, game_copy, game_copy.get_player_by_name(node.player.name))
                        
                        if next_player is None:
                            print("showdown..")
                        child_state = None if next_player is None else self.create_child_state(game_copy, next_player)
                        
                        node.children[action] = MCTSNode(
                            prior_prob=prob,
                            state=child_state,
                            game=game_copy,
                            player=next_player,
                            moves_left=node.moves_left,
                            action=action,
                            parent=node
                        )
            
            for action, child in root.children.items():
                avg_q = child.value_sums.get(root.player.position, 0.0) / max(1, child.visit_count)
                # אם מנרמלים לפי pot:
                q_norm = avg_q / max(1.0, root.game.pot_size)
                exploration = 1.4 * child.prior_prob * math.sqrt(root.visit_count) / (1 + child.visit_count)
                '''
                print(action, "visits=", child.visit_count,
                    "avg_q=", avg_q, "q_norm=", q_norm,
                    "prior=", child.prior_prob,
                    "exploration=", exploration,
                    "ucb=", q_norm + exploration)
                '''
            # === 3. SIMULATION ===
            ev_map = self.rollout(node, relevant_positions)

            # === 4. BACKPROPAGATION ===
            self.backpropagate(search_path, ev_map)

        # בסוף, נחזיר את הפעולה עם הכי הרבה ביקורים
        if not root.children:
            print("Error: No children found in root node")
            pdb.set_trace()
            return None
        
        best_score = -float('inf')
        best_action = None
        for action, child in root.children.items():
            print(f"action: {action}, value: {child_expected_value_for_parent(root, child)}, num of visits: {child.visit_count}")
            if child_expected_value_for_parent(root, child) > best_score:
                best_score = child_expected_value_for_parent(root, child)
                best_action = action
        return best_action

    def rollout(self, node, relevant_positions):
        game_copy = copy.deepcopy(node.game)
        while (not game_copy.is_over()):
            players = game_copy.get_players_to_act(True)
            player_to_act = players[0]
            state = self.create_child_state(game_copy, player_to_act)
            if state is None:
                pdb.set_trace()
                break
            action_dict = self.supervised_agent_act(state, self.models[game_copy.street])

            valid_moves = self.mask_invalid_moves(action_dict, player_to_act, game_copy, node.moves_left)
            action = random.choices(list(valid_moves.keys()),
                                    weights=[valid_moves[action] for action in valid_moves.keys()])[0]
            
            game_copy, player_to_act = self.simulate_action(action, game_copy, player_to_act)
            
            #if player_to_act is None and game_copy.street < RIVER:
            #    game_copy = game_copy.get_ready_for_next_street(game_copy.street)

        return self.calc_value(relevant_positions, game_copy)

    def calc_value(self, relevant_positions, game):
        ev_map = {}
        if len(game.get_players_in_hand()) == 1:
            winners = game.get_players_in_hand()
        else:
            winners, winning_hands = game.deck.find_winners(game.get_players_in_hand(), game.community_cards)

        for player in game.players:
            if player.position in relevant_positions:
                ev_map[player.position] = -player.mcts_chips_in_pot
                for winner in winners:
                    if player.position == winner.position:
                        ev_map[player.position] += (game.pot_size / len(winners))
        
        return ev_map

    def backpropagate(self, search_path, ev_map):
        for node in reversed(search_path):
            for position, ev in ev_map.items():
                if position not in node.value_sums:
                    node.value_sums[position] = 0
                node.value_sums[position] += max(min(2.5*self.root_game.pot_size, ev), -2.5*self.root_game.pot_size)  # TODO - this is to normalize actions values
            node.visit_count += 1

    
    def supervised_agent_act(self, game_state, agent):
        """
        Use the supervised poker agent to make decisions.
        Args: game_state: State object containing all game information
        Returns: (action_class, confidence)
        action_class: 0 (fold), 1 (check), 2 (call), 3 (raise)
        """
        try:
            import agent.data_preparation as processor
            data_processor = processor.PokerDataProcessor()
            # Check if data processor is available
            if data_processor is None:
                print("Data processor not initialized, cannot use supervised agent")
                return {}
            
            # Convert State object to dictionary for data processor
            game_state_data = game_state.to_dict() # TODO - problem with encription of position
            
            # Encode game state using the data processor
            features_tensor = data_processor.encode_game_state(game_state_data)
            
            # Get prediction from supervised agent for the current street
            with torch.no_grad():
                action_logits = agent(features_tensor.unsqueeze(0))
                
                # Apply softmax to get probabilities (masked actions will have 0 probability)
                action_probs = torch.softmax(action_logits, dim=1)
                '''
                predicted_action = torch.argmax(action_probs, dim=1).item()
                confidence = action_probs[0][predicted_action].item()
                '''
                # Convert to action name -> probability mapping
                action_prob_dict = {}
                
                for i, prob in enumerate(action_probs[0]):
                    action_name = action_map.get(i, f"UNKNOWN_{i}")
                    action_prob_dict[action_name] = round(float(prob.item()), 5)
                
                return action_prob_dict
            
        except Exception as e:
            print(f"Supervised agent prediction failed: {e}")
            return {}
    
    def mask_invalid_moves(self, action_dict, player, game, moves_left):
        agent_valid_moves = {}
        valid_moves = player.get_possible_actions(game)
        
        for action in action_dict.keys():
            if valid_moves is not None and action in valid_moves:
                agent_valid_moves[action] = action_dict[action]
            
        return agent_valid_moves

    def simulate_action(self, action, game, game_player):
        amount = self.calc_amount(action, game, game_player)
        game = game.act_after_mcts_choice(game_player, action, amount)
        next_players = game.get_players_to_act(True)

        next_player = self.get_next_player(game, next_players)
            
        return game, next_player

    def get_next_player(self, game, next_players):
        if len(next_players) == 0:
            next_player = None
            if not game.is_over():
                game.get_ready_for_next_street(game.street)
                next_players = game.get_players_to_act(True)
                next_player = next_players[0]
            else:
                next_player = None
        else:
            next_player = next_players[0]
        return next_player
        
    def create_child_state(self, game_simulation, next_player):
        from MCTS.agent_actions import State
        return State(game_simulation, next_player)

    def calc_amount(self, action, game, player):
        # Calculate amount based on action type
        if action == strings.CALL:
            return game.last_bet_size
        elif action == strings.RAISE:
            num_raises = game.get_num_raises_in_current_street();
            bet_amount = game.pot_size * (0.55 if num_raises == 0 else 2.2)
            # Use a standard raise size (e.g., 2.5x the current bet or pot size)
            return bet_amount
        return 0;
    

    def reset_mcts_chips_in_pot(self, game):
        for player in game.get_players_in_hand():
            player.mcts_chips_in_pot = 0






















    '''
    OLD CODE
    '''

    '''
    def calc_ev_map_simulation(self, players, game):
        players_names = [player.name.replace(' ', '_') for player in players]
        players_names.sort()
        key = '_'.join(players_names)
        
        if self.simulation_value_map.get(key) is None:
            winner_count = {}
            # Initialize winner_count for all players
            for player in (players):
                winner_count[player.name] = 0
            
            num_simulations = 20
            for _ in range(num_simulations):
                community_cards = copy.deepcopy(game.community_cards)
                deck = copy.deepcopy(game.deck)
                random.shuffle(deck.cards)
                community_cards.extend(deck.deal_next_cards_for_mcts(game.street))
                winner, winning_hand = deck.find_winner(game.get_players_in_hand(), community_cards)
                winner_count[winner.name] += 1

            # Initialize the dictionary for this key
            self.simulation_value_map[key] = {}
            for player in (players):
                self.simulation_value_map[key][player.name] = (winner_count[player.name]/num_simulations)

        return self.simulation_value_map[key];

    def save_game_state(self, game):
        """Save the current state of the game for restoration later"""
        saved_state = {
            # Game-level state
            'pot_size': game.pot_size,
            'street': game.street,
            'last_bet_size': game.last_bet_size,
            'community_cards': game.community_cards.copy() if game.community_cards else [],
            'current_hand_actions_strength': game.current_hand_actions_strength.copy(),
            'prev_actions_in_street': game.prev_actions_in_street.copy(),
            
            # Player states
            'players': []
        }
        
        # Save each player's state
        for player in game.players:
            player_state = {
                'index': player.index,
                'stack_size': player.stack_size,
                'chips_in_pot': player.chips_in_pot,
                'mcts_chips_in_pot': getattr(player, 'mcts_chips_in_pot', 0),
                'is_folded': player.is_folded,
                'is_all_in': player.is_all_in,
                'is_need_to_act': player.is_need_to_act,
                'is_pre_flop_agressor': player.is_pre_flop_agressor,
                'is_last_agressor': player.is_last_agressor,
                'actions': {street: actions.copy() for street, actions in player.actions.items()}
            }
            saved_state['players'].append(player_state)
        
        return saved_state
    '''
    '''
    def restore_game_state(self, game, saved_state):
        """Restore the game to the previously saved state"""
        # Restore game-level state
        game.pot_size = saved_state['pot_size']
        game.street = saved_state['street']
        game.last_bet_size = saved_state['last_bet_size']
        game.community_cards = saved_state['community_cards'].copy()
        game.current_hand_actions_strength = saved_state['current_hand_actions_strength'].copy()
        game.prev_actions_in_street = saved_state['prev_actions_in_street'].copy()
        
        # Restore each player's state
        for i, player_state in enumerate(saved_state['players']):
            if i < len(game.players):
                player = game.players[i]
                player.stack_size = player_state['stack_size']
                player.chips_in_pot = player_state['chips_in_pot']
                player.mcts_chips_in_pot = player_state['mcts_chips_in_pot']
                player.is_folded = player_state['is_folded']
                player.is_all_in = player_state['is_all_in']
                player.is_need_to_act = player_state['is_need_to_act']
                player.is_pre_flop_agressor = player_state['is_pre_flop_agressor']
                player.is_last_agressor = player_state['is_last_agressor']
                player.actions = {street: actions.copy() for street, actions in player_state['actions'].items()}
        
        return game
    '''

    '''
    def find_ev_map(self, game):
        ev_map = {}
        if len(game.get_players_in_hand()) == 1:
            winner = game.get_players_in_hand()[0]
            ev_map[winner.name] = game.pot_size - winner.mcts_chips_in_pot
        else:
            winner, winning_hand = game.deck.find_winner(game.get_players_in_hand(), game.community_cards)
            ev_map[winner.name] = game.pot_size - winner.mcts_chips_in_pot

    def backpropagate(self, search_path, ev_map):
        for node in reversed(search_path):
            if node.player is not None and node.player.name in ev_map:
                node.value_sum += ev_map[node.player.name]

            if node.parent is not None and node.parent.player is not None and node.parent.player.name in ev_map:
                node.parent.children_values[node.action] += ev_map[node.parent.player.name]
            node.visit_count += 1
 
    def find_best_action(self, state, game, player, moves_left = 12, prob = 1):
        moves_left = round(9/len(game.get_players_in_hand()), 0)
        root = MCTSNode(prob, state, game, player, moves_left)
        print("main method - ", player.name , "is searching for best action")
        self.find_best_action_recursive(root)
        print("finished finding best action")
        print("children visit count: ", [child.visit_count for child in root.children.values()])
        print("children values: ", [root.children_values[action] for action in root.children.keys()])
        
        best_action = None;
        best_value = -float('inf');
        print("final decision!")
        for action, child_node in root.children.items():
            print("optional action: ", action, "value: ", child_node.value())
            if child_node.value() > best_value:
                best_value = child_node.value()
                best_action = action
        
        print("final answer: best action: ", best_action, "with value: ", best_value)
        return best_action
    
    def find_best_action_recursive(self, root):
        print(root.player.name, "searching for best action")
        
        action_dict = self.supervised_agent_act(root.state, self.models[root.game.street])
        agent_valid_moves = self.mask_invalid_moves(action_dict, root.player, root.game, root.moves_left)
        print(root.player.name, "valid moves: ", agent_valid_moves)
        if (root.action is not None and root.parent is not None):
            if root.parent.player is not None:
                print("last player ", root.parent.player.name, "chose action: ", root.action)

        for action, prob in agent_valid_moves.items():
            root.children_values[action] = 0
            allowed = int(k * (root.visit_count ** alpha)) or 1
            if len(root.children) > allowed:
                continue
            game_copy = copy.deepcopy(root.game)
            player_copy = copy.deepcopy(root.player)
            game_simulation, next_player = self.simulate_action(root, action, game_copy, game_copy.get_player_by_name(root.player.name))
            # Create a DEEP COPY of the game simulation for the child node
            if next_player is not None:
                from MCTS.agent_actions import State
                child_state = State(game_copy, next_player)
                root.children[action] = MCTSNode(prior_prob=prob, state=child_state, game=game_simulation, player=next_player, moves_left=root.moves_left-1, action=action, parent=root)
            else:
                root.children[action] = MCTSNode(prior_prob=prob, state=None, game=game_copy, player=None, moves_left=0, action=action, parent=root)
            
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # SELECT
            if node.has_children() and not node.is_terminal:
                node, action  = node.select_child()
                print("selected action: ", action)
                search_path.append(node)

            if node.is_terminal:
                print("this is a terminal node")
                ev_map = self.calc_ev_map(node.game)
                print("ev_map: ", ev_map)
                self.backpropagate(search_path, ev_map)
            else:
                print("this is not a terminal node, now it's next player's turn")
                # Check if we've reached maximum depth
                if node.moves_left <= 0:
                    print("Reached maximum depth, treating as terminal")
                    ev_map = self.calc_ev_map(node.game)
                    self.backpropagate(search_path, ev_map)
                else:
                    self.find_best_action_recursive(node)
    
    def backpropagate(self, search_path, ev_map):
        for node in reversed(search_path):
            if node.player is not None and node.player.name in ev_map:
                node.value_sum += ev_map[node.player.name]

            if node.parent is not None and node.parent.player is not None and node.parent.player.name in ev_map:
                node.parent.children_values[node.action] += ev_map[node.parent.player.name]
            node.visit_count += 1

    def calc_ev_map(self, game):
        ev_map = {}
        
        if len(game.get_players_in_hand()) == 1:
            winner = game.get_players_in_hand()[0]
            #this is not accurate, because_chips_in_pot contatins also previous bets
            ev_map[winner.name] = game.pot_size
        else:
            if game.street == RIVER:
                winner, winning_hand = game.deck.find_winner(game.get_players_in_hand(), game.community_cards)
                ev_map[winner.name] = game.pot_size
            else:
                print("this should never happen")
                pdb.set_trace()
                #winner_map = self.calc_ev_map_simulation(game.get_players_in_hand(), game)
                #winner, winning_hand = game.deck.find_winner(game.get_players_in_hand(), game.community_cards)
                #for player in winner_map.keys():
                #    ev_map[player] = winner_map[player] * game.pot_size
                #ev_map[winner.name] = game.pot_size

        # Ensure all players have an entry in ev_map
        for player in game.get_players_in_hand():
            if player.name not in ev_map:
                ev_map[player.name] = 0

        for player_name in ev_map.keys():
            player_obj = game.get_player_by_name(player_name)
            if player_obj is not None:
                ev_map[player_name] -= getattr(player_obj, 'mcts_chips_in_pot', 0)

        return ev_map
'''

   