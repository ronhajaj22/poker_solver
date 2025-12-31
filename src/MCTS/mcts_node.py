"""
MCTS Node implementation for the Monte Carlo Tree Search algorithm.
Represents a single node in the search tree.
"""
import random, numpy as np, math

def ucb_score(parent, child, original_player_position=None):
    """
    Calculate UCB score for a child node.
    Uses PUCT (Predictor + UCT) formula from AlphaZero.
    """
    if child.visit_count == 0:
        return float('inf')  # Prioritize unvisited children
    
    # Q(s,a) - Average value (exploitation)
    exploitation = child_expected_value_for_parent(parent, child, original_player_position)
    
    # U(s,a) - Exploration bonus (UCB1 style)
    c_puct = 10  # exploration constant
    prior = child.prior_prob
    
    exploration = c_puct * prior * (math.sqrt(parent.visit_count) / (1 + child.visit_count))
    ucb_score = exploitation + exploration
    return ucb_score


def child_expected_value_for_parent(parent, child, original_player_position=None):
    """
    Calculate the expected value of a child node from the parent's perspective.
    """
    parent_position = parent.player_position

    if parent_position is None or parent_position not in child.value_sums:
        print("Error! parent position should not be empty")
        return 0.0

    if child.visit_count == 0 or parent_position != original_player_position:
        return 0.0
    
    value = child.value_sums[parent_position] / child.visit_count
    return value / parent.pot_size


class MCTSNode:
    """
    Represents a node in the MCTS search tree.
    Each node corresponds to a game state and contains statistics about visits and values.
    """
    
    def __init__(self, prior_prob, player_position=None, parent=None, is_fold_node=False, 
                 street=None, pot_size=None):
        self.prior_prob = prior_prob  # The prior probability of selecting this state from its parent
        self.player_position = player_position
        self.parent = parent  # the parent node
        self.children = {}  # A lookup of legal child nodes, basically a map of action -> the node it creates
        
        self.pot_size = pot_size
        self.street = street
        
        self.visit_count = 0  # The number of times this node has been visited
        self.value_sums = {}  # The total value of this node for each player from all visits

        self.is_fold_node = is_fold_node

        self.main_player_range = {}
    
    def apply_dirichlet_noise(self, epsilon=0.25):
        """
        Mix the children's Priors with Dirichlet noise.
        epsilon: how much weight to give to noise (0.25 is the standard of AlphaZero).
        """
        if not self.children:
            return
        
        actions = list(self.children.keys())
        alpha = 1/len(actions) # 'temperature' parameter. A low value (like 0.1) makes the noise bigger and the agent less expected.
        noise = np.random.dirichlet([alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = self.children[action]
            # The formula of AlphaZero for mixing noise
            child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * noise[i]

    def has_children(self):
        """Check if this node has any children"""
        return len(self.children) > 0

    def select_child_by_probability(self):
        """
        Select a child node based on probability distribution.
        Used as a fallback when UCB selection is not appropriate.
        """
        children_probabilities = {action: (0.75 * child.prior_prob + 0.25) for action, child in self.children.items()}
        action = random.choices(
            list(children_probabilities.keys()), weights=list(children_probabilities.values()))[0]
        return self.children[action], action
    
    def set_main_player_range(self, main_player_range):
        """Set the main player's range for this node"""
        self.main_player_range = main_player_range

    def select_child(self, original_player_position=None):
        """
        Select child using UCB1 formula.
        Returns the best child node and the action that leads to it.
        """
        # Collect all children with their scores
        children_scores = [(action, child, ucb_score(self, child, original_player_position)) for action, child in self.children.items()]
        # Find max score
        max_score = max(score for _, _, score in children_scores)
        # Get all children with max score
        best_children = [
            (action, child) for action, child, score in children_scores 
            if score == max_score
        ]
        # If multiple with same score, choose by prior (or randomly if same prior)
        if len(best_children) > 1:
            best_action, best_child = random.choices(best_children, weights=[child.prior_prob for _, child in best_children])[0]
        else:
            best_action, best_child = best_children[0]
        
        return best_child, best_action

    def __repr__(self):
        """String representation of the node"""
        prior = "{0:.2f}".format(self.prior_prob)
        return f"MCTSNode(Prior: {prior}, Count: {self.visit_count}, Position: {self.player_position})"

