import random
import json
import os
import torch
import numpy as np
from AppUtils.files_utils import find_agent_path
from AppUtils.constants import get_general_position, PREFLOP, FLOP, TURN, RIVER, ALL_POSITIONS_POST_FLOP
from AppUtils.cards_utils import cards_str
from AppUtils.agent_utils import action_map
import AppUtils.StringUtils as strings
from AppUtils.hand_board_calculations import calc_strength_pre, calc_board_dynamic, calc_hand_strength, calculate_flush_draw, calculate_straight_draw, is_flush_hit, calc_draw_on_turn
from MCTS.mcts import MCTS
import pdb
# Import Supervised Poker Agent
try:
    from agent.supervised_poker_agent import SupervisedPokerAgent

    SUPERVISED_AGENT_AVAILABLE = True
except ImportError:
    print("Warning: Supervised poker agent not available. Using fallback logic.")
    SUPERVISED_AGENT_AVAILABLE = False


class Action:
    """Represents a poker action with frequency and amount"""
    def __init__(self, action, freq, amount): 
        self.action = action
        self.freq = freq
        self.amount = amount

    def __repr__(self):
        return f"{self.action} ({self.freq}%) {self.amount}"

    def __str__(self):
        return f"{self.action} ({self.freq}%) {self.amount}"


class AgentActions:
    """Handles all agent-related decision making logic"""
    
    def __init__(self):
        # Initialize Supervised Poker Agent if available
        self.supervised_agents = {}  # Dictionary to store models for each street
        self.data_processor = None
        
        # Load models for each street with correct input dimensions
        self.street_models = {
            FLOP: (find_agent_path(FLOP), 86),
            TURN: (find_agent_path(TURN), 89), 
            RIVER: (find_agent_path(RIVER), 90)
        }

        self.MCTS_model = None

        if SUPERVISED_AGENT_AVAILABLE:
            self._initialize_supervised_agents()
            self._initialize_mcts()

    def _initialize_supervised_agents(self):
        """Initialize supervised poker agents for each street"""
        try:                
            for street, (model_path, input_dim) in self.street_models.items():
                try:
                    # Create agent for this street with correct input dimension
                    agent = SupervisedPokerAgent(input_dim=input_dim, num_actions=4)
                    # Load the trained model for this street
                    checkpoint = torch.load(model_path, map_location='cpu')
                    agent.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model for street {street}")
                    agent.eval()
                    
                    self.supervised_agents[street] = agent
                    print(f"Loaded {model_path} for street {street}")
                    
                except FileNotFoundError:
                    print(f"No trained model found for street {street} ({model_path})")
                except Exception as e:
                    print(f"Failed to load model for street {street}: {e}")
            
            if self.supervised_agents:
                # Initialize data processor for encoding game states
                print(f"Successfully loaded {len(self.supervised_agents)} street-specific models")
            else:
                print(f"No street-specific models available")
                
        except Exception as e:
            print(f"Failed to initialize supervised agents: {e}")
            self.supervised_agents = {}

    def _initialize_mcts(self):
        """Initialize MCTS with AgentActions wrapper as neural network"""
        try:
            # Use AgentActions wrapper (self) as the neural network for MCTS
            if self.supervised_agents:
                self.MCTS_model = MCTS(self.supervised_agents)
                print(f"MCTS initialized successfully with AgentActions wrapper")
            else:
                print(f"No supervised agents available for MCTS initialization")
        except Exception as e:
            print(f"Failed to initialize MCTS: {e}")


    def mcts_act(self, game, player):
        """ Use MCTS to find the best action """
        if not self.MCTS_model:
            print("MCTS not initialized")
            return False
        
        try:
            state = State(game, player)
            # Use MCTS to find best action
            best_action = self.MCTS_model.find_best_action(state, game, player)
            if best_action == strings.RAISE:
                num_raises = game.get_num_raises_in_current_street();
                bet_amount = game.pot_size * (0.55 if num_raises == 0 else 2.2)
            elif best_action == strings.CALL:
                bet_amount = game.last_bet_size
            else:
                bet_amount = 0
            
            if best_action:
                print(f"MCTS decided to {best_action}")
                return Action(best_action, 100, bet_amount)
            else:
                print("MCTS returned no action")
                return False
                
        except Exception as e:
            print(f"MCTS failed: {e}")
            return False

    def act(self, game_state):
        """
        Supervised agent decision-making method
        Args: game_state: State object containing all game state information
        Returns: Action: The chosen action, or False if failed
        """
        if not SUPERVISED_AGENT_AVAILABLE:
            return False

        if game_state.street not in self.supervised_agents:
            return False
        
        try:
            action, confidence = self.supervised_agent_act(game_state)
            action_name = action_map.get(action, f"UNKNOWN({action})")
            print(f"Supervised agent (street {game_state.street}) decided to {action_name} (action {action}) with confidence {confidence:.3f}")
            
            if action is not None:
                # Convert supervised agent prediction to Action object
                if action == 3:  # Raise
                    bet_amount = game_state.pot_size * 0.55  # 55% pot bet
                    return Action(strings.RAISE, 100, bet_amount)
                elif action == 2:  # Call
                    return Action(strings.CALL, 100, game_state.last_bet_size)
                elif action == 1:  # Check
                    return Action(strings.CHECK, 100, 0)
                elif action == 0:  # Fold
                    return Action(strings.FOLD, 100, 0)
            
            return False
            
        except Exception as e:
            print(f"Supervised agent prediction failed for street {game_state.street}: {e}")
            return False

class State:
    def __init__(self, game, player):
        self.stage = 0 # TODO
        self.street = game.street
        self.prev_actions_in_street = game.prev_actions_in_street
        self.last_bet_size = game.last_bet_size
        self.community_cards = game.community_cards
        self.pot_size = game.pot_size
        self.num_of_players = game.num_of_players
        self.hand_str = player.hand.to_string()
        self.hand_cards = player.hand.cards
        self.hand = player.hand  # Include the hand object for supervised agent
        self.position = get_general_position(player.position)
        self.position_index = player.position_index  # Include position index for supervised agent
        self.stacks_size = game.stacks_size
        self.players_in_hand = len(game.get_players_in_hand())
        self.num_of_players_in_streets = game.num_of_players_in_streets
        self.current_hand_actions_strength = game.current_hand_actions_strength
        self.stack_size = player.stack_size
        self.is_heads_up = game.num_of_players == 2
        self.is_pre_flop_agressor = getattr(player, 'is_pre_flop_agressor', False)
        self.is_last_agressor = getattr(player, 'is_last_agressor', False)
        
        if self.street == FLOP:
            self.flush_draw = calculate_flush_draw(self.hand.cards, self.community_cards)
            self.straight_draw = calculate_straight_draw(self.hand.cards, self.community_cards)
            self.board_dynamic = calc_board_dynamic(self.community_cards)
        elif self.street == TURN:
            self.draw = calc_draw_on_turn(self.hand.cards, self.community_cards)
            self.is_flush_hit = is_flush_hit(self.street, self.community_cards[:3], self.community_cards[3], None)
            self.turn_action = self.current_hand_actions_strength[TURN] if self.street >= TURN else 0
            self.is_hero_last_aggressor = self.is_last_agressor
        elif self.street == RIVER:
            self.is_flush_hit = is_flush_hit(self.street, self.community_cards[:3], self.community_cards[3], self.community_cards[4])
            self.turn_action = self.current_hand_actions_strength[TURN]
            self.river_action = self.current_hand_actions_strength[RIVER]
            self.is_hero_last_aggressor = self.is_last_agressor

    def to_dict(self):
        """Convert State object to dictionary for data processor"""
        game_state_data = {
            'street': self.street,
            'hero_cards': cards_str(self.hand.cards) if self.hand and self.hand.cards else [],
            'board_cards': cards_str(self.community_cards) if self.community_cards else [],
            'hero_pos': ALL_POSITIONS_POST_FLOP[self.position_index],
            'pot_size': self.pot_size,
            'stack_size': self.stack_size,
            'spr': self.stack_size / self.pot_size if self.pot_size > 0 else 0,
            'stage': self.stage,
            'hand_strength': calc_hand_strength(self.hand.cards, self.community_cards),
            'num_of_players_pre_flop': self.num_of_players_in_streets[PREFLOP] if self.street != PREFLOP else self.players_in_hand,
            'num_of_players_flop': self.num_of_players_in_streets[FLOP] if self.street != FLOP else self.players_in_hand,
            'num_of_players_turn': self.num_of_players_in_streets[TURN] if self.street != TURN else self.players_in_hand,
            'num_of_players_river': self.num_of_players_in_streets[RIVER] if self.street != RIVER else self.players_in_hand,
            'preflop_action': self.current_hand_actions_strength[PREFLOP],
            'flop_action': self.current_hand_actions_strength[FLOP],
            'is_hero_pre_flop_aggressor': self.is_pre_flop_agressor,
        }
        
        # Add street-specific features
        if self.street == FLOP:
            game_state_data.update({
                'flush_draw': getattr(self, 'flush_draw', 0),
                'straight_draw': getattr(self, 'straight_draw', 0),
                'board_dynamic': getattr(self, 'board_dynamic', 0),
            })
        elif self.street == TURN:
            game_state_data.update({
                'draw': getattr(self, 'draw', 0),
                'is_flush_hit': getattr(self, 'is_flush_hit', False),
                'turn_action': getattr(self, 'turn_action', 0),
                'is_hero_last_aggressor': getattr(self, 'is_hero_last_aggressor', False),
            })
        elif self.street == RIVER:
            game_state_data.update({
                'is_flush_hit': getattr(self, 'is_flush_hit', False),
                'turn_action': getattr(self, 'turn_action', 0),
                'river_action': getattr(self, 'river_action', 0),
                'is_hero_last_aggressor': getattr(self, 'is_hero_last_aggressor', False),
            })
        
        return game_state_data

    def print_state(self):
        """Print State object in a readable format"""
        print("=" * 50)
        print("STATE OBJECT:")
        print("=" * 50)
        print(f"Street: {self.street}")
        print(f"Hand: {self.hand_str}")
        print(f"Community Cards: {cards_str(self.community_cards) if self.community_cards else 'None'}")
        print(f"Position: {self.position} (index: {self.position_index})")
        print(f"Pot Size: {self.pot_size}")
        print(f"Stack Size: {self.stack_size}")
        print(f"Last Bet Size: {self.last_bet_size}")
        print(f"Number of Players: {self.num_of_players}")
        print(f"Players in Hand: {self.players_in_hand}")
        print(f"Is Heads Up: {self.is_heads_up}")
        print(f"Is Pre-flop Aggressor: {self.is_pre_flop_agressor}")
        print(f"Is Last Aggressor: {self.is_last_agressor}")
        
        # Street-specific features
        if self.street == FLOP:
            print(f"Flush Draw: {getattr(self, 'flush_draw', 'N/A')}")
            print(f"Straight Draw: {getattr(self, 'straight_draw', 'N/A')}")
            print(f"Board Dynamic: {getattr(self, 'board_dynamic', 'N/A')}")
        elif self.street == TURN:
            print(f"Draw: {getattr(self, 'draw', 'N/A')}")
            print(f"Is Flush Hit: {getattr(self, 'is_flush_hit', 'N/A')}")
            print(f"Turn Action: {getattr(self, 'turn_action', 'N/A')}")
            print(f"Is Hero Last Aggressor: {getattr(self, 'is_hero_last_aggressor', 'N/A')}")
        elif self.street == RIVER:
            print(f"Is Flush Hit: {getattr(self, 'is_flush_hit', 'N/A')}")
            print(f"Turn Action: {getattr(self, 'turn_action', 'N/A')}")
            print(f"River Action: {getattr(self, 'river_action', 'N/A')}")
            print(f"Is Hero Last Aggressor: {getattr(self, 'is_hero_last_aggressor', 'N/A')}")
        
        print("=" * 50)