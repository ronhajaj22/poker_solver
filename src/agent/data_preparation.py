import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from agent.card_encoder import create_one_hot_card, create_hand_encoding
from AppUtils.agent_utils import FEATURES_BY_STREET, agent_global_features, reversed_action_map
from AppUtils.constants import PREFLOP, FLOP, TURN, RIVER, ALL_POSITIONS_POST_FLOP
from AppUtils.cards_utils import SUITS, RANKS
import random

import pdb

class PokerDataProcessor:
    """
    Converts raw poker hand data into training-ready tensors for supervised learning.
    Processes game states (cards, position, stacks) and extracts action labels.
    """
    
    def __init__(self, card_encoder=None):
        """
        Initialize mappings for cards, positions, and actions.
        card_encoder: Advanced card encoder for sophisticated representations
        """
        # Initialize card encoder if not provided
        if card_encoder is None:
            from agent.card_encoder import CardEncoder
            self.card_encoder = CardEncoder(embedding_dim=32)
        else:
            self.card_encoder = card_encoder
        
        # Set card encoder to evaluation mode to avoid batch norm issues
        self.card_encoder.eval()
        
        # Map card strings (As, Kh) to indices 0-51 for one-hot encoding
        self.card_to_index = self._create_card_mapping()
    
    def _create_card_mapping(self) -> Dict[str, int]:
        """
        Create 52-card mapping: 2s=0, 3s=1, ..., As=12, 2h=13, ..., Ac=51.
        Returns dictionary mapping card strings to indices for one-hot encoding.
        """
        card_mapping = {}
        index = 0
        
        # Create mapping: each suit gets 13 consecutive indices
        for suit in SUITS:
            for rank in RANKS:
                card_mapping[f"{rank}{suit}"] = index
                index += 1
        
        return card_mapping
    
    def encode_cards(self, card_strings: List[str]) -> torch.Tensor:
        """
        Convert card list ['As', 'Kh'] to 32-dim card embeddings.
        Uses card encoder for sophisticated card representations.
        """
        if self.card_encoder is not None:
            # Use card encoder for advanced representations
            card_indices = [self.card_to_index[card] for card in card_strings]
            one_hot_encoding = create_hand_encoding(card_indices)
            # Add batch dimension for card encoder
            one_hot_batch = one_hot_encoding.unsqueeze(0)
            # Get 32-dim embeddings from card encoder
            with torch.no_grad():
                # Ensure we're in eval mode and handle single sample
                self.card_encoder.eval()
                card_embeddings = self.card_encoder(one_hot_batch)
            return card_embeddings.squeeze(0)  # Remove batch dimension
        else:
            # Fallback to one-hot encoding
            card_indices = [self.card_to_index[card] for card in card_strings]
            return create_hand_encoding(card_indices)
    
    def encode_game_state(self, hand_data: Dict) -> torch.Tensor:
        """
        Encode complete poker situation into single feature vector.
        Uses features from agent_utils.py based on current street.
        Returns concatenated tensor ready for neural network input.
        """
        
        # Get current street and determine which features to use
        street = hand_data.get('street', 0)
            
        # Select features based on street
        feature_list = FEATURES_BY_STREET[street]

        features = []
        
        # Process each feature in the selected feature list
        for feature_name in feature_list:
            if feature_name == 'hero_cards' or feature_name == 'board_cards':
                if feature_name == 'hero_hand':
                    pdb.set_trace()
                cards = hand_data.get(feature_name, [])
                if cards and all(card in self.card_to_index for card in cards):
                    card_encoding = self.encode_cards(cards)
                    features.append(card_encoding)
                else:
                    print(f"Invalid cards: {cards}")
                    features.append(torch.zeros(32))
                    
            elif feature_name == 'hero_pos':
                # Player position: 9-dim one-hot encoding
                position_encoding = torch.zeros(9)
                hero_pos = hand_data.get('hero_pos', 0)
                hero_pos_index = ALL_POSITIONS_POST_FLOP.index(hero_pos)
                if isinstance(hero_pos, int) and 0 <= hero_pos_index < 9:
                    position_encoding[hero_pos_index] = 1
                features.append(position_encoding)
                
            elif feature_name in ['pot_size', 'stack_size', 'spr']:
                # Numeric features - normalize for training stability
                value = hand_data.get(feature_name, 0)
                if isinstance(value, (int, float)):
                    if feature_name in ['pot_size', 'stack_size']:
                        normalized_value = value / 100.0  # Normalize to [0,1]
                    else:  # spr
                        normalized_value = value / 10.0   # Stack-to-pot ratio
                    features.append(torch.tensor([normalized_value]))
                else:
                    features.append(torch.tensor([0.0]))
                    
            elif feature_name in ['hand_strength', 'flush_draw', 'straight_draw', 'is_flush_hit', 'board_dynamic', 'draw']:
                # Hand evaluation features
                value = hand_data.get(feature_name, 0)
                if isinstance(value, (int, float)):
                    if feature_name == 'is_flush_hit':
                        if isinstance(value, bool):
                            features.append(torch.tensor([1.0 if value else 0.0]))
                        else:
                            features.append(torch.tensor([0.0]))
                    if feature_name == 'hand_strength':
                        normalized_value = value / 100.0
                    else:
                        normalized_value = float(value)  # Keep as is for draws
                    features.append(torch.tensor([normalized_value]))
                else:
                    features.append(torch.tensor([0.0]))
                    
            elif feature_name in ['preflop_action', 'flop_action', 'turn_action', 'river_action']:
                # Action features
                value = hand_data.get(feature_name, 0)
                if isinstance(value, (int, float)):
                    features.append(torch.tensor([float(value)]))
                else:
                    features.append(torch.tensor([0.0]))
                    
            elif feature_name in ['num_of_players_pre_flop', 'num_of_players_flop', 'num_of_players_turn', 'num_of_players_river']:
                # Player count features
                value = hand_data.get(feature_name, 0)
                if isinstance(value, (int, float)):
                    features.append(torch.tensor([float(value / 10.0)]))
                else:
                    features.append(torch.tensor([0.0]))
                    
            elif feature_name in ['is_hero_pre_flop_aggressor', 'is_hero_last_aggressor', 'is_flush_hit']:
                # Boolean features
                value = hand_data.get(feature_name, False)
                if isinstance(value, bool):
                    features.append(torch.tensor([1 if value else 0]))
                else:
                    features.append(torch.tensor([0]))
                    
            elif feature_name in ['stage']:
                value = hand_data.get(feature_name, 0)
                if isinstance(value, (int, float)):
                    features.append(torch.tensor([float(value/10.0)]))
                else:
                    features.append(torch.tensor([0]))
                    
            else:
                print(f"Unknown feature: {feature_name}")
                # Unknown feature - use default value
                features.append(torch.tensor([0]))

           # print(f"feature: {feature_name}", "feature_value: ", hand_data.get(feature_name))
        
        # Concatenate all features into single input vector
        return torch.cat(features)
    
    def extract_action_label(self, hand_data: Dict) -> Optional[int]:
        """
        Extract action number from hand data for supervised learning.
        Returns None if no valid action found in data.
        """
        action_str = hand_data.get('action', '')
        if not action_str:
            return None

        return reversed_action_map[action_str.upper()]

    def extract_masked_options(self, hand_data: Dict) -> Optional[List[str]]:
        """
        Extract masked options from hand data for action masking.
        Returns list of available actions for the current street.
        """
        street = hand_data.get('street', 0)
        action_value = 0
        # Extract action value based on current street
        if street == PREFLOP and 'preflop_action' in hand_data:
            action_value = hand_data['preflop_action']  
        elif street == FLOP and 'flop_action' in hand_data:
            action_value = hand_data['flop_action']    
        elif street == TURN and 'turn_action' in hand_data:
            action_value = hand_data['turn_action']    
        elif street == RIVER and 'river_action' in hand_data:
            action_value = hand_data['river_action']  
        else:
            print(f"No action data for this street: {street}")
            pdb.set_trace()
            return None  # No action data for this street
        
        return [['FOLD', 'CALL'] if action_value < 0 else ['CHECK'] if action_value > 0 else []]
    
    
    
    def process_data(self, hands_data: List[Dict]):
        for hand in hands_data:
            pdb.set_trace()

    def load_and_process_data(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load JSON poker data and convert to training tensors.
        Processes each hand: encodes game state, extracts action label.
        Returns (features, labels) tensors ready for neural network training.
        """
        
        hands_data = self.load_file(file_path)
        
        features_list = []
        labels_list = []
        
        # Process each poker hand into training example
        for hand in hands_data:
            try:
                # Encode complete game state as feature vector
                game_state = self.encode_game_state(hand)
                
                # Extract action label for supervised learning - TODO
                action_label = self.extract_action_label(hand)
                if action_label is not None:
                    features_list.append(game_state)
                    labels_list.append(action_label)
                else:
                    print(f"No action label for hand: {hand}")

            except Exception as e:
                print(f"Error processing hand {hand.get('hand_id', 'unknown')}: {e}")
                continue

        # Convert lists to PyTorch tensors for training
        features = torch.stack(features_list)
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        print(f"Processed {len(features)} hands")
        print(f"Feature tensor shape: {features.shape}")
        print(f"Label tensor shape: {labels.shape}")
        
        return features, labels
    
    def create_data_loaders(self, features: torch.Tensor, labels: torch.Tensor, 
                           train_split: float = 0.8, batch_size: int = 32):
        """
        Create training and validation data loaders for neural network training.
        Splits data 80/20, creates batches, enables shuffling for training.
        Returns (train_loader, val_loader) for supervised learning.
        """
        from torch.utils.data import TensorDataset, DataLoader, random_split
        
        # Create PyTorch dataset from features and labels
        dataset = TensorDataset(features, labels)
        
        # Split dataset into training (80%) and validation (20%) sets
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders with batching and shuffling
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader

def main():
    print("Data preparation!")

if __name__ == "__main__":
    main() 