import pdb
from typing import Dict, List, Optional
from AppUtils.constants import ALL_POSITIONS_POST_FLOP
from AppUtils.cards_utils import card_str_to_int_rank, RANKS, SUITS
from AppUtils import agent_utils
from AppUtils.files_utils import find_parsed_hands_path
import torch
import json
from typing import Tuple

def load_data_and_create_tensors(current_street: int) -> Tuple[torch.Tensor, torch.Tensor]:
    data_path = find_parsed_hands_path(current_street);
    hands_data = load_file(data_path)
    
    all_hands_features = []
    all_hands_actions = []
    print(f"current_street: {current_street}")

    for hand_index, hand in enumerate(hands_data):
        features_tensor = encode_features(hand, agent_utils.FEATURES_BY_STREET[current_street], current_street)
        action = extract_action_label(hand)
        #for feature_name, feature_value in features_map.items():
         #   print(f"{feature_name}: {feature_value}")
        #print(f"action: {action}")
        if (action is not None):
            all_hands_features.append(features_tensor)
            all_hands_actions.append(action)
    
    final_features_values_tensor = torch.stack(all_hands_features)
    final_actions_tensor = torch.tensor(all_hands_actions, dtype=torch.long)
    print(f"final_features_values_tensor: {final_features_values_tensor}")
    print(f"final_actions_tensor: {final_actions_tensor}")
    return final_features_values_tensor, final_actions_tensor

def load_file(file_path: str) -> List[Dict]:
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            hands_data = json.load(f)
        return hands_data

def encode_features(hand_data: Dict, feature_list: List[str], street: int) -> torch.Tensor:
    # Encode features into a single tensor.
    features = {}
    # first we hande action features as one tensor and num of players as one tensor
    action_features = ['preflop_action', 'flop_action', 'turn_action', 'river_action']
    num_players_features = ['num_of_players_pre_flop', 'num_of_players_flop', 'num_of_players_turn', 'num_of_players_river']
    actions_tensor = create_action_tensor(hand_data, action_features, street)
    features['actions_tensor'] = actions_tensor
    num_players_tensor = create_num_players_tensor(hand_data, num_players_features, street)
    features['num_players_tensor'] = num_players_tensor

    #print(f"features: {features}")
    for feature_name in feature_list:
        if feature_name in action_features or feature_name in num_players_features:
            continue # already handled
        elif feature_name == 'stage':
            features[feature_name] = torch.tensor([round(hand_data.get(feature_name, 0)/10, 1)])
        elif feature_name == 'hero_cards' or feature_name == 'board_cards':
            cards = hand_data.get(feature_name, [])
            cards_set  = []
            suits_set = torch.zeros(4)
            if cards and all(card[0] in RANKS for card in cards):
                #card_encoding = self.encode_cards(cards)
                #features.append(card_encoding)
                for card in cards:
                    cards_set.append(card_str_to_int_rank(card))
                    suits_set[SUITS.index(card[1])] += round(1)
                features[feature_name] = torch.tensor(cards_set)
                features[feature_name+'_suit'] = suits_set
                #print ({feature_name: features[feature_name]})
                #print ({feature_name+'_suit': features[feature_name+'_suit']})
            else:
                print(f"Invalid cards: {cards}")
        elif feature_name == 'hero_pos':
            position_encoding = torch.zeros(9)
            position_encoding[ALL_POSITIONS_POST_FLOP.index(hand_data.get(feature_name, 0))] = round(1)
            features[feature_name] = position_encoding
        elif feature_name in ['stack_size', 'pot_size']:
            features[feature_name] = torch.tensor([round(hand_data.get(feature_name, 0)/100, 3)])
        elif feature_name == 'spr':
            features[feature_name] = torch.tensor([round(hand_data.get(feature_name, 0)/10, 2)])
        elif feature_name in ['is_hero_pre_flop_aggressor', 'is_hero_last_aggressor', 'is_flush_hit']:
            value = hand_data.get(feature_name, False)
            if isinstance(value, bool):
                features[feature_name] = torch.tensor([round(1) if value else round(0)])
            else:
                features[feature_name] = torch.tensor([0])
        elif feature_name == 'hand_strength':
            features[feature_name] = torch.tensor([round(hand_data.get(feature_name, 0)/100, 4)])
        elif feature_name in ['draw', 'flush_draw', 'board_dynamic', 'straight_draw']:
            features[feature_name] = torch.tensor([round(hand_data.get(feature_name, 0), 4)])
        else:
            print(f"Unknown feature: {feature_name}")
            # Unknown feature - use default value
            features[feature_name] = torch.tensor([0])
    
    features_tensor = torch.cat(list(features.values()))
    return features_tensor


def create_action_tensor(hand_data: Dict, action_features: List[str], street: int) -> torch.Tensor:
    """
    Create action tensor from hand data.
    """
    action_values = []
    for feature_name in action_features[:street+1]:
        value = hand_data.get(feature_name, 0)
        if isinstance(value, (int, float)):
            action_values.append(float(value))
        else:
            print("no instance, break")
            break;
    return torch.tensor(action_values)

def create_num_players_tensor(hand_data: Dict, num_players_features: List[str], street: int) -> torch.Tensor:
    num_players_values = []
    for feature_name in num_players_features[:street+1]:
        value = hand_data.get(feature_name, 0)
        if isinstance(value, (int, float)):
            num_players_values.append(round(float(value/10), 1))
        else:
            print("no instance, break")
            break;
    return torch.tensor(num_players_values)

def extract_action_label(hand_data: Dict) -> Optional[int]:
    """
    Extract action number from hand data for supervised learning.
    Returns None if no valid action found in data.
    """
    action_str = hand_data.get('action', '')
    if not action_str:
        return None

    return agent_utils.reversed_action_map[action_str.upper()]

def create_train_and_val_loaders(features: torch.Tensor, labels: torch.Tensor, 
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