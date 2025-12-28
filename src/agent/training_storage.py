"""
=== TRAINING STORAGE ===

Singleton storage for collecting and managing training data for supervised learning.

MOTIVATION:
Provides centralized storage for training samples (features, actions, bet sizes, masks) collected
during gameplay. Supports separate storage for regular agents and blind agents, organized by street.
Enables batch retrieval of training data for neural network training.

CLASSES:
- TrainingDataStorage - Storage for training data collection
  Methods: __init__(), add_training_sample(street, is_blind_agent, feature_tensor, action, bet_size, masked_actions, bet_size_mask), get_count(street), get_training_data(street, is_blind_agent=False)

FUNCTIONS (signatures):
- get_training_storage() - Get or create singleton instance of TrainingDataStorage
"""

import torch
import numpy as np
import os
import pickle
from AppUtils.constants import FLOP, TURN, RIVER

# Singleton instance of TrainingDataStorage
training_storage_instance = None

def get_training_storage():
    """Get or create the singleton instance of TrainingDataStorage."""
    global training_storage_instance
    if training_storage_instance is None:
        training_storage_instance = TrainingDataStorage()
    return training_storage_instance

class TrainingDataStorage:
    """Storage for collecting training data for supervised learning."""
    def __init__(self):
        self.training_data = {
            FLOP: {'features': [], 'action_labels': [], 'bet_size_labels': [], 'action_masks': [], 'bet_size_bounds': []},
            TURN: {'features': [], 'action_labels': [], 'bet_size_labels': [], 'action_masks': [], 'bet_size_bounds': []},
            RIVER: {'features': [], 'action_labels': [], 'bet_size_labels': [], 'action_masks': [], 'bet_size_bounds': []}
        }
        self.blind_training_data = {
            FLOP: {'features': [], 'action_labels': [], 'bet_size_labels': [], 'action_masks': [], 'bet_size_bounds': []},
            TURN: {'features': [], 'action_labels': [], 'bet_size_labels': [], 'action_masks': [], 'bet_size_bounds': []},
            RIVER: {'features': [], 'action_labels': [], 'bet_size_labels': [], 'action_masks': [], 'bet_size_bounds': []}
        }
    
    def add_training_sample(self, street, is_blind_agent, feature_tensor=None, action=None, bet_size=None, masked_actions=None, bet_size_mask=None):
        if street not in self.training_data:
            return
        
        training_data = self.blind_training_data if is_blind_agent else self.training_data

        if feature_tensor is not None:
            training_data[street]['features'].append(feature_tensor.clone().detach())
        
        # Handle action
        if action is not None:
            # Convert to tensor if it's numpy array or other type
            if isinstance(action, np.ndarray):
                # If it's one-hot vector, get the index
                if action.ndim > 0 and len(action) > 1:
                    action_label = torch.tensor(np.argmax(action), dtype=torch.long)
                else:
                    action_label = torch.tensor(int(action), dtype=torch.long)
            elif isinstance(action, torch.Tensor):
                if action.dim() > 0 and action.numel() > 1:
                    # One-hot vector - get the index
                    action_label = torch.argmax(action).long()
                else:
                    action_label = action.long()
            else:
                # Scalar value
                action_label = torch.tensor(int(action), dtype=torch.long)
            
            training_data[street]['action_labels'].append(action_label)
            
        else:
            print(f"Error! Action is None")
            return

        # Handle bet_size
        if bet_size is not None:
            # Convert to tensor if it's not already
            if isinstance(bet_size, torch.Tensor):
                bet_size_label = bet_size.clone().detach()
            elif isinstance(bet_size, np.ndarray):
                bet_size_label = torch.tensor(bet_size, dtype=torch.float32)
            else:
                bet_size_label = torch.tensor([float(bet_size)], dtype=torch.float32)
            
            # Ensure it's shape (1,) for consistency
            if bet_size_label.dim() == 0:
                bet_size_label = bet_size_label.unsqueeze(0)
            
            training_data[street]['bet_size_labels'].append(bet_size_label)
            
        else: # Add zero bet size if not provided
            training_data[street]['bet_size_labels'].append(torch.tensor([0.0], dtype=torch.float32))
        
        # Handle masked_actions (action mask)
        if masked_actions is not None:
            action_mask = masked_actions.clone().detach()
        else:
            action_mask = torch.ones(4, dtype=torch.float32)
        
        training_data[street]['action_masks'].append(action_mask)
        
        # Handle bet_size_mask - lambda function that filters valid bet sizes
        training_data[street]['bet_size_bounds'].append(bet_size_mask)
    

    # Get number of training samples for a street
    def get_count(self, street):
        if street not in self.training_data:
            return 0
        return len(self.training_data[street]['features'])
    
    def get_training_data(self, street, is_blind_agent=False):
        """Get training data for a specific street as tensors."""
        data_source = self.blind_training_data if is_blind_agent else self.training_data
        
        if street not in data_source:
            return None, None, None, None, None
        
        street_data = data_source[street]
        
        if len(street_data['features']) == 0:
            return None, None, None, None, None
        
        # Convert lists to tensors
        features = torch.stack(street_data['features'])
        action_labels = torch.stack(street_data['action_labels'])
        bet_size_labels = torch.stack(street_data['bet_size_labels'])
        action_masks = torch.stack(street_data['action_masks'])
        bet_size_masks = street_data['bet_size_bounds']  # Keep as list of lambda functions
        
        return features, action_labels, bet_size_labels, action_masks, bet_size_masks
    
    def save_to_file(self, filepath: str = "training_data.pkl"):
        # Save all training data to a pickle file
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Prepare data for saving (convert tensors to numpy for better compatibility)
        save_data = {
            'training_data': {},
            'blind_training_data': {}
        }
        
        for street in [FLOP, TURN, RIVER]:
            # Save regular training data
            street_data = self.training_data[street]
            save_data['training_data'][street] = {
                'features': [t.numpy() for t in street_data['features']],
                'action_labels': [t.numpy() for t in street_data['action_labels']],
                'bet_size_labels': [t.numpy() for t in street_data['bet_size_labels']],
                'action_masks': [t.numpy() for t in street_data['action_masks']],
                # Note: bet_size_bounds (lambda functions) cannot be pickled, so we skip them
            }
            
            # Save blind training data
            blind_street_data = self.blind_training_data[street]
            save_data['blind_training_data'][street] = {
                'features': [t.numpy() for t in blind_street_data['features']],
                'action_labels': [t.numpy() for t in blind_street_data['action_labels']],
                'bet_size_labels': [t.numpy() for t in blind_street_data['bet_size_labels']],
                'action_masks': [t.numpy() for t in blind_street_data['action_masks']],
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Training data saved to {filepath}")
        print(f"  FLOP: {len(self.training_data[FLOP]['features'])} regular, {len(self.blind_training_data[FLOP]['features'])} blind samples")
        print(f"  TURN: {len(self.training_data[TURN]['features'])} regular, {len(self.blind_training_data[TURN]['features'])} blind samples")
        print(f"  RIVER: {len(self.training_data[RIVER]['features'])} regular, {len(self.blind_training_data[RIVER]['features'])} blind samples")
    
    def load_from_file(self, filepath: str = "training_data.pkl"):
        # Load training data from a pickle file. Returns True if successful, False otherwise
        if not os.path.exists(filepath):
            print(f"Training data file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Clear existing data
            for street in [FLOP, TURN, RIVER]:
                self.training_data[street] = {'features': [], 'action_labels': [], 'bet_size_labels': [], 'action_masks': [], 'bet_size_bounds': []}
                self.blind_training_data[street] = {'features': [], 'action_labels': [], 'bet_size_labels': [], 'action_masks': [], 'bet_size_bounds': []}
            
            # Load regular training data
            for street in [FLOP, TURN, RIVER]:
                if street in save_data.get('training_data', {}):
                    street_data = save_data['training_data'][street]
                    self.training_data[street]['features'] = [torch.tensor(t, dtype=torch.float32) for t in street_data['features']]
                    self.training_data[street]['action_labels'] = [torch.tensor(t, dtype=torch.long) for t in street_data['action_labels']]
                    self.training_data[street]['bet_size_labels'] = [torch.tensor(t, dtype=torch.float32) for t in street_data['bet_size_labels']]
                    self.training_data[street]['action_masks'] = [torch.tensor(t, dtype=torch.float32) for t in street_data['action_masks']]
                    # bet_size_bounds will remain empty (lambda functions cannot be pickled)
            
            # Load blind training data
            for street in [FLOP, TURN, RIVER]:
                if street in save_data.get('blind_training_data', {}):
                    street_data = save_data['blind_training_data'][street]
                    self.blind_training_data[street]['features'] = [torch.tensor(t, dtype=torch.float32) for t in street_data['features']]
                    self.blind_training_data[street]['action_labels'] = [torch.tensor(t, dtype=torch.long) for t in street_data['action_labels']]
                    self.blind_training_data[street]['bet_size_labels'] = [torch.tensor(t, dtype=torch.float32) for t in street_data['bet_size_labels']]
                    self.blind_training_data[street]['action_masks'] = [torch.tensor(t, dtype=torch.float32) for t in street_data['action_masks']]
            
            print(f"Training data loaded from {filepath}")
            print(f"  FLOP: {len(self.training_data[FLOP]['features'])} regular, {len(self.blind_training_data[FLOP]['features'])} blind samples")
            print(f"  TURN: {len(self.training_data[TURN]['features'])} regular, {len(self.blind_training_data[TURN]['features'])} blind samples")
            print(f"  RIVER: {len(self.training_data[RIVER]['features'])} regular, {len(self.blind_training_data[RIVER]['features'])} blind samples")
            return True
            
        except Exception as e:
            print(f"Error loading training data from {filepath}: {e}")
            return False


# Singleton instance of TrainingDataStorage
training_storage_instance = None

def get_training_storage():
    """Get or create the singleton instance of TrainingDataStorage."""
    global training_storage_instance
    if training_storage_instance is None:
        training_storage_instance = TrainingDataStorage()
    return training_storage_instance

