"""
=== SUPERVISED POKER AGENT ===

Neural network implementation for poker decision-making using supervised learning.

MOTIVATION:
Implements PyTorch neural networks for poker action prediction (FOLD/CHECK/CALL/RAISE) and bet sizing.
Includes training infrastructure, model saving/loading, and prediction methods for both single and batch inference.

CLASSES:
- SupervisedPokerAgent(nn.Module) - Main neural network for action prediction
  Methods: __init__(input_dim, hidden_dims, num_actions, dropout), forward(x, action_mask), predict_action(state, action_mask), predict_batch(batch_states, action_mask), save_model(path), load_model(path)
- BetSizeAgent(nn.Module) - Neural network for bet size prediction
  Methods: __init__(input_dim, hidden_dims), forward(x), predict_bet_size(state)
- BaseTrainer - Base class with common training functionality
  Methods: __init__(agent, learning_rate, model_save_path), save_model(path), load_model(path)
- PokerAgentTrainer(BaseTrainer) - Training infrastructure for SupervisedPokerAgent
  Methods: train(train_loader, val_loader, num_epochs, patience), validate(val_loader), evaluate_detailed(val_loader)
- BetSizeTrainer(BaseTrainer) - Training infrastructure for BetSizeAgent
  Methods: train(train_loader, val_loader, num_epochs, patience), validate(val_loader)

FUNCTIONS (signatures):
- initialize_agents() - Initialize and load pre-trained agents
- train_agent_for_street(current_street, street_name, agents, training_storage, is_blind_agent=False) - Train agent for specific street
- main(street=None) - Main training entry point
"""

import torch, numpy as np, os, logging
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod

from AppUtils.files_utils import AGENTS_FILES_DIR
from AppUtils.constants import FLOP, TURN, RIVER
from agent.agent_utils import FEATURES_COUNT, BLIND_FEATURES_COUNT, RAISE, INT_VALUE_TO_STRING_ACTION_MAP
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
HIDDEN_DIMS_ACTION = [256, 128, 64]
HIDDEN_DIMS_BET_SIZE = [128, 64]
DEFAULT_DROPOUT = 0.0
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
DEFAULT_PATIENCE = 15
TRAIN_VAL_SPLIT = 0.8
GRADIENT_CLIP_MAX_NORM = 1.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6
LOG_INTERVAL = 100
PRINT_EPOCH_INTERVAL = 10

# =============================================================================
# LOGGING SETUP
# =============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# =============================================================================
# ACTION NAMES MAPPING
# =============================================================================
action_names = [name.capitalize() for name in INT_VALUE_TO_STRING_ACTION_MAP.values()]
action_names_map = {i: name.capitalize() for i, name in INT_VALUE_TO_STRING_ACTION_MAP.items()}


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================
def validate_input_data(features: torch.Tensor, labels: torch.Tensor) -> None:
    """
    Validate training data before training.
    
    Args:
        features: Feature tensor
        labels: Label tensor
        
    Raises:
        ValueError: If data is invalid
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Features and labels size mismatch: {features.shape[0]} vs {labels.shape[0]}")
    
    if features.shape[0] == 0:
        raise ValueError("Empty dataset provided")
    
    if torch.isnan(features).any():
        raise ValueError("Features contain NaN values")
    
    if torch.isinf(features).any():
        raise ValueError("Features contain infinite values")

class SupervisedPokerAgent(nn.Module):
    """Neural network agent that learns poker strategies from expert data."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 num_actions: int = 4, dropout: float = DEFAULT_DROPOUT):
        # Avoid mutable default argument
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS_ACTION.copy()
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers with batch norm and dropout for training stability
        # first layer is the input layer, then we have 3 hidden layers, and then the output layer
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),      # this take the previous layer and transform it to the hidden dimension
                nn.BatchNorm1d(hidden_dim),           # Normalize activations for training stability
                nn.ReLU(),                            # Non-linear activation
                nn.Dropout(dropout)                   # every time we pass through a layer, we drop out 5% of the features so it won't rely on one feature
            ])
            prev_dim = hidden_dim
        
        # Final output layer: maps the last layer into action probabilities
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers) # create a sequential model with the layers
        
    def forward(self, x: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        # BatchNorm requires batch_size > 1 during training
        # If batch_size == 1 and in training mode, temporarily switch to eval mode
        was_training = self.training
        if was_training and x.size(0) == 1:
            self.eval()
        
        logits = self.network(x)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        
        # Restore training mode if it was changed
        if was_training and x.size(0) == 1:
            self.train()
        
        return logits
    
    def predict_action(self, game_state: torch.Tensor, action_mask: torch.Tensor = None) -> Tuple[int, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            if game_state.dim() == 1:
                game_state = game_state.unsqueeze(0)
            if action_mask is not None and action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            
            logits = self.forward(game_state, action_mask)
            probabilities = torch.softmax(logits, dim=1)
            action_index = torch.argmax(probabilities, dim=1).item()
            
        return action_index, probabilities.squeeze(0)
    
    def get_action_probabilities(self, game_state: torch.Tensor) -> torch.Tensor:
        """
        Get action probability distribution for game state.
        Returns probability tensor for all possible actions.
        """
        self.eval()
        
        with torch.no_grad():
            if game_state.dim() == 1:
                game_state = game_state.unsqueeze(0)
            
            logits = self.forward(game_state)
            probabilities = torch.softmax(logits, dim=1)
            
        return probabilities.squeeze(0)

    def predict_batch(self, batch_states: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Predict action probabilities for a batch of game states.
        Much faster than calling predict_action individually.
        
        Args:
            batch_states: Tensor of shape (batch_size, input_dim)
            action_mask: Optional tensor of shape (batch_size, num_actions) or (num_actions,)
        
        Returns:
            Tensor of shape (batch_size, num_actions) with probabilities
        """
        self.eval()
        with torch.no_grad():
            # Ensure batch dimension
            if batch_states.dim() == 1:
                batch_states = batch_states.unsqueeze(0)
            
            # Handle action mask
            if action_mask is not None:
                if action_mask.dim() == 1:
                    action_mask = action_mask.unsqueeze(0).expand(batch_states.size(0), -1)
            
            logits = self.forward(batch_states, action_mask)
            probabilities = torch.softmax(logits, dim=1)
        
        return probabilities

class BetSizeAgent(nn.Module):
    """Neural network for predicting bet size (regression model)."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS_BET_SIZE.copy()
        
        self.input_dim = input_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer - no activation (linear output for regression)
        # Bet sizes are typically positive, so we apply Softplus for smooth positive output
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())  # Smooth approximation of ReLU, better for gradients
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict_bet_size(self, game_state: torch.Tensor, action_one_hot: torch.Tensor = None) -> float:
        """
        Predict bet size for a given game state.
        
        Args:
            game_state: Feature tensor for the game state
            action_one_hot: Optional one-hot encoded action tensor
            
        Returns:
            Predicted bet size as float
        """
        self.eval()
        with torch.no_grad():
            if game_state.dim() == 1:
                game_state = game_state.unsqueeze(0)
            
            if action_one_hot is not None:
                if action_one_hot.dim() == 1:
                    action_one_hot = action_one_hot.unsqueeze(0)
                model_input = torch.cat([game_state, action_one_hot], dim=1)
            else:
                model_input = game_state
            
            output = self.forward(model_input)
        return output.squeeze().item()

def initialize_agents() -> Dict[str, Any]:
    """
    Initialize supervised poker agents for each street (flop, turn, river).
    Agents are created lazily during training based on actual feature dimensions from data.
    
    Returns:
        Dictionary containing agents dictionary (initially empty) and training storage
    """
    from agent.training_storage import get_training_storage
    training_storage = get_training_storage()
    
    # Agents will be created during training based on actual_input_dim from data
    agents = {}
    
    return {
        'agents': agents,
        'training_storage': training_storage
    }


# =============================================================================
# BASE TRAINER CLASS
# =============================================================================
class BaseTrainer(ABC):
    """
    Base trainer class with common functionality for all trainers.
    
    Provides shared infrastructure for training, validation, model saving/loading,
    and early stopping logic.
    """
    
    def __init__(self, agent: nn.Module, learning_rate: float = DEFAULT_LEARNING_RATE,
                 model_save_path: str = 'model.pth'):
        self.agent = agent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent.to(self.device)
        self.model_save_path = model_save_path
        self.learning_rate = learning_rate
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.agent.parameters(), 
            lr=learning_rate, 
            weight_decay=DEFAULT_WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=SCHEDULER_FACTOR, 
            patience=SCHEDULER_PATIENCE, 
            min_lr=SCHEDULER_MIN_LR
        )
    
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        pass
    
    @abstractmethod
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model. Returns average loss."""
        pass
    
    def _training_loop(self, train_loader: DataLoader, val_loader: DataLoader,
                       num_epochs: int, patience: int) -> Dict[str, List[float]]:
        """
        Common training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if (epoch + 1) % PRINT_EPOCH_INTERVAL == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(self.model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model with optimizer state, scheduler state, and training history.
        
        Args:
            filepath: Path to save the model
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from checkpoint file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {filepath}")

class PokerAgentTrainer(BaseTrainer):
    """
    Handles training, validation, and model management for poker action agent.
    
    Extends BaseTrainer with classification-specific functionality including
    class weighting, accuracy tracking, and detailed evaluation metrics.
    """
    
    def __init__(self, agent: SupervisedPokerAgent, learning_rate: float = DEFAULT_LEARNING_RATE, 
                 model_save_path: str = 'agent.pth', street: int = FLOP):
        super().__init__(agent, learning_rate, model_save_path)
        
        self.street = street
        
        # Class weights: slightly higher weight for RAISE actions
        miss_raise_penalty = 1.0
        class_weights = torch.tensor(
            [1.0, 1.0, 1.0, miss_raise_penalty], 
            dtype=torch.float32
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        
        # Additional accuracy tracking for classification
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
    
    def _extract_batch_data(self, batch_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract features, labels, and action masks from batch.
        Args: batch_data: Tuple of tensors from DataLoader
        Returns: Tuple of (features, labels, action_masks)
        """
        if len(batch_data) < 2:
            raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
        features, labels = batch_data[0], batch_data[1]
        batch_action_masks = batch_data[3] if len(batch_data) > 3 else None
        return features, labels, batch_action_masks
    
    def _apply_action_mask(self, outputs: torch.Tensor, labels: torch.Tensor, 
                           batch_action_masks: Optional[torch.Tensor],
                           features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply action masking and filter invalid samples.
        outputs: Model output logits
        labels: True labels
        batch_action_masks: Action availability masks
        features: Optional features tensor to filter as well
            
        Returns: Filtered outputs, labels, masks, and features (if provided)
        """
        if batch_action_masks is None:
            return outputs, labels, batch_action_masks, features
        
        valid_samples = []
        for i, label in enumerate(labels):
            if batch_action_masks[i][label.item()] != 0:
                valid_samples.append(i)
        
        if len(valid_samples) < len(labels):
            outputs = outputs[valid_samples]
            labels = labels[valid_samples]
            batch_action_masks = batch_action_masks[valid_samples]
            if features is not None:
                features = features[valid_samples]
        
        outputs = outputs.masked_fill(batch_action_masks == 0, float('-inf'))
        return outputs, labels, batch_action_masks, features
    
    # This is for bad river decisions - call with nothing, fold with the nuts, etc..
    def _compute_sample_weights(self, features: torch.Tensor, predicted: torch.Tensor, 
                                 labels: torch.Tensor) -> torch.Tensor:
        """
        Compute sample weights for training.
        Give higher weight (penalty) to errors where model predicted FOLD 
        when hand_strength was high.
        """
        batch_size = features.size(0)
        sample_weights = torch.ones(batch_size, dtype=torch.float32, device=self.device)
        

        has_hand_strength = FEATURES_COUNT[self.street] == features.size(1)
        
        if has_hand_strength:
            hand_strength_idx = 0
            hand_strengths = features[:, hand_strength_idx]
            
            # Find samples where:
            # 1. Model predicted FOLD (predicted == 0)
            # 2. hand_strength > 0.7
            # 3. It's an error (predicted != labels)
            fold_prediction = (predicted == 0)
            call_prediction = (predicted == 2)
            
            is_error = (predicted != labels)

            # Big hand fold penalty
            high_hand_strength = (hand_strengths > 0.8)
            penalty = hand_strengths**3*10            
            penalty_mask = fold_prediction & high_hand_strength & is_error
            sample_weights[penalty_mask] = penalty[penalty_mask]  # 10x penalty for high hand strength fold errors

            # Super big hand - penalty for just calling
            super_hand = (hand_strengths > 0.9)
            super_hand_penalty_mask = super_hand & call_prediction & (self.street == RIVER) & is_error
            sample_weights[super_hand_penalty_mask] = hand_strengths**4*10

            # Very low hand strength in the river - penalty for calling
            low_hand_strength = (hand_strengths < 0.15)
            bad_call_penalty_mask = call_prediction & (self.street == RIVER) & low_hand_strength & is_error
            sample_weights[bad_call_penalty_mask] = 25.0 if hand_strengths < 0.1 else 5.0
        return sample_weights
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train agent for one epoch on training data.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.agent.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        valid_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            features, labels, batch_action_masks = self._extract_batch_data(batch_data)
            
            features = features.to(self.device)
            labels = labels.to(self.device)
            if batch_action_masks is not None:
                batch_action_masks = batch_action_masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.agent(features)
            
            outputs, labels, batch_action_masks, features = self._apply_action_mask(
                outputs, labels, batch_action_masks, features
            )
            
            if len(labels) == 0:
                continue
            
            # Skip batches with size 1 during training (BatchNorm requires batch_size > 1)
            if len(labels) == 1 and self.agent.training:
                continue
            
            # Compute per-sample losses
            per_sample_loss = self.criterion(outputs, labels)
            
            # Get predictions for sample weighting
            _, predicted = torch.max(outputs.data, 1)
            
            # Compute sample weights (penalty for FOLD predictions with high hand_strength)
            # features is now filtered to match labels/predicted size
            sample_weights = self._compute_sample_weights(features, predicted, labels)
            
            # Apply sample weights to losses
            weighted_loss = (per_sample_loss * sample_weights).mean()
            
            if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                continue
            
            weighted_loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
            self.optimizer.step()
            
            total_loss += weighted_loss.item()
            valid_batches += 1
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            if batch_idx % LOG_INTERVAL == 0:
                logger.debug(f'Batch {batch_idx}/{len(train_loader)}, Loss: {weighted_loss.item():.4f}')
        
        if valid_batches == 0:
            return float('inf')
        
        # Store accuracy for this epoch
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        self._last_train_accuracy = accuracy
        
        return total_loss / valid_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Evaluate agent performance on validation data.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.agent.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        valid_batches = 0
        self._last_predictions = []
        self._last_labels = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                features, labels, batch_action_masks = self._extract_batch_data(batch_data)
                
                features = features.to(self.device)
                labels = labels.to(self.device)
                if batch_action_masks is not None:
                    batch_action_masks = batch_action_masks.to(self.device)
                
                outputs = self.agent(features)
                outputs, labels, batch_action_masks, _ = self._apply_action_mask(
                    outputs, labels, batch_action_masks, None
                )
                
                if len(labels) == 0:
                    continue
                
                # For validation, use mean loss (no sample weighting)
                per_sample_loss = self.criterion(outputs, labels)
                loss = per_sample_loss.mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss.item()
                valid_batches += 1
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                self._last_predictions.extend(predicted.cpu().numpy())
                self._last_labels.extend(labels.cpu().numpy())
        
        if valid_batches == 0:
            return float('inf')
        
        # Store accuracy for this epoch
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        self._last_val_accuracy = accuracy
        
        return total_loss / valid_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = DEFAULT_EPOCHS, patience: int = DEFAULT_PATIENCE) -> Dict[str, List[float]]:
        """
        Complete training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Get accuracies from the epoch methods
            train_acc = getattr(self, '_last_train_accuracy', 0.0)
            val_acc = getattr(self, '_last_val_accuracy', 0.0)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            if (epoch + 1) % PRINT_EPOCH_INTERVAL == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(self.model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
        
        # Plot training history
        self._plot_training_history()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def _plot_training_history(self) -> None:
        """Plot training and validation loss over time."""
        epochs = list(range(len(self.train_losses)))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Time")
        plt.legend()
        plt.close()
    
    def evaluate_detailed(self, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Perform detailed evaluation with confusion matrix and per-class metrics.
        Args: val_loader: Validation data loader
        Returns: Dictionary with detailed evaluation metrics
        """
        from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
        
        # Run validation to get predictions
        val_loss = self.validate(val_loader)
        val_acc = getattr(self, '_last_val_accuracy', 0.0)
        predictions = np.array(self._last_predictions)
        labels = np.array(self._last_labels)
        
        cm = confusion_matrix(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Get actual number of classes in the data
        num_classes = cm.shape[0]
        present_action_names = action_names[:num_classes]
        
        logger.info("\n" + "="*60)
        logger.info("DETAILED EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Overall Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        logger.info(f"Validation Loss: {val_loss:.4f}\n")
        
        logger.info("CONFUSION MATRIX:")
        logger.info("           " + "  ".join(f"{name:>8}" for name in present_action_names))
        for i, true_class in enumerate(present_action_names):
            row = f"{true_class:>10} " + " ".join(f"{cm[i,j]:>8}" for j in range(num_classes))
            logger.info(row)
        
        logger.info(f"\n{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10} {'Support':<10}")
        logger.info("-" * 65)
        for i, class_name in enumerate(present_action_names):
            logger.info(f"{class_name:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} "
                       f"{per_class_accuracy[i]:<10.4f} {support[i]:<10.0f}")
        
        logger.info("\nCLASSIFICATION REPORT:")
        logger.info(classification_report(labels, predictions, target_names=present_action_names))
        
        return {
            'overall_accuracy': val_acc,
            'validation_loss': val_loss,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class_accuracy': per_class_accuracy,
            'support': support,
            'predictions': predictions,
            'labels': labels,
            'action_names': action_names
        }
    
    def plot_confusion_matrix(self, val_loader: DataLoader, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create and display a visual confusion matrix for the agent.
        Args:
            val_loader: Validation data loader
            save_path: Optional path to save the plot
        Returns: Tuple of (confusion_matrix, normalized_confusion_matrix)
        """
        from sklearn.metrics import confusion_matrix
        
        # Run validation to get predictions
        self.validate(val_loader)
        val_acc = getattr(self, '_last_val_accuracy', 0.0)
        predictions = np.array(self._last_predictions)
        labels = np.array(self._last_labels)
        local_action_names = ['Fold', 'Check', 'Call', 'Raise']
        
        cm = confusion_matrix(labels, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Get actual number of classes in the data
        num_classes = cm.shape[0]
        present_action_names = local_action_names[:num_classes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.figure.colorbar(im1, ax=ax1, label='Count')
        ax1.set(xticks=np.arange(num_classes),
               yticks=np.arange(num_classes),
               xticklabels=present_action_names, yticklabels=present_action_names,
               title=f'Confusion Matrix (Absolute)\nAccuracy: {val_acc:.2%}',
               ylabel='True Action', xlabel='Predicted Action')
        
        thresh = cm.max() / 2. if cm.max() > 0 else 0.5
        for i in range(num_classes):
            for j in range(num_classes):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax2.figure.colorbar(im2, ax=ax2, label='Percentage (%)')
        ax2.set(xticks=np.arange(num_classes),
               yticks=np.arange(num_classes),
               xticklabels=present_action_names, yticklabels=present_action_names,
               title='Confusion Matrix (Normalized %)',
               ylabel='True Action', xlabel='Predicted Action')
        
        thresh = cm_normalized.max() / 2. if cm_normalized.max() > 0 else 0.5
        for i in range(num_classes):
            for j in range(num_classes):
                ax2.text(j, i, format(cm_normalized[i, j], '.1f'),
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        if save_path:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
        return cm, cm_normalized
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model with optimizer state, scheduler state, and training history.
        Overrides base to include accuracy history.
        
        Args: filepath: Path to save the model
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from checkpoint file.
        Overrides base to include accuracy history.
        Args: filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        logger.info(f"Model loaded from {filepath}")

class BetSizeTrainer(BaseTrainer):
    """
    Handles training for bet size prediction (regression model).
    
    Extends BaseTrainer with regression-specific functionality including
    MSE loss and constraint-based penalties for valid bet size bounds.
    """
    
    def __init__(self, agent: BetSizeAgent, learning_rate: float = DEFAULT_LEARNING_RATE,
                 model_save_path: str = '_bet_size_agent.pth'):
        super().__init__(agent, learning_rate, model_save_path)
        self.criterion = nn.MSELoss()
    
    def _extract_batch_data(self, batch_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract features, actions, bet_sizes, and optional bounds from batch.
        Args: batch_data: Tuple of tensors from DataLoader
        Returns: Tuple of (features, actions, bet_sizes, min_bounds, max_bounds)
        """
        if len(batch_data) == 5:
            features, actions, bet_sizes, min_bounds, max_bounds = batch_data
            return features, actions, bet_sizes, min_bounds, max_bounds
        else:
            features, actions, bet_sizes = batch_data
            return features, actions, bet_sizes, None, None
    
    def _compute_constraint_loss(self, outputs: torch.Tensor, 
                                  min_bounds: Optional[torch.Tensor], 
                                  max_bounds: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute penalty for outputs outside valid bounds.
        
        Args:
            outputs: Model predictions
            min_bounds: Minimum valid bet sizes
            max_bounds: Maximum valid bet sizes
            
        Returns: Constraint loss tensor
        """
        if min_bounds is None or max_bounds is None:
            return torch.tensor(0.0, device=self.device)
        
        below_min = torch.clamp(min_bounds - outputs, min=0.0)
        above_max = torch.clamp(outputs - max_bounds, min=0.0)
        return (below_min.mean() + above_max.mean()) * 0.5
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train bet size agent for one epoch.
        Args: train_loader: Training data loader
        Returns: Average loss for the epoch
        """
        self.agent.train()
        total_loss = 0.0
        valid_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            features, actions, bet_sizes, min_bounds, max_bounds = self._extract_batch_data(batch_data)
            
            # Skip batches with size 1 (BatchNorm requires batch_size > 1 in training mode)
            if features.size(0) == 1:
                continue
            
            features = features.to(self.device)
            actions = actions.to(self.device)
            bet_sizes = bet_sizes.to(self.device)
            if min_bounds is not None:
                min_bounds = min_bounds.to(self.device)
            if max_bounds is not None:
                max_bounds = max_bounds.to(self.device)
            
            # Create one-hot encoding for actions
            action_one_hot = torch.zeros(actions.size(0), 4, device=self.device)
            action_one_hot.scatter_(1, actions.unsqueeze(1), 1)
            
            # Concatenate features with action one-hot encoding
            model_input = torch.cat([features, action_one_hot], dim=1)
            
            self.optimizer.zero_grad()
            outputs = self.agent(model_input).squeeze()
            
            # Base MSE loss
            mse_loss = self.criterion(outputs, bet_sizes.squeeze())
            
            # Add penalty for outputs outside valid bounds
            constraint_loss = self._compute_constraint_loss(outputs, min_bounds, max_bounds)
            
            total_loss_batch = mse_loss + constraint_loss
            
            if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                continue
            
            total_loss_batch.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            valid_batches += 1
            
            if batch_idx % LOG_INTERVAL == 0:
                constraint_val = constraint_loss.item() if isinstance(constraint_loss, torch.Tensor) else constraint_loss
                logger.debug(f'Bet Size Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss_batch.item():.4f} '
                            f'(MSE: {mse_loss.item():.4f}, Constraint: {constraint_val:.4f})')
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate bet size agent.
        Args: val_loader: Validation data loader
        Returns: Average validation loss
        """
        self.agent.eval()
        total_loss = 0.0
        valid_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                features, actions, bet_sizes, min_bounds, max_bounds = self._extract_batch_data(batch_data)
                
                # Skip batches with size 1 (BatchNorm requires batch_size > 1 in training mode)
                # In eval mode it should work, but we skip for consistency
                if features.size(0) == 1:
                    continue
                
                features = features.to(self.device)
                actions = actions.to(self.device)
                bet_sizes = bet_sizes.to(self.device)
                if min_bounds is not None:
                    min_bounds = min_bounds.to(self.device)
                if max_bounds is not None:
                    max_bounds = max_bounds.to(self.device)
                
                action_one_hot = torch.zeros(actions.size(0), 4, device=self.device)
                action_one_hot.scatter_(1, actions.unsqueeze(1), 1)
                model_input = torch.cat([features, action_one_hot], dim=1)
                
                outputs = self.agent(model_input).squeeze()
                
                # Base MSE loss
                mse_loss = self.criterion(outputs, bet_sizes.squeeze())
                
                # Add penalty for outputs outside valid bounds
                constraint_loss = self._compute_constraint_loss(outputs, min_bounds, max_bounds)
                
                total_loss_batch = mse_loss + constraint_loss
                
                if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                    continue
                
                total_loss += total_loss_batch.item()
                valid_batches += 1
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = DEFAULT_EPOCHS, patience: int = DEFAULT_PATIENCE) -> Dict[str, List[float]]:
        """
        Complete training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns: Training history dictionary
        """
        logger.info(f"Training bet size agent on device: {self.device}")
        return self._training_loop(train_loader, val_loader, num_epochs, patience)

# =============================================================================
# TRAINING DATA PREPARATION UTILITIES
# =============================================================================
def _log_action_distribution(action_labels: torch.Tensor, action_masks: torch.Tensor) -> None:
    """
    Log the distribution of actions and available actions in training data.
    
    Args:
        action_labels: Tensor of action labels
        action_masks: Tensor of action availability masks
    """
    unique_actions, counts = torch.unique(action_labels, return_counts=True)
    logger.info("\nAction distribution:")
    for action_idx, count in zip(unique_actions, counts):
        action_name = action_names_map.get(action_idx.item(), f'Unknown({action_idx.item()})')
        percentage = count.item() / len(action_labels) * 100
        logger.info(f"  {action_name}: {count.item()} samples ({percentage:.2f}%)")
    
    logger.info("\nAction mask distribution (how many samples have each action available):")
    for i, action_name in enumerate(action_names):
        available_count = (action_masks[:, i] > 0).sum().item()
        percentage = available_count / len(action_labels) * 100
        logger.info(f"  {action_name}: {available_count} samples ({percentage:.2f}%)")


def _prepare_data_loaders(features: torch.Tensor, action_labels: torch.Tensor, 
                          bet_size_labels: torch.Tensor, action_masks: torch.Tensor,
                          batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Prepare train and validation data loaders.
    
    Args:
        features: Feature tensor
        action_labels: Action label tensor
        bet_size_labels: Bet size label tensor
        action_masks: Action mask tensor
        batch_size: Batch size for data loaders
        
    Returns: Tuple of (train_loader, val_loader, train_size, val_size)
    """
    dataset = TensorDataset(features, action_labels, bet_size_labels, action_masks)
    train_size = int(TRAIN_VAL_SPLIT * features.shape[0])
    val_size = features.shape[0] - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_size, val_size


def _create_agents_for_training(agents: Dict[int, Dict[str, Any]], current_street: int,
                                 actual_input_dim: int, is_blind_agent: bool) -> Tuple[SupervisedPokerAgent, BetSizeAgent]:
    """
    Create or retrieve agents for training.
    
    Args:
        agents: Dictionary of existing agents
        current_street: Current street being trained
        actual_input_dim: Actual input dimension from data
        is_blind_agent: Whether training blind agent
        
    Returns: Tuple of (action_agent, bet_size_agent)
    """
    # Create agents based on actual_input_dim from data (not predefined constants)
    # This ensures agents match the actual feature dimensions, whether regular or blind
    agent_key = f"{current_street}_{'blind' if is_blind_agent else 'regular'}"
    
    if agent_key in agents and agents[agent_key]['action_agent'].input_dim == actual_input_dim:
        # Reuse existing agent if dimensions match
        action_agent = agents[agent_key]['action_agent']
        bet_size_agent = agents[agent_key]['bet_size_agent']
    else:
        # Create new agent with correct dimensions
        action_agent = SupervisedPokerAgent(
            input_dim=actual_input_dim,
            hidden_dims=HIDDEN_DIMS_ACTION.copy(),
            num_actions=4,
            dropout=DEFAULT_DROPOUT
        )
        bet_size_input_dim = actual_input_dim + 4
        bet_size_agent = BetSizeAgent(
            input_dim=bet_size_input_dim,
            hidden_dims=HIDDEN_DIMS_BET_SIZE.copy(),
            dropout=DEFAULT_DROPOUT
        )
        
        # Store for potential reuse
        agents[agent_key] = {
            'action_agent': action_agent,
            'bet_size_agent': bet_size_agent
        }
    
    return action_agent, bet_size_agent


def _train_action_agent(action_agent: SupervisedPokerAgent, train_loader: DataLoader,
                        val_loader: DataLoader, model_path: str, 
                        street: int, street_name: str, is_blind_agent: bool) -> PokerAgentTrainer:
    """
    Train the action prediction agent.
    Args:
        action_agent: Agent to train
        train_loader: Training data loader
        val_loader: Validation data loader
        model_path: Path to save model
        street: Street number
        street_name: Human-readable street name
        is_blind_agent: Whether training blind agent
        
    Returns: Trained PokerAgentTrainer instance
    """
    agent_type = "blind " if is_blind_agent else ""
    
    trainer = PokerAgentTrainer(
        action_agent, 
        learning_rate=DEFAULT_LEARNING_RATE, 
        model_save_path=model_path, 
        street=street
    )
    
    logger.info("Starting action agent training...")
    history = trainer.train(train_loader, val_loader, num_epochs=DEFAULT_EPOCHS)
    
    logger.info(f"\nTraining completed for {agent_type}{street_name}!")
    if history['val_accuracies']:
        logger.info(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")
    logger.info(f"Model saved as: {model_path}")
    
    logger.info("\nPerforming detailed evaluation...")
    trainer.evaluate_detailed(val_loader)
    
    return trainer


def _extract_bet_size_bounds(bet_size: torch.Tensor, mask_func: Optional[Callable]) -> Tuple[bool, float, float]:
    """
    Extract min/max bounds from bet size mask function.
    
    Args:
        bet_size: Bet size tensor
        mask_func: Mask function for validation
        
    Returns:
        Tuple of (is_valid, min_bound, max_bound)
    """
    bet_size_val = bet_size.item() if isinstance(bet_size, torch.Tensor) else bet_size
    
    if mask_func is None:
        return True, 0.0, 2.0
    
    if not mask_func(bet_size_val):
        return False, 0.0, 0.0
    
    # Try to get bounds from attributes
    if hasattr(mask_func, 'min_size') and hasattr(mask_func, 'max_size'):
        return True, mask_func.min_size, mask_func.max_size
    
    # Fallback: extract bounds by testing values
    test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
    
    min_val = 0.0
    for test_val in test_values:
        if mask_func(test_val):
            min_val = test_val
            break
    
    max_val = 2.0
    for test_val in reversed(test_values):
        if mask_func(test_val):
            max_val = test_val
            break
    
    return True, min_val, max_val


def _prepare_bet_size_data(features: torch.Tensor, action_labels: torch.Tensor,
                           bet_size_labels: torch.Tensor, bet_size_masks: List,
                           batch_size: int = DEFAULT_BATCH_SIZE) -> Optional[Tuple[DataLoader, DataLoader, int, int]]:
    """
    Prepare bet size training data from RAISE actions only.
    
    Args:
        features: Feature tensor
        action_labels: Action label tensor
        bet_size_labels: Bet size label tensor
        bet_size_masks: List of bet size mask functions
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, train_size, val_size) or None if no valid data
    """
    # Filter only RAISE actions
    raise_mask = action_labels == RAISE
    raise_indices = torch.where(raise_mask)[0]
    
    if len(raise_indices) == 0:
        return None
    
    logger.info(f"Found {len(raise_indices)} RAISE actions out of {len(action_labels)} total actions")
    
    raise_features = features[raise_indices]
    raise_actions = action_labels[raise_indices]
    raise_bet_sizes = bet_size_labels[raise_indices]
    raise_bet_size_masks = [bet_size_masks[i] for i in raise_indices.cpu().numpy()]
    
    # Filter valid bet sizes and extract bounds
    valid_indices = []
    min_bounds = []
    max_bounds = []
    
    for i, (bet_size, mask_func) in enumerate(zip(raise_bet_sizes, raise_bet_size_masks)):
        is_valid, min_bound, max_bound = _extract_bet_size_bounds(bet_size, mask_func)
        if is_valid:
            valid_indices.append(i)
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)
    
    if len(valid_indices) == 0:
        return None
    
    logger.info(f"Found {len(valid_indices)} valid bet sizes out of {len(raise_indices)} RAISE actions")
    
    # Filter to only valid samples
    valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
    raise_features = raise_features[valid_indices_tensor]
    raise_actions = raise_actions[valid_indices_tensor]
    raise_bet_sizes = raise_bet_sizes[valid_indices_tensor]
    min_bounds_tensor = torch.tensor(min_bounds, dtype=torch.float32)
    max_bounds_tensor = torch.tensor(max_bounds, dtype=torch.float32)
    
    # Ensure bet_sizes has shape (N, 1) for consistency
    if raise_bet_sizes.dim() == 1:
        raise_bet_sizes = raise_bet_sizes.unsqueeze(1)
    
    # Create dataset
    bet_size_dataset = TensorDataset(
        raise_features, raise_actions, raise_bet_sizes, 
        min_bounds_tensor, max_bounds_tensor
    )
    bet_train_size = int(TRAIN_VAL_SPLIT * len(bet_size_dataset))
    bet_val_size = len(bet_size_dataset) - bet_train_size
    
    if bet_train_size == 0 or bet_val_size == 0:
        return None
    
    bet_train_dataset, bet_val_dataset = random_split(bet_size_dataset, [bet_train_size, bet_val_size])
    
    bet_train_loader = DataLoader(bet_train_dataset, batch_size=batch_size, shuffle=True)
    bet_val_loader = DataLoader(bet_val_dataset, batch_size=batch_size, shuffle=False)
    
    return bet_train_loader, bet_val_loader, bet_train_size, bet_val_size


def _train_bet_size_agent(bet_size_agent: BetSizeAgent, train_loader: DataLoader,
                          val_loader: DataLoader, model_path: str,
                          street_name: str, is_blind_agent: bool) -> None:
    """
    Train the bet size prediction agent.
    
    Args:
        bet_size_agent: Agent to train
        train_loader: Training data loader
        val_loader: Validation data loader
        model_path: Path to save model
        street_name: Human-readable street name
        is_blind_agent: Whether training blind agent
    """
    agent_type = "blind " if is_blind_agent else ""
    
    bet_size_trainer = BetSizeTrainer(
        bet_size_agent, 
        learning_rate=DEFAULT_LEARNING_RATE,
        model_save_path=model_path
    )
    
    logger.info("Starting bet size training...")
    bet_history = bet_size_trainer.train(train_loader, val_loader, num_epochs=DEFAULT_EPOCHS)
    
    logger.info(f"\nBet size training completed for {agent_type}{street_name}!")
    if bet_history['val_losses']:
        logger.info(f"Final validation loss: {bet_history['val_losses'][-1]:.4f}")
    logger.info(f"Bet size model saved as: {model_path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def train_agent_for_street(current_street: int, street_name: str, 
                           agents: Dict[int, Dict[str, Any]], 
                           training_storage: Any,
                           is_blind_agent: bool = False) -> Optional[Dict[str, Any]]:
    """
    Train agent for a specific street.
    
    Args:
        current_street: Street number (FLOP=1, TURN=2, RIVER=3)
        street_name: Human readable street name
        agents: Dictionary of agents per street
        training_storage: Storage containing training data
        is_blind_agent: Whether to train blind-specific agent
        
    Returns:
        Training history dictionary or None if no data
    """
    agent_prefix = "blind_" if is_blind_agent else ""
    agent_type = "Blind " if is_blind_agent else ""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {agent_type}Agent for {street_name.capitalize()}")
    logger.info(f"{'='*60}")
    
    # Load training data
    features, action_labels, bet_size_labels, action_masks, bet_size_masks = \
        training_storage.get_training_data(current_street, is_blind_agent=is_blind_agent)
    
    if features is None:
        logger.info(f"No training data found for {agent_type.lower()}{street_name}. Skipping...")
        return None
    
    # Validate input data
    try:
        validate_input_data(features, action_labels)
    except ValueError as e:
        logger.error(f"Invalid training data: {e}")
        return None
    
    logger.info(f"Loaded {features.shape[0]} training samples")
    
    # Log action distribution
    _log_action_distribution(action_labels, action_masks)
    
    # Prepare data loaders
    train_loader, val_loader, train_size, val_size = _prepare_data_loaders(
        features, action_labels, bet_size_labels, action_masks
    )
    logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Get actual input dimension and create agents
    actual_input_dim = features.shape[1]
    action_agent, bet_size_agent = _create_agents_for_training(
        agents, current_street, actual_input_dim, is_blind_agent
    )
    logger.info(f"Using agent with input_dim={actual_input_dim}")
    
    # Ensure output directory exists
    os.makedirs(AGENTS_FILES_DIR, exist_ok=True)
    
    # Train action agent
    model_filename = os.path.join(AGENTS_FILES_DIR, f"{agent_prefix}{street_name}_agent.pth")
    trainer = _train_action_agent(
        action_agent, train_loader, val_loader, model_filename,
        current_street, street_name, is_blind_agent
    )
    
    # Plot confusion matrix
    confusion_matrix_path = os.path.join(AGENTS_FILES_DIR, f"confusion_matrix_{agent_prefix}{street_name}.png")
    trainer.plot_confusion_matrix(val_loader, save_path=confusion_matrix_path)
    
    # Train bet size agent
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Bet Size Agent for {agent_type}{street_name.capitalize()}")
    logger.info(f"{'='*60}")
    
    '''
    bet_size_data = _prepare_bet_size_data(
        features, action_labels, bet_size_labels, bet_size_masks
    )
    
    if bet_size_data is None:
        logger.info(f"No valid bet size data found for {agent_type.lower()}{street_name}. Skipping bet size training...")
    else:
        bet_train_loader, bet_val_loader, bet_train_size, bet_val_size = bet_size_data
        logger.info(f"Bet size training samples: {bet_train_size}, Validation samples: {bet_val_size}")
        
        bet_size_model_filename = os.path.join(AGENTS_FILES_DIR, f"{agent_prefix}{street_name}_bet_size_agent.pth")
        _train_bet_size_agent(
            bet_size_agent, bet_train_loader, bet_val_loader, 
            bet_size_model_filename, street_name, is_blind_agent
        )
    '''
    return {'trainer': trainer}
    
def main(street: Optional[int] = None, load_from_file: bool = True, data_filepath: str = "data/training_data.pkl") -> None:
    # Main training script: load data from TrainingDataStorage, create agent, train supervised model
    logger.info("Starting supervised poker agent training...")
    
    agents_info = initialize_agents()
    agents = agents_info['agents']
    training_storage = agents_info['training_storage']
    
    # Try to load training data from file if requested
    if load_from_file:
        from agent.parse_data_to_agent import load_training_data_from_file
        if load_training_data_from_file(data_filepath):
            logger.info(f"Successfully loaded training data from {data_filepath}")
        else:
            logger.warning(f"Could not load training data from {data_filepath}. Make sure to run parse_sessions() first or provide valid data file.")
            logger.warning("Continuing with in-memory training data (if any exists)...")
    
    streets_to_train = [FLOP, TURN, RIVER] if street is None else [street]
    street_names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
    
    for current_street in streets_to_train:
        street_name = street_names.get(current_street)
        
        if street_name is None:
            logger.warning(f"Unknown street: {current_street}. Skipping...")
            continue
        
        train_agent_for_street(current_street, street_name, agents, training_storage, is_blind_agent=False)
        train_agent_for_street(current_street, street_name, agents, training_storage, is_blind_agent=True)
    
    logger.info("\nAll training completed!")

if __name__ == "__main__":
    main()
