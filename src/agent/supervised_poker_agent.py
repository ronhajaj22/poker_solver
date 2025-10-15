import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, List
from agent.card_encoder import CardEncoder
from .data_preparation import PokerDataProcessor
from AppUtils import agent_utils
from AppUtils.constants import FLOP, TURN, RIVER
import matplotlib.pyplot as plt
import pdb

class SupervisedPokerAgent(nn.Module):
    """
    Neural network agent that learns poker strategies from expert data.
    Takes game state as input, outputs action probabilities for decision making.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 num_actions: int = 4, dropout: float = 0.05):
        """
        Initialize neural network architecture for poker decision making.
        input_dim: Size of game state feature vector
        hidden_dims: Layer sizes for deep network
        num_actions: Number of possible actions (fold, check, call, bet, raise, all_in)
        dropout: Regularization rate to prevent overfitting
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        # Build deep neural network layer by layer
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
        
        # Optional card encoder for advanced card representations
        self.card_encoder = CardEncoder(embedding_dim=32)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through neural network with action masking.
        Input: game state tensor (batch_size, input_dim)
        Output: action logits (batch_size, num_actions) with masked unavailable actions
        """
        logits = self.network(x)
        
        # TODO: Add action masking logic here
        # You will implement the logic to determine available actions
        # Use is_fold_call_possible variable to control masking
        
        return logits
    
    def predict_action(self, game_state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Predict best action for given game state.
        Returns: (action_index, action_probabilities) for decision making.
        """
        self.eval()  # Set to evaluation mode for inference
        
        with torch.no_grad():
            # Add batch dimension if single game state provided
            if game_state.dim() == 1:
                game_state = game_state.unsqueeze(0)
            
            # Get raw action logits from network
            logits = self.forward(game_state)
            
            # Convert logits to probabilities using softmax
            probabilities = torch.softmax(logits, dim=1)
            
            # Select action with highest probability
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

class PokerAgentTrainer:
    """
    Handles training, validation, and model management for poker agent.
    Implements supervised learning with early stopping and model checkpointing.
    """
    
    def __init__(self, agent: SupervisedPokerAgent, learning_rate: float = 0.001, model_save_path: str = 'best_poker_agent.pth'):
        """
        Initialize trainer with agent, optimizer, and training history.
        learning_rate: Step size for gradient descent optimization
        model_save_path: Path to save the best model during training
        """
        self.agent = agent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent.to(self.device)
        
        # Loss function: cross-entropy for multi-class action classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer: Adam for adaptive learning rate with weight decay (L2 regularization)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Model save path for street-specific models
        self.model_save_path = model_save_path
        
        # Training history for monitoring  progress
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train agent for one epoch on training data.
        Returns: (average_loss, accuracy) for monitoring progress.
        """
        self.agent.train()  # Set to training mode
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Process each batch of training data
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move data to GPU/CPU device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass: compute predictions
            self.optimizer.zero_grad()
            outputs = self.agent(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass: compute gradients and update weights
            loss.backward()
            self.optimizer.step()
            
            # Track training statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate epoch statistics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate agent performance on validation data.
        Returns: (average_loss, accuracy) without updating weights.
        """
        self.agent.eval()  # Set to evaluation mode
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # For confusion matrix and detailed metrics
        all_predictions = []
        all_labels = []
        
        # Evaluate without computing gradients
        with torch.no_grad():
            for features, labels in val_loader:
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass only
                outputs = self.agent(features)
                loss = self.criterion(outputs, labels)
                
                # Track validation statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Store predictions and labels for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation statistics
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 100, patience: int = 15) -> Dict: # patience detect how many epochs we wait before we decide there is no improvement
        """
        Complete training loop with early stopping.
        Trains agent, validates performance, saves best model.
        Returns training history for analysis.
        """
        print(f"Training on device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop: iterate through epochs
        for epoch in range(num_epochs):
            if (epoch+1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train on training data
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate on validation data
            val_loss, val_acc, predictions, labels = self.validate(val_loader)
            
            # Store training history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            if (epoch+1) % 10 == 0:
                # Print epoch results
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping: save best model and stop if no improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(self.model_save_path)  # Save best model with street-specific path
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        epochs = list(range(len(self.train_losses)))
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Time")
        plt.legend()
        plt.show()
        
        # Return training history for analysis
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate_detailed(self, val_loader: DataLoader) -> Dict:
        """
        Perform detailed evaluation with confusion matrix and per-class metrics.
        Returns comprehensive evaluation results.
        """
        import numpy as np
        from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
        
        # Get predictions and labels
        val_loss, val_acc, predictions, labels = self.validate(val_loader)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Action class names
        action_names = ['Fold', 'Check', 'Call', 'Raise']
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
        
        # Classification report
        class_report = classification_report(labels, predictions, target_names=action_names, output_dict=True)
        
        # Calculate per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Print detailed results
        print("\n" + "="*60)
        print("DETAILED EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"Validation Loss: {val_loss:.4f}")
        print()
        
        print("CONFUSION MATRIX:")
        print("Predicted →")
        print("Actual ↓")
        print("           " + "  ".join(f"{name:>8}" for name in action_names))
        for i, true_class in enumerate(action_names):
            row = f"{true_class:>10} "
            for j in range(len(action_names)):
                row += f"{cm[i,j]:>8} "
            print(row)
        print()
        
        print("hero_cards")
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10} {'Support':<10}")
        print("-" * 65)
        for i, class_name in enumerate(action_names):
            print(f"{class_name:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {per_class_accuracy[i]:<10.4f} {support[i]:<10.0f}")
        print()
        
        print("CLASSIFICATION REPORT:")
        print(classification_report(labels, predictions, target_names=action_names))
        
        # Return detailed results
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
    
    def save_model(self, filepath: str):
        """Save trained model with optimizer state and training history."""
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from checkpoint file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Model loaded from {filepath}")

def main(street: int = None):
    """
    Main training script: load data, create agent, train supervised model.
    Demonstrates complete pipeline from raw data to trained poker agent.
    """
    print("Starting supervised poker agent training...")
    
    # If no street specified, train all streets
    if street is None:
        streets_to_train = [FLOP, TURN, RIVER]  # flop, turn, river
    else:
        streets_to_train = [street]
    
    for current_street in streets_to_train:
        street_names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
        print(f"\n{'='*60}")
        print(f"Training for {street_names[current_street].capitalize}")
        print(f"{'='*60}")
        
        # Initialize data processor for converting poker hands to training data
        processor = PokerDataProcessor()
        
        # Load and process poker hand data into training tensors
        print("Loading and processing data...")
        
        street_name = street_names.get(current_street)
        data_path = f'src/gg_hands_parser/create_hero/parsed_hands/{street_name}_parsed_hands.json'
        
        try:
            features, labels = processor.load_and_process_data(data_path)
            print(f"features: {features}")
            print(f"labels: {labels}")
        except FileNotFoundError:
            print(f"Data file not found: {data_path}")
            continue
        
        # Create data loaders for batch training
        train_loader, val_loader = processor.create_data_loaders(features, labels, batch_size=64)
        
        
        # Initialize neural network agent with appropriate dimensions
        input_dim = features.shape[1]
        num_actions = len(torch.unique(labels))
        agent = SupervisedPokerAgent(input_dim=input_dim, num_actions=num_actions)
        
        # Initialize trainer with street-specific model save path
        model_filename = f"best_poker_agent_{street_name}.pth"
        trainer = PokerAgentTrainer(agent, learning_rate=0.001, model_save_path=model_filename)
        
        # Train agent on expert poker data
        print("Starting training...")
        history = trainer.train(train_loader, val_loader, num_epochs=30)
        
        print(f"Training completed for street {current_street} ({street_name})!")
        print(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")
        print(f"Model saved as: {model_filename}")
        
        # Perform detailed evaluation
        print("\nPerforming detailed evaluation...")
        detailed_results = trainer.evaluate_detailed(val_loader)
    
    print("\nAll training completed!")

if __name__ == "__main__":
    main() 