import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, List
from agent.card_encoder import CardEncoder
from AppUtils import agent_utils
from AppUtils.agent_utils import FOLD, CALL, CHECK, RAISE
from AppUtils.constants import FLOP, TURN, RIVER
import agent.data_prep_new as processor
import matplotlib.pyplot as plt
import pdb

street_names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}

class PokerAgent(nn.Module):
    def __init__(self, n_features: int, hidden_dim_layers: List[int] = [512, 256, 64], n_actions: int = 4, dropout: float = 0.1):
        super(PokerAgent, self).__init__()
        all_dimensions = [n_features] + hidden_dim_layers + [n_actions]
        
        # TODO - check this line
        self.attention = nn.Linear(n_features, n_features)
        
        layers = []
        for i in range(len(all_dimensions) - 2):
            # this is a function that takes the previous layer and transform it to the hidden dimension using weights
            layers.append(nn.Linear(all_dimensions[i], all_dimensions[i+1]))
            if i+1 == len(all_dimensions) - 1: # if it's the last layer, we don't need to add batch norm and dropout
                print("last layer")
                break;
            layers.append(nn.BatchNorm1d(all_dimensions[i+1])) # Normalize activations for training stability
            layers.append(nn.ReLU()) # this is the function we use - Non-linear activation
            layers.append(nn.Dropout(dropout)) # every time we pass through a layer, we drop out 10% of the features so it won't rely on one feature
        

        # same as self.network = nn.Sequential(layer1, layer2, layer3, layer4) each layer gets as an input the output of the previous layer
        self.network = nn.Sequential(*layers) 

    def forward(self, x: torch.Tensor, action_mask=None) -> torch.Tensor:
         # 1. Attention לתכונות
        attention_weights = torch.softmax(self.attention(x), dim=1)
        weighted_features = x * attention_weights
        
        # final result
        logits = self.network(weighted_features)
       # print(f"logits: {logits}")
        
        if action_mask is not None:
            logits = logits.masked_fill(action_mask==0, -1e9) # switch 0 with -1e9
        
        return logits

    # this function is the main function that trains the model
    # train_loader is the data we use to train the model
    # val_loader is the data we use to evaluate the model
    # num_epochs is the number of times we will go through the data
    # patience is the number of epochs with no improvement before we stop
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 100, patience: int = 15):
        
        # create a loss-function
        self.criterion = nn.CrossEntropyLoss()
        # this is a function that updates the weights of the model
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        patience_counter = 0
        all_train_loss = 0
        all_val_loss = 0

        # num_epochs is the number of times we will go through the data
        for epoch in range(num_epochs):
            train_loss = self.model_train(train_loader)
            val_loss = self.model_eval(val_loader)
            all_train_loss += train_loss
            all_val_loss += val_loss
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 15 epochs with no improvement, we stop
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Batch avg train loss: {round(all_train_loss/num_epochs, 2)}")
        print(f"Batch avg val loss: {round(all_val_loss/num_epochs, 2)}")
        
        # Get final predictions for evaluation
        final_val_result = self.model_eval(val_loader)
        if isinstance(final_val_result, tuple):
            val_loss, final_predictions, final_targets = final_val_result
        else:
            # If only loss is returned, run evaluation again to get predictions
            val_loss = final_val_result
            print("Running final evaluation to get predictions...")
            # We need to modify model_eval to always return predictions
            final_predictions = []
            final_targets = []
            self.eval()
            with torch.no_grad():
                for batch_features, batch_actions in val_loader:
                    logits = self.forward(batch_features)
                    predictions = torch.argmax(logits, dim=1)
                    final_predictions.extend(predictions.cpu().numpy())
                    final_targets.extend(batch_actions.cpu().numpy())
        
        # Generate confusion matrix and detailed metrics - same format as supervised_poker_agent
        self.generate_evaluation_report(final_targets, final_predictions)

    def model_train(self, train_loader: DataLoader):
        self.train()
        # every batch is a set of features and actions (set of hands)
        total_epoch_loss = 0
        batch_counter = 0
        for batch_features, batch_actions in train_loader:
            self.optimizer.zero_grad()
            # logits is the output of the model
            logits = self.forward(batch_features) 
            base_loss = self.criterion(logits, batch_actions)
            # custom penalty for call/fold vs check
            custom_penalty = self.calculate_custom_penalty(logits, batch_actions)
            total_loss = base_loss + custom_penalty
            
            # Total loss - save the number for printing
            batch_loss = total_loss

            # backpropagation - the model change the weights
            total_loss.backward()
            # update the weights
            self.optimizer.step()
            
        #    print(f"total loss in this batch: {batch_loss}")

            total_epoch_loss += batch_loss
            batch_counter += 1
        
        avg_loss_per_batch = round((total_epoch_loss/batch_counter).item(), 2)
        avg_loss_per_hand = round((total_epoch_loss/len(train_loader)).item(), 2)
        print(f"Train: total loss in this epoch: {round(total_epoch_loss.item(), 2)}")
        print("Train: epoch avg loss per batch: ", avg_loss_per_batch)
        print("Train: epoch avg loss per hand: ", avg_loss_per_hand)
        return avg_loss_per_batch

    def model_eval(self, val_loader: DataLoader):
        self.eval()
        total_epoch_loss = 0
        with torch.no_grad():
            val_batch_counter = 0
            for batch_features, batch_actions in val_loader:
                logits = self.forward(batch_features)
                base_loss = self.criterion(logits, batch_actions)
                custom_penalty = self.calculate_custom_penalty(logits, batch_actions)
                total_loss = base_loss + custom_penalty

                total_epoch_loss += total_loss
                val_batch_counter += 1

        avg_loss_per_batch = round((total_epoch_loss/val_batch_counter).item(), 2)
        avg_loss_per_hand = round((total_epoch_loss/len(val_loader)).item(), 2)
        print(f"Evaluation: total loss in this epoch: {round(total_epoch_loss.item(), 2)}")
        print("Evaluation: epoch avg loss per batch: ", avg_loss_per_batch)
        print("Evaluation: epoch avg loss per hand: ", avg_loss_per_hand)
        return avg_loss_per_batch
        
    def calculate_custom_penalty(self, logits, targets):
        # Custom penalty for call vs check
        predictions = torch.argmax(logits, dim=1)
        custom_penalty = 0
        
        for i in range(len(targets)):
            if (targets[i] == CHECK and (predictions[i] == CALL or predictions[i] == FOLD)) or (targets[i] == CALL or targets[i] == FOLD) and predictions[i] == CHECK:
                custom_penalty += 5.0
        
        return custom_penalty

    def create_action_mask(self, actions_stength: int) -> torch.Tensor:
        action_mask = [1] * len(agent_utils.action_map.keys())
        if actions_stength <= 0:
            action_mask[CALL] = 0
            action_mask[FOLD] = 0
        else:
            action_mask[CHECK] = 0
        return action_mask
    
    def generate_evaluation_report(self, true_labels, predictions):
        """
        Generate comprehensive evaluation report with confusion matrix and metrics.
        Same format as supervised_poker_agent.py
        """
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, precision_recall_fscore_support
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(true_labels)
        
        # Action class names
        action_names = ['Fold', 'Check', 'Call', 'Raise']
        
        # Calculate overall accuracy
        accuracy = np.mean(labels == predictions)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
        
        # Calculate per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Print detailed results - same format as supervised_poker_agent
        print("\n" + "="*60)
        print("DETAILED EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
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
        
        print("PER-CLASS METRICS:")
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10} {'Support':<10}")
        print("-" * 65)
        for i, class_name in enumerate(action_names):
            print(f"{class_name:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {per_class_accuracy[i]:<10.4f} {support[i]:<10}")
        
        # Generate detailed classification report
        print(f"\nDETAILED CLASSIFICATION REPORT:")
        report = classification_report(labels, predictions, target_names=action_names)
        print(report)
        
        return cm, accuracy
    
    def save_model(self, filepath: str, train_losses: list = None, val_losses: list = None):
        """Save trained model with training history. Same format as supervised_poker_agent."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'train_losses': train_losses or [],
            'val_losses': val_losses or []
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from checkpoint file. Same format as supervised_poker_agent."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Model loaded from {filepath}")
        return train_losses, val_losses

def start_model(street: int = None):
    print("street: ", street)
    
    # If no street specified, train all streets
    if street is None:
        pdb.set_trace()
        streets_to_train = [FLOP, TURN, RIVER] 
    else:
        streets_to_train = [street]
    
    for current_street in streets_to_train:
        # this function loads the data, handle the agent's features and create action and features tensors
        features_tensor, actions_tensor = processor.load_data_and_create_tensors(current_street)
        #print(f"features: {features_tensor}")
        #print(f"actions: {actions_tensor}")

        #this function splits the data for two sets - train and validation
        train_loader, val_loader = processor.create_train_and_val_loaders(features_tensor, actions_tensor)

        # .shape is a function in python that returns the sizes of two+-dimensional arrays
        # the tensors are basicaly 2D array with number of rows = number of hands and number of columns = number of features
        # so the numbers of colmuns (shape[1]) is the number of features
        # we will call the number of features 'num_of_input_dimensions'
        num_of_input_dimensions = features_tensor.shape[1] # TODO - save this in your code so you can use it later
        print(f"input_dimensions (number of parsed features): {num_of_input_dimensions}")
        possible_actions = len(agent_utils.action_map.keys())


        # create a model
        model = PokerAgent(n_features=num_of_input_dimensions, n_actions=possible_actions)
        model.train_model(train_loader, val_loader, num_epochs=100, patience=15)
        
        # Save the trained model - same format as supervised_poker_agent
        model_save_path = f'agents/sep_poker_agent_{street_names[current_street]}.pth'
        model.save_model(model_save_path)
