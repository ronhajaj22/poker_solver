"""
Poker AI Agent - Machine Learning Model for Flop Action Prediction

This module trains a Random Forest classifier to predict optimal poker actions
on the flop based on hand history data. It processes parsed poker hands and
learns patterns to recommend FOLD, CHECK, CALL, or RAISE actions.

Each agent:
- Loads data from its respective JSON file (flop_parsed_hands.json, turn_parsed_hands.json, river_parsed_hands.json)
- Converts card representations to numerical features
- Trains a machine learning model on historical actions
- Saves and loads trained models for reuse
- Provides action predictions for new situations

Author: Poker Solver Team
"""
'''
import json
import os
import pandas as pd
from .json_formatter import JsonFormatter
from .feature_extractor import FeatureExtractor
from AppUtils import utils
from AppUtils.constants import FLOP, TURN, RIVER, ALL_POSITIONS_POST_FLOP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import pdb


def load_json_file(file_path="parsed_hands.json"):
    """Load poker hand data from a JSON file."""
    try:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Successfully loaded {len(data)} hands from {file_path}")
        return data
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []


def turn_json_to_decimal_format(data, street):
    """
    Convert raw poker hand data into numerical features for machine learning.
    This function normalizes and scales various poker features to create
    a consistent numerical representation that the ML model can learn from.
    """
    try:        
        return_data = FeatureExtractor().extract_features_from_params(
                    street, 
                    data.get('stage', 0), 
                    data.get('hero_cards', ''),
                    data.get('board_cards', ''),
                    data.get('hero_pos', ''),
                    data.get('pot_size', 0), 
                    data.get('stack_size', 0),
                    data.get('spr', 0), 
                    data.get('hand_strength', 0),
                    data.get('num_of_players_pre_flop', 0),
                    data.get('num_of_players_flop', 0),
                    data.get('preflop_action', 0),
                    data.get('is_hero_pre_flop_aggressor', False),
                    data.get('flop_action', 0),
                    data.get('flush_draw', 0),
                    data.get('straight_draw', 0),
                    data.get('board_dynamic', 0),
                    data.get('draw', 0),
                    data.get('is_flush_hit', False),
                    data.get('is_hero_last_aggressor', False),
                    data.get('num_of_players_turn', 2),
                    data.get('turn_action', 0),
                    data.get('num_of_players_river', 2),
                    data.get('river_action', 0)
                    )
        
        #if data['action'] == 'RAISE':
        #       action_size = float(round(data['action_size']/data['pot_size'], 2))
        return_data['action'] = utils.reversed_action_map[data['action']]
        return_data['action_size'] = 0 if (data['action_size'] == None or data['action_size'] == 'null') else round(data['action_size']/100, 2)
        
        return return_data
        
    except KeyError as e:
        print(f"Missing key in data: {e}")
        return None
    except Exception as e:
        pdb.set_trace()
        print(f"Error processing data: {e}")
        return None


def learn_action(json_file_name, street):
    data = load_json_file(json_file_name)
    if not data:
        print("No data loaded")
        return
    # Process each hand individually to convert to ML-ready format
    processed_data = []
    for hand in data:
        processed_hand = turn_json_to_decimal_format(hand, street)
        if processed_hand is not None:
            processed_data.append(processed_hand)
    
    if not processed_data:
        print("No valid data after processing")
        return
    
    print(f"Processed {len(processed_data)} hands")
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(processed_data)
    
    # Create target variables for supervised learning
    y_action = df["action"]  # What we want to predict (1-4)
    y_action_size = df["action_size"]  # Bet sizing (could be used for regression)
    
    # Filter to only use features that exist in our dataset
    features = utils.FEATURES_BY_STREET[street]
    available_columns = [col for col in features if col in df.columns]
    X = df[available_columns]

    # Split data into training (80%) and testing (20%) sets for action classification
    X_train, X_test, y_train, y_test = train_test_split(X, y_action, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model for action classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model's performance on unseen test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print detailed performance metrics
    print(f"Model trained on {len(X_train)} samples")
    print(f"Features used: {available_columns}")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n" + "="*80)
    
    # Save the trained model to disk for future use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = 'poker_model' + str(street) + '.pk'
    model_path = os.path.join(script_dir, 'agents', model_filename)
    # Ensure the agents directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as '{model_path}'")

    # --- Train regression model for action_size (bet sizing) ---
    # Only use rows where action_size > 0 (i.e., RAISE or BET)
    betting_df = df[df['action_size'] > 0]
    if not betting_df.empty:
        X_bet = betting_df[available_columns]
        y_bet_size = betting_df['action_size']
        X_bet_train, X_bet_test, y_bet_train, y_bet_test = train_test_split(X_bet, y_bet_size, test_size=0.2, random_state=42)
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_bet_train, y_bet_train)
        bet_size_score = regressor.score(X_bet_test, y_bet_test)
        print(f"Bet size regression R^2 score: {bet_size_score:.3f}")
        # Save the regressor
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bet_size_model_path = os.path.join(script_dir, 'agents', 'poker_bet_size_model.pkl')
        # Ensure the agents directory exists
        os.makedirs(os.path.dirname(bet_size_model_path), exist_ok=True)
        with open(bet_size_model_path, 'wb') as f:
            pickle.dump(regressor, f)
        print(f"Bet size model saved as '{bet_size_model_path}'")
    else:
        regressor = None
        print("No betting actions found for bet size regression.")

    # Return both models and test data
    return model, X_test, y_test, regressor if betting_df is not None else None

def load_model(model_path='poker_model.pkl'):
    """
    Load a previously trained poker model from disk.
    Args: model_path (str): Path to the saved model file (.pkl)
    Returns: RandomForestClassifier: Loaded model, or None if file not found
        
    Example: model = load_model('my_poker_model.pkl')
    action = predict_action(model, features)
    """
    try:
        # If it's a relative path, make it absolute relative to the agents directory
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'agents', model_path)
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file {model_path} not found")
        return None

def predict_action(model, features):
    """
    Use a trained model to predict the optimal poker action.
    
    Args:
        model: Trained RandomForestClassifier model
        features (list): Feature vector in the same order as training data        
        Returns: str: Predicted action ('FOLD', 'CHECK', 'CALL', 'RAISE') or 'UNKNOWN'
        
    Example:
        features = [0.2, 0.6, 0, 0.3, 0.8, 0.15, 1.2, 14, 13, 10, 7, 4, 0.8, 1, 0, 0.3, 0.7, 0]
        action = predict_action(model, features)
        print(f"AI suggests: {action}")
    """
    if model is None:
        return None
    
    # Ensure features are in the same format as training data
    # Model expects a 2D array, so we wrap features in a list
    prediction = model.predict([features])[0]
    
    # Convert numerical prediction back to action string
    return utils.action_map.get(prediction, 'UNKNOWN')

# Main execution block - runs when script is executed directly
if __name__ == "__main__":
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct absolute paths to the parsed hands files
        base_dir = os.path.dirname(script_dir)  # src/
        parsed_hands_dir = os.path.join(base_dir, "gg_hands_parser", "create_hero", "parsed_hands")
        
        flop_result = learn_action(os.path.join(parsed_hands_dir, "flop_parsed_hands.json"), 1)
        if flop_result is not None:
            model, X_test, y_test, bet_size_model = flop_result
        turn_result = learn_action(os.path.join(parsed_hands_dir, "turn_parsed_hands.json"), 2)
        if turn_result is not None:
            model, X_test, y_test, bet_size_model = turn_result
        river_result = learn_action(os.path.join(parsed_hands_dir, "river_parsed_hands.json"), 3)
        if river_result is not None:
            model, X_test, y_test, bet_size_model = river_result
            print("Model training completed successfully!")
        else:
            print("No data available for training")
    except Exception as e:
        print(f"Error running learn_flop_action: {e}")
'''