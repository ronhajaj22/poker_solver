"""
ML Agent for Poker Action Prediction

This module provides a wrapper class that manages trained ML models
and provides predictions for poker actions based on current game state.
"""
'''
import os
import pickle
import numpy as np
from .feature_extractor import FeatureExtractor
from AppUtils.constants import FLOP, TURN, RIVER
from AppUtils.utils import FEATURES_BY_STREET
 
class MLAgent:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.bet_size_models = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models from disk"""
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_files = {
            1: os.path.join(script_dir, 'agents', 'poker_model1.pk'),
            2: os.path.join(script_dir, 'agents', 'poker_model2.pk'), 
            3: os.path.join(script_dir, 'agents', 'poker_model3.pk')
        }
        print("loading models..")
        
        for street, path in model_files.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        print("loading model from", path)
                        self.models[street] = pickle.load(f)
                    print(f"Loaded model for stage {street} from {path}")
                else:
                    print(f"Model file not found: {path}")
            except Exception as e:
                print(f"Error loading model for street  {street}: {e}")
        
        # Load bet sizing model
        bet_size_path = os.path.join(script_dir, 'agents', 'poker_bet_size_model.pkl')
        try:
            if os.path.exists(bet_size_path):
                with open(bet_size_path, 'rb') as f:
                    self.bet_size_models['bet_size'] = pickle.load(f)
                print(f"Loaded bet sizing model from {bet_size_path}")
            else:
                print(f"Bet sizing model file not found: {bet_size_path}")
        except Exception as e:
            print(f"Error loading bet sizing model: {e}")
    
    # This is the main prefict function. we get the params from player.py
    def predict_action(self, street, 
                      # Global features (16)
                      stage, hero_cards, board_cards, 
                      hero_pos, pot_size, stack_size, spr, hand_strength, 
                      num_of_players_pre_flop, num_of_players_flop, preflop_action, 
                      is_hero_pre_flop_aggressor, flop_action,
                      # Flop features (3) - default 0
                      flush_draw=0, straight_draw=0, board_dynamic=0,
                      # Turn features (7) - default 0
                      draw=0, is_flush_hit=0, is_hero_last_aggressor=0, 
                      num_of_players_turn=2, turn_action=0,
                      # River features (2) - default 0
                      num_of_players_river=2, river_action=0):
        if street not in self.models:
            print(f"No ML model available for street {street}")
            return None, None
        
        try:
            # here we calculate the features values for the model
            features = self.feature_extractor.extract_features_from_params(
                street, stage, hero_cards, board_cards, hero_pos, pot_size, stack_size, spr, hand_strength, 
                num_of_players_pre_flop, num_of_players_flop, preflop_action, is_hero_pre_flop_aggressor, flop_action,
                flush_draw, straight_draw, board_dynamic,
                draw, is_flush_hit, is_hero_last_aggressor, num_of_players_turn, turn_action,
                num_of_players_river, river_action
            )
            
            if features is None:
                print("Failed to extract features")
                return None, None
            
            # Get model features in correct order
            model_features = self.get_model_features(street, features)
            
            if model_features is None:
                print("Failed to extract model features")
                return None, None
            
            # Predict action
            model = self.models[street]
            print(f"Making prediction with {len(model_features)} features")
            prediction = model.predict([model_features])[0]
            print(f"Raw prediction: {prediction}")
            
            # Convert to action
            action = self.convert_prediction_to_action(prediction)
            print(f"Converted action: {action}")
            
            # Predict bet size if raising
            bet_size = None
            if action == 'RAISE' and 'bet_size' in self.bet_size_models:
                bet_size = self.predict_bet_size(model_features)
                print(f"Predicted bet size: {bet_size}")
            else:
                print(f"No bet size prediction (action: {action}, bet_size_models: {list(self.bet_size_models.keys())})")
            
            print(f"Returning action: {action}, bet_size: {bet_size}")
            return action, bet_size
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, None
        
        # Fallback return in case something goes wrong
        print("Unexpected exit from predict_action, returning fallback")
        return None, None
    
    def get_model_features(self, street, features):
        try:
            # Select feature list based on stage
            feature_list = FEATURES_BY_STREET[street]
            str = "";
            # Extract features in the correct order
            model_features = []
            for feature_name in feature_list:
                if feature_name in features:
                    model_features.append(features[feature_name])
                    str += f"{feature_name}: {features[feature_name]}, "
                else:
                    # Use default value for missing features
                    default_value = self.get_default_feature_value(feature_name)
                    model_features.append(default_value)
                    print(f"Warning: Missing feature {feature_name}, using default {default_value}")
            
            print(f"Extracted {len(model_features)} features for model")

            print(str)
            return model_features
            
        except Exception as e:
            print(f"Error preparing model features: {e}")
            return None
    
    def get_default_feature_value(self, feature_name):
        """Get default value for missing features"""
        defaults = {
            'stage': 0,
            'hero_card1': 0, 'hero_card2': 0,
            'hero_pos': 0, 'stack_size': 1.0,
            'board_card1': 0, 'board_card2': 0, 'board_card3': 0,
            'board_card4': 0, 'board_card5': 0,
            'pot_size': 0.0, 'spr': 0.0,
            'hand_strength': 0, 'flush_draw': 0, 'straight_draw': 0,
            'draw': 0, 'board_dynamic': 0, 'is_flush_hit': 0,
            'num_of_players_pre_flop': 6, 'num_of_players_flop': 6,
            'num_of_players_turn': 6, 'num_of_players_river': 6,
            'is_hero_pre_flop_aggressor': 0, 'is_hero_last_aggressor': 0,
            'preflop_action': 0, 'flop_action': 0, 'turn_action': 0, 'river_action': 0
        }
        return defaults.get(feature_name, 0)
    
    def convert_prediction_to_action(self, prediction):
        """
        Convert numerical prediction to action string
        Args: prediction: Numerical prediction from model
        Returns: str: Action string
        """
        action_map = {
            0: 'FOLD', 
            1: 'CHECK', 
            2: 'CALL', 
            3: 'RAISE', 
        }
        return action_map.get(prediction, 'FOLD')
    
    def predict_bet_size(self, features):
        """
        Predict bet size using regression model
        Args: features: Feature vector for bet sizing
        Returns: float: Predicted bet size in BB
        """
        try:
            if 'bet_size' in self.bet_size_models:
                bet_size = self.bet_size_models['bet_size'].predict([features])[0]
                # Ensure bet size is reasonable (positive and not too large)
                return max(0.1, min(bet_size, 100.0))
            return None
        except Exception as e:
            print(f"Error predicting bet size: {e}")
            return None
    
    def get_prediction_confidence(self, street, 
                                # Global features (16)
                                stage, hero_cards, board_cards,  hero_pos, pot_size, stack_size, spr, 
                                hand_strength, num_of_players_pre_flop, num_of_players_flop, preflop_action, 
                                is_hero_pre_flop_aggressor, flop_action,
                                # Flop features (3) 
                                flush_draw=0, straight_draw=0, board_dynamic=0,
                                # Turn features (7)
                                draw=0, is_flush_hit=0, is_hero_last_aggressor=0, num_of_players_turn=2, turn_action=0,
                                # River features (2)
                                num_of_players_river=2, river_action=0):
        
        """
        Get confidence score for ML prediction
        Args: Same as predict_action
        Returns: float: Confidence score (0-1)
        """
        if street not in self.models:
            return 0.0
        
        try:
            features = self.feature_extractor.extract_features_from_params(
                street, stage, hero_cards, board_cards, 
                hero_pos, pot_size, stack_size, spr, hand_strength, 
                num_of_players_pre_flop, num_of_players_flop, preflop_action,
                is_hero_pre_flop_aggressor, flop_action,
                flush_draw, straight_draw, board_dynamic,
                draw, is_flush_hit, is_hero_last_aggressor, 
                num_of_players_turn, turn_action,
                num_of_players_river, river_action
            )
            model_features = self.get_model_features(street, features)
            
            if model_features is None:
                return 0.0
            
            model = self.models[street]
            probabilities = model.predict_proba([model_features])[0] # Get prediction probabilities
            return max(probabilities) # return the highest probability
            
        except Exception as e:
            print(f"Error getting prediction confidence: {e}")
            return 0.0
    
    def is_model_available(self, street):
        """Check if ML model is available for given stage"""
        return street in self.models
    
    def get_available_stages(self):
        """Get list of stages with available models"""
        return list(self.models.keys())
    
    def reload_models(self):
        """Reload all models from disk"""
        self.models.clear()
        self.bet_size_models.clear()
        self.load_models() 
        '''