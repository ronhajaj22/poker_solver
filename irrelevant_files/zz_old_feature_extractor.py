"""
Feature Extractor for Poker ML Models

This module provides a simple parser/formatter that converts individual game state parameters
into the numerical format expected by the ML models.
"""
'''
from .json_formatter import JsonFormatter
from AppUtils.constants import FLOP, TURN, RIVER, ALL_POSITIONS_POST_FLOP
from AppUtils import utils
import pdb

# Define the order of poker positions for consistent indexing

class FeatureExtractor:
    def __init__(self):
        self.formatter = JsonFormatter()
    
    def extract_features_from_params(self, street, 
            # Global features (16)
            stage, hero_cards, board_cards, 
            hero_pos, pot_size, stack_size, spr, hand_strength, 
            num_of_players_pre_flop, num_of_players_flop, preflop_action, 
            is_hero_pre_flop_aggressor, flop_action,
            flush_draw=0, straight_draw=0, board_dynamic=0, # (Flop features (3))
            draw=0, is_flush_hit=False, is_hero_last_aggressor=False, num_of_players_turn=2, turn_action=0, # (Turn features (7))
            num_of_players_river=2, river_action=0):# River features (2)        
        """
        Returns: dict: Formatted features in numerical format
        """
        #print(f"Extracting features: stage={stage}, hero_cards={hero_cards}, hero_pos={hero_pos}, stack_size={stack_size}, board_cards={board_cards}, pot_size={pot_size}, spr={spr}, hand_strength={hand_strength}, flush_draw={flush_draw}, straight_draw={straight_draw}, board_dynamic={board_dynamic}, num_of_players_pre_flop={num_of_players_pre_flop}, num_of_players_flop={num_of_players_flop}, is_hero_pre_flop_aggressor={is_hero_pre_flop_aggressor}, preflop_action={preflop_action}, flop_action={flop_action}")

        try:
            features = {feature: 0 for feature in utils.FEATURES_BY_STREET[street]}
            features['stage'] = round(stage / 10, 1)
            
            # Format hero cards from string
            if hero_cards and len(hero_cards) == 2:
                hero_ranks = self.formatter.format_cards_to_rank(hero_cards)
                if len(hero_ranks) >= 2:
                    features['hero_card1'] = hero_ranks[0]
                    features['hero_card2'] = hero_ranks[1]
            
            # Format position (normalize to 0-1)
            features['hero_pos'] = round(ALL_POSITIONS_POST_FLOP.index(hero_pos) / 10, 1)
               
            # Format stack size (normalize by dividing by 100)
            features['stack_size'] = round(stack_size / 100, 2)
            
            # Format board cards from string
            if board_cards and len(board_cards) >= 3:
                board_ranks = self.formatter.format_cards_to_rank(board_cards)
                if len(board_ranks) >= 3:
                    features['board_card1'] = board_ranks[0]
                    features['board_card2'] = board_ranks[1]
                    features['board_card3'] = board_ranks[2]
                if len(board_ranks) >= 4 and street >= TURN:
                    features['board_card4'] = board_ranks[3]
                if len(board_ranks) >= 5 and street >= RIVER:
                    features['board_card5'] = board_ranks[4]
            
            # Format pot size (normalize by dividing by 100)
            features['pot_size'] = round(pot_size / 100, 3)
            
            # Format SPR (normalize by dividing by 10)
            features['spr'] = round((stack_size/pot_size)/10, 2)
            
            # Use the calculated values directly
            features['hand_strength'] = round(hand_strength/100, 4)
            features['is_hero_pre_flop_aggressor'] = 1 if is_hero_pre_flop_aggressor else 0
            features['preflop_action'] = preflop_action
            features['flop_action'] = flop_action
            
            # Set player counts (normalize to 0-1)
            features['num_of_players_pre_flop'] = round(num_of_players_pre_flop / 10, 1)
            features['num_of_players_flop'] = round(num_of_players_flop / 10, 1)
            
            # Add stage-specific features
            if street == FLOP:  # FLOP
                features['flush_draw'] = flush_draw
                features['straight_draw'] = straight_draw
                features['board_dynamic'] = board_dynamic
            elif street == TURN:  # TURN
                features['num_of_players_turn'] = round(num_of_players_turn / 10, 1)
                features['draw'] = draw 
                features['is_flush_hit'] = 1 if is_flush_hit else 0
                features['is_hero_last_aggressor'] = 1 if is_hero_last_aggressor else 0
                features['turn_action'] = turn_action 
                features['board_dynamic'] = board_dynamic
            else:  # RIVER
                features['num_of_players_turn'] = round(num_of_players_turn / 10, 1)
                features['num_of_players_river'] = round(num_of_players_river / 10, 1) 
                features['is_flush_hit'] = 1 if is_flush_hit else 0
                features['is_hero_last_aggressor'] = 1 if is_hero_last_aggressor else 0
                features['turn_action'] = turn_action  
                features['river_action'] = river_action 

            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return self._get_minimal_features(stage)
    
    def _get_minimal_features(self, stage):
        """Return minimal feature set when extraction fails"""
        if stage == FLOP:  # FLOP
            return {
                'stage': stage,
                'hero_card1': 0, 'hero_card2': 0,
                'hero_pos': 0, 'stack_size': 1.0,
                'board_card1': 0, 'board_card2': 0, 'board_card3': 0,
                'pot_size': 0.1, 'spr': 0.1,
                'hand_strength': 0, 'flush_draw': 0, 'straight_draw': 0,
                'board_dynamic': 0, 'num_of_players_pre_flop': 0.6,
                'num_of_players_flop': 0.2, 'is_hero_pre_flop_aggressor': 0,
                'preflop_action': 0, 'flop_action': 0
            }
        elif stage == TURN:  # TURN
            return {
                'stage': stage,
                'hero_card1': 0, 'hero_card2': 0,
                'hero_pos': 0, 'stack_size': 1.0,
                'board_card1': 0, 'board_card2': 0, 'board_card3': 0, 'board_card4': 0,
                'pot_size': 0.1, 'spr': 0.1,
                'hand_strength': 0, 'draw': 0, 'board_dynamic': 0,
                'is_flush_hit': 0, 'is_hero_pre_flop_aggressor': 0,
                'is_hero_last_aggressor': 0, 'num_of_players_pre_flop': 0.2,
                'num_of_players_flop': 0.2, 'num_of_players_turn': 0.2,
                'preflop_action': 0, 'flop_action': 0, 'turn_action': 0
            }
        else:  # RIVER
            return {
                'stage': stage,
                'hero_card1': 0, 'hero_card2': 0,
                'hero_pos': 0, 'stack_size': 1.0,
                'board_card1': 0, 'board_card2': 0, 'board_card3': 0, 'board_card4': 0, 'board_card5': 0,
                'pot_size': 0.2, 'spr': 0.2,
                'hand_strength': 0, 'is_hero_pre_flop_aggressor': 0,
                'is_flush_hit': 0, 'is_hero_last_aggressor': 0,
                'num_of_players_pre_flop': 0.2, 'num_of_players_flop': 0.2,
                'num_of_players_turn': 0.2, 'num_of_players_river': 0.2,
                'preflop_action': 0, 'flop_action': 0, 'turn_action': 0, 'river_action': 0
            }
            '''