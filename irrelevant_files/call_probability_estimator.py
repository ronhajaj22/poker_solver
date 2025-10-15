"""
Call Probability Estimation System for Poker Analysis

This module provides sophisticated call probability estimation based on:
- Bet sizing patterns
- Player history and tendencies
- Board texture
- Position and stack sizes
- Hand strength vs range
"""

from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import math
from equity_calculator import EquityCalculator
from range_builder import RangeBuilder


class CallProbabilityEstimator:
    """
    Estimates call probabilities for different hand combinations based on various factors.
    """
    
    def __init__(self):
        self.equity_calculator = EquityCalculator()
        self.range_builder = RangeBuilder()
        
        # Player tendency profiles
        self.player_profiles = {
            'tight': {'fold_frequency': 0.7, 'call_frequency': 0.2, 'raise_frequency': 0.1},
            'loose': {'fold_frequency': 0.4, 'call_frequency': 0.4, 'raise_frequency': 0.2},
            'aggressive': {'fold_frequency': 0.3, 'call_frequency': 0.3, 'raise_frequency': 0.4},
            'passive': {'fold_frequency': 0.5, 'call_frequency': 0.4, 'raise_frequency': 0.1},
            'balanced': {'fold_frequency': 0.5, 'call_frequency': 0.3, 'raise_frequency': 0.2}
        }
        
        # Bet sizing categories
        self.bet_sizing_categories = {
            'small': (0.0, 0.33),
            'medium': (0.33, 0.67),
            'large': (0.67, 1.0),
            'overbet': (1.0, 2.0),
            'all_in': (2.0, float('inf'))
        }
    
    def estimate_call_probability(self, 
                                 combo: str, 
                                 board: List[dict],
                                 bet_size_ratio: float,
                                 player_profile: str = 'balanced',
                                 position: str = 'BTN',
                                 stack_size: float = 100.0,
                                 pot_size: float = 10.0,
                                 action_history: List[str] = None) -> float:
        """
        Estimate call probability for a specific combo.
        
        Args:
            combo: Hand combination (e.g., 'AKs', 'QQ')
            board: Board cards
            bet_size_ratio: Bet size as ratio of pot
            player_profile: Player tendency profile
            position: Player position
            stack_size: Player's stack size
            pot_size: Current pot size
            action_history: List of previous actions in the hand
            
        Returns:
            Call probability between 0.0 and 1.0
        """
        if action_history is None:
            action_history = []
        
        # Base call probability from hand strength
        base_prob = self._get_base_call_probability(combo, board)
        
        # Adjust for bet sizing
        sizing_adjustment = self._get_sizing_adjustment(bet_size_ratio)
        
        # Adjust for player profile
        profile_adjustment = self._get_profile_adjustment(player_profile)
        
        # Adjust for position
        position_adjustment = self._get_position_adjustment(position)
        
        # Adjust for stack size
        stack_adjustment = self._get_stack_adjustment(stack_size, pot_size)
        
        # Adjust for action history
        history_adjustment = self._get_history_adjustment(action_history)
        
        # Combine all adjustments
        final_prob = base_prob * sizing_adjustment * profile_adjustment * position_adjustment * stack_adjustment * history_adjustment
        
        # Ensure probability is between 0 and 1
        return max(0.0, min(1.0, final_prob))
    
    def create_call_probability_function(self, 
                                       board: List[dict],
                                       bet_size_ratio: float,
                                       player_profile: str = 'balanced',
                                       position: str = 'BTN',
                                       stack_size: float = 100.0,
                                       pot_size: float = 10.0,
                                       action_history: List[str] = None) -> Callable[[str], float]:
        """
        Create a call probability function for a specific situation.
        
        Returns:
            Function that takes a combo string and returns call probability
        """
        def p_call(combo: str) -> float:
            return self.estimate_call_probability(
                combo, board, bet_size_ratio, player_profile, 
                position, stack_size, pot_size, action_history
            )
        
        return p_call
    
    def _get_base_call_probability(self, combo: str, board: List[dict]) -> float:
        """Get base call probability based on hand strength."""
        hand = self._combo_to_cards(combo)
        if not hand:
            return 0.5
        
        # Calculate hand strength percentile
        strength = self.equity_calculator.calculate_hand_strength_percentile(hand, board)
        
        # Convert strength to call probability
        # Stronger hands call more often
        if strength >= 0.8:
            return 0.9
        elif strength >= 0.6:
            return 0.7
        elif strength >= 0.4:
            return 0.5
        elif strength >= 0.2:
            return 0.3
        else:
            return 0.1
    
    def _get_sizing_adjustment(self, bet_size_ratio: float) -> float:
        """Get adjustment factor based on bet sizing."""
        if bet_size_ratio <= 0.33:  # Small bet
            return 1.2
        elif bet_size_ratio <= 0.67:  # Medium bet
            return 1.0
        elif bet_size_ratio <= 1.0:  # Large bet
            return 0.8
        elif bet_size_ratio <= 2.0:  # Overbet
            return 0.6
        else:  # All-in
            return 0.4
    
    def _get_profile_adjustment(self, player_profile: str) -> float:
        """Get adjustment factor based on player profile."""
        profile = self.player_profiles.get(player_profile, self.player_profiles['balanced'])
        return profile['call_frequency'] / 0.3  # Normalize to balanced profile
    
    def _get_position_adjustment(self, position: str) -> float:
        """Get adjustment factor based on position."""
        position_adjustments = {
            'UTG': 0.8,    # Early position - more conservative
            'MP': 0.9,     # Middle position
            'CO': 1.0,     # Cutoff
            'BTN': 1.1,    # Button - more aggressive
            'SB': 0.9,     # Small blind
            'BB': 1.0      # Big blind
        }
        return position_adjustments.get(position, 1.0)
    
    def _get_stack_adjustment(self, stack_size: float, pot_size: float) -> float:
        """Get adjustment factor based on stack size."""
        stack_to_pot_ratio = stack_size / pot_size
        
        if stack_to_pot_ratio >= 20:  # Deep stack
            return 1.1
        elif stack_to_pot_ratio >= 10:  # Medium stack
            return 1.0
        elif stack_to_pot_ratio >= 5:  # Short stack
            return 0.9
        else:  # Very short stack
            return 0.8
    
    def _get_history_adjustment(self, action_history: List[str]) -> float:
        """Get adjustment factor based on action history."""
        if not action_history:
            return 1.0
        
        # Count recent actions
        recent_actions = action_history[-3:] if len(action_history) >= 3 else action_history
        
        fold_count = recent_actions.count('FOLD')
        call_count = recent_actions.count('CALL')
        raise_count = recent_actions.count('RAISE')
        
        total_actions = len(recent_actions)
        if total_actions == 0:
            return 1.0
        
        # Adjust based on recent tendencies
        call_frequency = call_count / total_actions
        
        if call_frequency >= 0.6:  # Very loose calling
            return 1.2
        elif call_frequency >= 0.4:  # Loose calling
            return 1.1
        elif call_frequency >= 0.2:  # Balanced
            return 1.0
        elif call_frequency >= 0.1:  # Tight calling
            return 0.9
        else:  # Very tight calling
            return 0.8
    
    def estimate_range_call_probabilities(self, 
                                        range_dict: Dict[str, float],
                                        board: List[dict],
                                        bet_size_ratio: float,
                                        player_profile: str = 'balanced',
                                        position: str = 'BTN',
                                        stack_size: float = 100.0,
                                        pot_size: float = 10.0,
                                        action_history: List[str] = None) -> Dict[str, float]:
        """
        Estimate call probabilities for an entire range.
        
        Returns:
            Dictionary mapping combos to call probabilities
        """
        call_probs = {}
        
        for combo, weight in range_dict.items():
            call_prob = self.estimate_call_probability(
                combo, board, bet_size_ratio, player_profile,
                position, stack_size, pot_size, action_history
            )
            call_probs[combo] = call_prob
        
        return call_probs
    
    def get_fold_equity(self, 
                       range_dict: Dict[str, float],
                       call_probabilities: Dict[str, float]) -> float:
        """
        Calculate fold equity (probability opponent folds).
        
        Args:
            range_dict: Opponent's range
            call_probabilities: Call probabilities for each combo
            
        Returns:
            Fold equity between 0.0 and 1.0
        """
        total_fold_weight = 0.0
        total_weight = 0.0
        
        for combo, weight in range_dict.items():
            call_prob = call_probabilities.get(combo, 0.5)
            fold_prob = 1.0 - call_prob
            total_fold_weight += weight * fold_prob
            total_weight += weight
        
        return total_fold_weight / total_weight if total_weight > 0 else 0.5
    
    def get_calling_range_weights(self, 
                                 range_dict: Dict[str, float],
                                 call_probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Get weighted calling range.
        
        Args:
            range_dict: Full range
            call_probabilities: Call probabilities for each combo
            
        Returns:
            Calling range with adjusted weights
        """
        calling_range = {}
        
        for combo, weight in range_dict.items():
            call_prob = call_probabilities.get(combo, 0.5)
            calling_range[combo] = weight * call_prob
        
        # Normalize weights
        total_weight = sum(calling_range.values())
        if total_weight > 0:
            return {combo: weight/total_weight for combo, weight in calling_range.items()}
        
        return calling_range
    
    def analyze_bet_sizing_impact(self, 
                                 range_dict: Dict[str, float],
                                 board: List[dict],
                                 bet_sizes: List[float],
                                 player_profile: str = 'balanced') -> Dict[float, Dict[str, float]]:
        """
        Analyze how different bet sizes affect call probabilities.
        
        Args:
            range_dict: Opponent's range
            board: Board cards
            bet_sizes: List of bet sizes to analyze
            player_profile: Player profile
            
        Returns:
            Dictionary mapping bet sizes to call probability distributions
        """
        results = {}
        
        for bet_size in bet_sizes:
            call_probs = self.estimate_range_call_probabilities(
                range_dict, board, bet_size, player_profile
            )
            results[bet_size] = call_probs
        
        return results
    
    def _combo_to_cards(self, combo: str) -> Optional[List[dict]]:
        """Convert combo string to card objects."""
        try:
            if len(combo) == 2:  # Pocket pair
                rank = combo[0]
                return [{'rank': rank, 'suit': 's'}, {'rank': rank, 'suit': 'h'}]
            elif len(combo) == 3:  # Suited/offsuit
                rank1, rank2, suited = combo[0], combo[1], combo[2]
                if suited == 's':  # Suited
                    return [{'rank': rank1, 'suit': 's'}, {'rank': rank2, 'suit': 's'}]
                elif suited == 'o':  # Offsuit
                    return [{'rank': rank1, 'suit': 's'}, {'rank': rank2, 'suit': 'h'}]
            return None
        except:
            return None


class AdvancedCallProbabilityEstimator(CallProbabilityEstimator):
    """
    Advanced call probability estimator with machine learning capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.hand_strength_cache = {}
        self.board_texture_cache = {}
    
    def estimate_call_probability_with_ml(self, 
                                        combo: str, 
                                        board: List[dict],
                                        bet_size_ratio: float,
                                        player_stats: Dict[str, float],
                                        game_context: Dict[str, any]) -> float:
        """
        Estimate call probability using machine learning features.
        
        Args:
            combo: Hand combination
            board: Board cards
            bet_size_ratio: Bet size ratio
            player_stats: Player statistics (VPIP, PFR, etc.)
            game_context: Additional game context
            
        Returns:
            Call probability
        """
        # Extract features
        features = self._extract_features(combo, board, bet_size_ratio, player_stats, game_context)
        
        # Use ML model to predict (placeholder for actual ML model)
        prediction = self._predict_with_ml(features)
        
        return prediction
    
    def _extract_features(self, 
                         combo: str, 
                         board: List[dict],
                         bet_size_ratio: float,
                         player_stats: Dict[str, float],
                         game_context: Dict[str, any]) -> Dict[str, float]:
        """Extract features for ML model."""
        features = {}
        
        # Hand strength features
        hand = self._combo_to_cards(combo)
        if hand:
            strength = self.equity_calculator.calculate_hand_strength_percentile(hand, board)
            features['hand_strength'] = strength
        
        # Bet sizing features
        features['bet_size_ratio'] = bet_size_ratio
        features['bet_size_category'] = self._get_bet_size_category(bet_size_ratio)
        
        # Board texture features
        features['board_wetness'] = self._calculate_board_wetness(board)
        features['board_coordination'] = self._calculate_board_coordination(board)
        
        # Player stats features
        features['vpip'] = player_stats.get('vpip', 0.2)
        features['pfr'] = player_stats.get('pfr', 0.15)
        features['fold_to_cbet'] = player_stats.get('fold_to_cbet', 0.5)
        features['call_cbet'] = player_stats.get('call_cbet', 0.3)
        
        # Game context features
        features['position'] = self._encode_position(game_context.get('position', 'BTN'))
        features['stack_to_pot_ratio'] = game_context.get('stack_to_pot_ratio', 10.0)
        features['number_of_players'] = game_context.get('number_of_players', 6)
        
        return features
    
    def _predict_with_ml(self, features: Dict[str, float]) -> float:
        """Predict call probability using ML model (placeholder)."""
        # This would use an actual trained ML model
        # For now, use a simple weighted combination
        
        weights = {
            'hand_strength': 0.4,
            'bet_size_ratio': 0.2,
            'vpip': 0.15,
            'fold_to_cbet': 0.1,
            'board_wetness': 0.1,
            'position': 0.05
        }
        
        prediction = 0.0
        for feature, weight in weights.items():
            prediction += features.get(feature, 0.5) * weight
        
        return max(0.0, min(1.0, prediction))
    
    def _get_bet_size_category(self, bet_size_ratio: float) -> float:
        """Get bet size category as numeric value."""
        if bet_size_ratio <= 0.33:
            return 0.0  # Small
        elif bet_size_ratio <= 0.67:
            return 0.25  # Medium
        elif bet_size_ratio <= 1.0:
            return 0.5  # Large
        elif bet_size_ratio <= 2.0:
            return 0.75  # Overbet
        else:
            return 1.0  # All-in
    
    def _calculate_board_wetness(self, board: List[dict]) -> float:
        """Calculate board wetness (how many draws are possible)."""
        if len(board) < 3:
            return 0.0
        
        wetness = 0.0
        
        # Check for flush draws
        suit_counts = {}
        for card in board:
            suit = card['suit']
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        max_suit_count = max(suit_counts.values())
        if max_suit_count >= 2:
            wetness += 0.3
        
        # Check for straight draws
        ranks = [card['rank'] for card in board]
        rank_values = [self._rank_to_value(rank) for rank in ranks]
        rank_values.sort()
        
        # Check for consecutive ranks
        consecutive_count = 0
        for i in range(len(rank_values) - 1):
            if rank_values[i+1] - rank_values[i] == 1:
                consecutive_count += 1
        
        if consecutive_count >= 2:
            wetness += 0.4
        
        return min(1.0, wetness)
    
    def _calculate_board_coordination(self, board: List[dict]) -> float:
        """Calculate board coordination (how connected the board is)."""
        if len(board) < 3:
            return 0.0
        
        ranks = [card['rank'] for card in board]
        rank_values = [self._rank_to_value(rank) for rank in ranks]
        rank_values.sort()
        
        # Calculate average gap between ranks
        gaps = []
        for i in range(len(rank_values) - 1):
            gaps.append(rank_values[i+1] - rank_values[i])
        
        if not gaps:
            return 0.0
        
        avg_gap = sum(gaps) / len(gaps)
        
        # Lower average gap = more coordinated
        coordination = max(0.0, 1.0 - (avg_gap - 1) / 4.0)
        
        return coordination
    
    def _encode_position(self, position: str) -> float:
        """Encode position as numeric value."""
        position_encoding = {
            'UTG': 0.0,
            'MP': 0.2,
            'CO': 0.4,
            'BTN': 0.6,
            'SB': 0.8,
            'BB': 1.0
        }
        return position_encoding.get(position, 0.5)
    
    def _rank_to_value(self, rank: str) -> int:
        """Convert rank to numeric value."""
        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_values.get(rank, 7)


# Example usage
if __name__ == "__main__":
    estimator = CallProbabilityEstimator()
    
    # Example scenario
    board = [
        {'rank': 'A', 'suit': 'c'}, 
        {'rank': 'K', 'suit': 'd'}, 
        {'rank': 'Q', 'suit': 's'}
    ]
    
    # Test different combos
    combos = ['AA', 'AKs', 'QQ', 'AJo', 'KTs', '72o']
    
    for combo in combos:
        call_prob = estimator.estimate_call_probability(
            combo, board, 0.5, 'balanced', 'BTN', 100.0, 10.0
        )
        print(f"{combo}: {call_prob:.1%} call probability")
    
    # Test range analysis
    range_dict = {'AA': 0.1, 'AKs': 0.1, 'QQ': 0.1, 'AJo': 0.1, 'KTs': 0.1, '72o': 0.1}
    call_probs = estimator.estimate_range_call_probabilities(
        range_dict, board, 0.5, 'balanced', 'BTN', 100.0, 10.0
    )
    
    print(f"\nRange call probabilities:")
    for combo, prob in call_probs.items():
        print(f"{combo}: {prob:.1%}")
    
    # Calculate fold equity
    fold_equity = estimator.get_fold_equity(range_dict, call_probs)
    print(f"\nFold equity: {fold_equity:.1%}")
