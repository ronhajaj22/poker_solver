"""
Range Building and Updating System for Poker Analysis

This module provides comprehensive range building, updating, and analysis
for opponent hand ranges based on betting actions and board texture.
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import itertools
from AppUtils.constants import RANKS, SUITS
from equity_calculator import EquityCalculator


class RangeBuilder:
    """
    Builds and updates opponent hand ranges based on actions and board texture.
    """
    
    def __init__(self):
        self.equity_calculator = EquityCalculator()
        self.preflop_ranges = self._initialize_preflop_ranges()
    
    def _initialize_preflop_ranges(self) -> Dict[str, Dict[str, float]]:
        """Initialize standard preflop ranges for different positions and actions."""
        return {
            'UTG': self._create_utg_range(),
            'MP': self._create_mp_range(),
            'CO': self._create_co_range(),
            'BTN': self._create_btn_range(),
            'SB': self._create_sb_range(),
            'BB': self._create_bb_range()
        }
    
    def build_initial_range(self, position: str, action: str, bet_size: Optional[float] = None) -> Dict[str, float]:
        """
        Build initial range based on position and preflop action.
        
        Args:
            position: Player position (UTG, MP, CO, BTN, SB, BB)
            action: Preflop action (FOLD, CALL, RAISE, 3BET, 4BET)
            bet_size: Bet size (for sizing-based range adjustments)
            
        Returns:
            Range dictionary (combo -> weight)
        """
        base_range = self.preflop_ranges.get(position, self._create_btn_range())
        
        # Adjust based on action
        if action == 'FOLD':
            return {}
        elif action == 'CALL':
            return self._adjust_for_call(base_range)
        elif action == 'RAISE':
            return self._adjust_for_raise(base_range, bet_size)
        elif action == '3BET':
            return self._adjust_for_3bet(base_range, bet_size)
        elif action == '4BET':
            return self._adjust_for_4bet(base_range, bet_size)
        
        return base_range
    
    def update_range_for_action(self, 
                               current_range: Dict[str, float], 
                               action: str, 
                               board: List[dict],
                               bet_size: Optional[float] = None,
                               position: str = 'BTN') -> Dict[str, float]:
        """
        Update range based on postflop action.
        
        Args:
            current_range: Current range to update
            action: Action taken (FOLD, CALL, RAISE, CHECK)
            board: Current board cards
            bet_size: Bet size (for sizing-based adjustments)
            position: Player position
            
        Returns:
            Updated range
        """
        if not current_range:
            return {}
        
        # Calculate hand strengths for current range
        hand_strengths = {}
        for combo, weight in current_range.items():
            hand = self._combo_to_cards(combo)
            if hand and board:  # Only calculate if we have a board
                strength = self.equity_calculator.calculate_hand_strength_percentile(hand, board)
                hand_strengths[combo] = strength
            else:
                # Default strength for preflop or when no board
                hand_strengths[combo] = 0.5
        
        # Update range based on action
        if action == 'FOLD':
            return self._remove_weak_hands(current_range, hand_strengths, 0.3)
        elif action == 'CALL':
            return self._keep_medium_hands(current_range, hand_strengths, 0.2, 0.8)
        elif action == 'RAISE':
            return self._keep_strong_hands(current_range, hand_strengths, 0.6)
        elif action == 'CHECK':
            return self._keep_passive_hands(current_range, hand_strengths, 0.4)
        
        return current_range
    
    def update_range_for_board(self, 
                              current_range: Dict[str, float], 
                              board: List[dict]) -> Dict[str, float]:
        """
        Update range based on board texture (remove impossible hands).
        
        Args:
            current_range: Current range
            board: Board cards
            
        Returns:
            Updated range with impossible hands removed
        """
        if not current_range:
            return {}
        
        # Get used cards on board
        used_ranks = [card['rank'] for card in board]
        used_suits = [card['suit'] for card in board]
        
        updated_range = {}
        for combo, weight in current_range.items():
            if self._is_hand_possible(combo, used_ranks, used_suits):
                updated_range[combo] = weight
        
        # Normalize weights
        total_weight = sum(updated_range.values())
        if total_weight > 0:
            return {combo: weight/total_weight for combo, weight in updated_range.items()}
        
        return updated_range
    
    def get_calling_range(self, 
                         full_range: Dict[str, float], 
                         board: List[dict],
                         bet_size_ratio: float) -> Dict[str, float]:
        """
        Extract the portion of range that would call a bet.
        
        Args:
            full_range: Full range
            board: Board cards
            bet_size_ratio: Bet size as ratio of pot
            
        Returns:
            Calling range
        """
        calling_range = {}
        
        for combo, weight in full_range.items():
            hand = self._combo_to_cards(combo)
            if hand:
                strength = self.equity_calculator.calculate_hand_strength_percentile(hand, board)
                call_probability = self._get_call_probability(strength, bet_size_ratio)
                calling_range[combo] = weight * call_probability
        
        # Normalize weights
        total_weight = sum(calling_range.values())
        if total_weight > 0:
            return {combo: weight/total_weight for combo, weight in calling_range.items()}
        
        return calling_range
    
    def get_folding_range(self, 
                         full_range: Dict[str, float], 
                         board: List[dict],
                         bet_size_ratio: float) -> Dict[str, float]:
        """
        Extract the portion of range that would fold to a bet.
        
        Args:
            full_range: Full range
            board: Board cards
            bet_size_ratio: Bet size as ratio of pot
            
        Returns:
            Folding range
        """
        folding_range = {}
        
        for combo, weight in full_range.items():
            hand = self._combo_to_cards(combo)
            if hand:
                strength = self.equity_calculator.calculate_hand_strength_percentile(hand, board)
                call_probability = self._get_call_probability(strength, bet_size_ratio)
                fold_probability = 1.0 - call_probability
                folding_range[combo] = weight * fold_probability
        
        # Normalize weights
        total_weight = sum(folding_range.values())
        if total_weight > 0:
            return {combo: weight/total_weight for combo, weight in folding_range.items()}
        
        return folding_range
    
    def _create_utg_range(self) -> Dict[str, float]:
        """Create UTG (under the gun) preflop range."""
        range_dict = {
            'AA': 1.0, 'KK': 1.0, 'QQ': 1.0, 'JJ': 1.0, 'TT': 1.0, '99': 1.0,
            'AKs': 1.0, 'AQs': 1.0, 'AJs': 1.0, 'ATs': 1.0, 'A9s': 1.0,
            'AKo': 1.0, 'AQo': 1.0, 'AJo': 1.0,
            'KQs': 1.0, 'KJs': 1.0, 'KTs': 1.0,
            'KQo': 1.0, 'KJo': 1.0,
            'QJs': 1.0, 'QTs': 1.0,
            'QJo': 1.0,
            'JTs': 1.0, 'JTo': 1.0,
            'T9s': 1.0, 'T9o': 1.0,
            '98s': 1.0, '98o': 1.0,
            '87s': 1.0, '87o': 1.0,
            '76s': 1.0, '76o': 1.0,
            '65s': 1.0, '65o': 1.0,
            '54s': 1.0, '54o': 1.0
        }
        return self._normalize_range(range_dict)
    
    def _create_mp_range(self) -> Dict[str, float]:
        """Create MP (middle position) preflop range."""
        range_dict = self._create_utg_range()
        # Add more hands for MP
        additional_hands = {
            '88': 1.0, '77': 1.0, '66': 1.0, '55': 1.0, '44': 1.0, '33': 1.0, '22': 1.0,
            'A8s': 1.0, 'A7s': 1.0, 'A6s': 1.0, 'A5s': 1.0, 'A4s': 1.0, 'A3s': 1.0, 'A2s': 1.0,
            'ATo': 1.0, 'A9o': 1.0, 'A8o': 1.0, 'A7o': 1.0, 'A6o': 1.0, 'A5o': 1.0, 'A4o': 1.0, 'A3o': 1.0, 'A2o': 1.0,
            'K9s': 1.0, 'K8s': 1.0, 'K7s': 1.0, 'K6s': 1.0, 'K5s': 1.0, 'K4s': 1.0, 'K3s': 1.0, 'K2s': 1.0,
            'KTo': 1.0, 'K9o': 1.0, 'K8o': 1.0, 'K7o': 1.0, 'K6o': 1.0, 'K5o': 1.0, 'K4o': 1.0, 'K3o': 1.0, 'K2o': 1.0,
            'Q9s': 1.0, 'Q8s': 1.0, 'Q7s': 1.0, 'Q6s': 1.0, 'Q5s': 1.0, 'Q4s': 1.0, 'Q3s': 1.0, 'Q2s': 1.0,
            'QTo': 1.0, 'Q9o': 1.0, 'Q8o': 1.0, 'Q7o': 1.0, 'Q6o': 1.0, 'Q5o': 1.0, 'Q4o': 1.0, 'Q3o': 1.0, 'Q2o': 1.0,
            'J9s': 1.0, 'J8s': 1.0, 'J7s': 1.0, 'J6s': 1.0, 'J5s': 1.0, 'J4s': 1.0, 'J3s': 1.0, 'J2s': 1.0,
            'JTo': 1.0, 'J9o': 1.0, 'J8o': 1.0, 'J7o': 1.0, 'J6o': 1.0, 'J5o': 1.0, 'J4o': 1.0, 'J3o': 1.0, 'J2o': 1.0,
            'T8s': 1.0, 'T7s': 1.0, 'T6s': 1.0, 'T5s': 1.0, 'T4s': 1.0, 'T3s': 1.0, 'T2s': 1.0,
            'T9o': 1.0, 'T8o': 1.0, 'T7o': 1.0, 'T6o': 1.0, 'T5o': 1.0, 'T4o': 1.0, 'T3o': 1.0, 'T2o': 1.0,
            '97s': 1.0, '96s': 1.0, '95s': 1.0, '94s': 1.0, '93s': 1.0, '92s': 1.0,
            '98o': 1.0, '97o': 1.0, '96o': 1.0, '95o': 1.0, '94o': 1.0, '93o': 1.0, '92o': 1.0,
            '86s': 1.0, '85s': 1.0, '84s': 1.0, '83s': 1.0, '82s': 1.0,
            '87o': 1.0, '86o': 1.0, '85o': 1.0, '84o': 1.0, '83o': 1.0, '82o': 1.0,
            '75s': 1.0, '74s': 1.0, '73s': 1.0, '72s': 1.0,
            '76o': 1.0, '75o': 1.0, '74o': 1.0, '73o': 1.0, '72o': 1.0,
            '64s': 1.0, '63s': 1.0, '62s': 1.0,
            '65o': 1.0, '64o': 1.0, '63o': 1.0, '62o': 1.0,
            '53s': 1.0, '52s': 1.0,
            '54o': 1.0, '53o': 1.0, '52o': 1.0,
            '43s': 1.0, '42s': 1.0,
            '43o': 1.0, '42o': 1.0,
            '32s': 1.0,
            '32o': 1.0
        }
        range_dict.update(additional_hands)
        return self._normalize_range(range_dict)
    
    def _create_co_range(self) -> Dict[str, float]:
        """Create CO (cutoff) preflop range."""
        return self._create_mp_range()  # Similar to MP for simplicity
    
    def _create_btn_range(self) -> Dict[str, float]:
        """Create BTN (button) preflop range."""
        return self._create_mp_range()  # Similar to MP for simplicity
    
    def _create_sb_range(self) -> Dict[str, float]:
        """Create SB (small blind) preflop range."""
        return self._create_mp_range()  # Similar to MP for simplicity
    
    def _create_bb_range(self) -> Dict[str, float]:
        """Create BB (big blind) preflop range."""
        return self._create_mp_range()  # Similar to MP for simplicity
    
    def _normalize_range(self, range_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize range weights to sum to 1.0."""
        total = sum(range_dict.values())
        return {combo: weight/total for combo, weight in range_dict.items()}
    
    def _adjust_for_call(self, base_range: Dict[str, float]) -> Dict[str, float]:
        """Adjust range for calling action (remove premium hands)."""
        adjusted = {}
        for combo, weight in base_range.items():
            # Reduce weight for premium hands that would typically raise
            if combo in ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo']:
                adjusted[combo] = weight * 0.3
            else:
                adjusted[combo] = weight
        return self._normalize_range(adjusted)
    
    def _adjust_for_raise(self, base_range: Dict[str, float], bet_size: Optional[float]) -> Dict[str, float]:
        """Adjust range for raising action."""
        adjusted = {}
        for combo, weight in base_range.items():
            # Increase weight for premium hands
            if combo in ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo', 'AQs', 'AJo']:
                adjusted[combo] = weight * 1.5
            else:
                adjusted[combo] = weight * 0.8
        return self._normalize_range(adjusted)
    
    def _adjust_for_3bet(self, base_range: Dict[str, float], bet_size: Optional[float]) -> Dict[str, float]:
        """Adjust range for 3-betting action."""
        adjusted = {}
        for combo, weight in base_range.items():
            # Only keep premium hands for 3-bet
            if combo in ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo']:
                adjusted[combo] = weight * 2.0
            else:
                adjusted[combo] = weight * 0.1
        return self._normalize_range(adjusted)
    
    def _adjust_for_4bet(self, base_range: Dict[str, float], bet_size: Optional[float]) -> Dict[str, float]:
        """Adjust range for 4-betting action."""
        adjusted = {}
        for combo, weight in base_range.items():
            # Only keep the very best hands for 4-bet
            if combo in ['AA', 'KK', 'AKs']:
                adjusted[combo] = weight * 3.0
            else:
                adjusted[combo] = weight * 0.05
        return self._normalize_range(adjusted)
    
    def _remove_weak_hands(self, range_dict: Dict[str, float], strengths: Dict[str, float], threshold: float) -> Dict[str, float]:
        """Remove hands below strength threshold."""
        filtered = {}
        for combo, weight in range_dict.items():
            if strengths.get(combo, 0.5) >= threshold:
                filtered[combo] = weight
        return self._normalize_range(filtered)
    
    def _keep_medium_hands(self, range_dict: Dict[str, float], strengths: Dict[str, float], min_threshold: float, max_threshold: float) -> Dict[str, float]:
        """Keep hands within strength range."""
        filtered = {}
        for combo, weight in range_dict.items():
            strength = strengths.get(combo, 0.5)
            if min_threshold <= strength <= max_threshold:
                filtered[combo] = weight
        return self._normalize_range(filtered)
    
    def _keep_strong_hands(self, range_dict: Dict[str, float], strengths: Dict[str, float], threshold: float) -> Dict[str, float]:
        """Keep hands above strength threshold."""
        filtered = {}
        for combo, weight in range_dict.items():
            if strengths.get(combo, 0.5) >= threshold:
                filtered[combo] = weight
        return self._normalize_range(filtered)
    
    def _keep_passive_hands(self, range_dict: Dict[str, float], strengths: Dict[str, float], threshold: float) -> Dict[str, float]:
        """Keep hands below strength threshold (for checking)."""
        filtered = {}
        for combo, weight in range_dict.items():
            if strengths.get(combo, 0.5) <= threshold:
                filtered[combo] = weight
        return self._normalize_range(filtered)
    
    def _get_call_probability(self, hand_strength: float, bet_size_ratio: float) -> float:
        """Get call probability based on hand strength and bet size."""
        # Simple heuristic - stronger hands call more often
        base_prob = hand_strength
        
        # Adjust for bet size
        if bet_size_ratio <= 0.33:  # Small bet
            return min(1.0, base_prob * 1.2)
        elif bet_size_ratio <= 0.67:  # Medium bet
            return min(1.0, base_prob * 1.0)
        elif bet_size_ratio <= 1.0:  # Large bet
            return min(1.0, base_prob * 0.8)
        else:  # Overbet
            return min(1.0, base_prob * 0.6)
    
    def _is_hand_possible(self, combo: str, used_ranks: List[str], used_suits: List[str]) -> bool:
        """Check if a hand is possible given the board."""
        if len(combo) == 2:  # Pocket pair
            rank = combo[0]
            return rank not in used_ranks or used_ranks.count(rank) < 2
        elif len(combo) == 3:  # Suited/offsuit
            rank1, rank2 = combo[0], combo[1]
            # Check if both ranks are available
            rank1_available = rank1 not in used_ranks or used_ranks.count(rank1) < 2
            rank2_available = rank2 not in used_ranks or used_ranks.count(rank2) < 2
            return rank1_available and rank2_available
        
        return True
    
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


# Example usage
if __name__ == "__main__":
    range_builder = RangeBuilder()
    
    # Build initial range for UTG raise
    initial_range = range_builder.build_initial_range('UTG', 'RAISE', 3.0)
    print(f"Initial UTG raise range: {len(initial_range)} combos")
    
    # Update range for flop action
    board = [
        {'rank': 'A', 'suit': 'c'}, 
        {'rank': 'K', 'suit': 'd'}, 
        {'rank': 'Q', 'suit': 's'}
    ]
    
    updated_range = range_builder.update_range_for_action(
        initial_range, 'CALL', board, 0.5, 'UTG'
    )
    print(f"Updated range after flop call: {len(updated_range)} combos")
    
    # Get calling range
    calling_range = range_builder.get_calling_range(updated_range, board, 0.5)
    print(f"Calling range: {len(calling_range)} combos")
    
    # Get folding range
    folding_range = range_builder.get_folding_range(updated_range, board, 0.5)
    print(f"Folding range: {len(folding_range)} combos")
