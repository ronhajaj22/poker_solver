"""
Bluff Classification System for Poker Analysis

This module implements the mathematical framework for classifying river bets as:
- VALUE: Profitable when called (high equity vs calling range)
- BLUFF: Profitable mainly through fold equity
- THIN: Ambiguous middle ground

Based on the comprehensive mathematical analysis provided by the user.
"""

from typing import Dict, Callable, Tuple, List
from AppUtils.hand_board_calculations import calc_hand_strength
from treys import Evaluator, Card
from AppUtils.constants import card_to_str
import itertools


class BluffClassifier:
    """
    Classifies river bets using mathematical equity and fold equity analysis.
    
    Key Variables:
    - P: Pot before bet (in money units)
    - B: Bet size (in money units) 
    - R: Opponent's range (combo -> weight, weights sum to 1.0)
    - p_call(combo): Probability combo calls the bet [0..1]
    - eq(hero, combo, board): Hero's equity vs combo on showdown [0..1]
    """
    
    def __init__(self, delta_eq: float = 0.03, delta_fe: float = 0.03):
        """
        Initialize classifier with sensitivity parameters.
        
        Args:
            delta_eq: Equity threshold buffer for VALUE classification
            delta_fe: Fold equity threshold buffer for BLUFF classification
        """
        self.delta_eq = delta_eq
        self.delta_fe = delta_fe
        self.evaluator = Evaluator()
    
    def classify_river_bet(self, 
                          P: float, 
                          B: float, 
                          R: Dict[str, float], 
                          p_call_func: Callable[[str], float],
                          hero_hand: List[dict], 
                          board: List[dict]) -> str:
        """
        Classify a river bet as VALUE, BLUFF, or THIN.
        
        Args:
            P: Pot before bet
            B: Bet size
            R: Opponent's range (combo -> weight)
            p_call_func: Function that returns call probability for a combo
            hero_hand: Hero's cards in format [{'rank': 'A', 'suit': 's'}, ...]
            board: Board cards in same format
            
        Returns:
            "VALUE" | "BLUFF" | "THIN"
        """
        # 1) Compute call weights and totals
        call_weights = {}
        for combo, weight in R.items():
            call_weights[combo] = weight * p_call_func(combo)
        
        CallTotal = sum(call_weights.values())
        FE = max(0.0, 1.0 - CallTotal)  # Fold Equity
        
        # If nobody calls, it's a pure bluff
        if CallTotal == 0:
            return "BLUFF"
        
        # 2) Compute eq_call (weighted equity vs calling range)
        weighted_eq_sum = 0.0
        for combo, call_weight in call_weights.items():
            equity = self._calculate_equity(hero_hand, combo, board)
            weighted_eq_sum += call_weight * equity
        
        eq_call = weighted_eq_sum / CallTotal
        
        # 3) Break-even equity when called
        beq = B / (P + 2.0 * B)
        
        # 4) Check for VALUE classification
        if eq_call >= beq + self.delta_eq:
            return "VALUE"
        
        # 5) Compute FE_needed for BLUFF classification
        numerator = B - eq_call * (P + 2.0 * B)
        denominator = (P + B) - eq_call * (P + 2.0 * B)
        
        if denominator <= 1e-12:
            FE_needed = 0.0
        else:
            FE_needed = max(0.0, numerator / denominator)
        
        # 6) Check for BLUFF classification
        if FE >= FE_needed + self.delta_fe:
            return "BLUFF"
        
        return "THIN"
    
    def _calculate_equity(self, hero_hand: List[dict], combo: str, board: List[dict]) -> float:
        """
        Calculate hero's equity vs a specific combo on the given board.
        
        Args:
            hero_hand: Hero's cards
            combo: Opponent's combo (e.g., 'AKs', 'QQ', 'AJo')
            board: Board cards
            
        Returns:
            Equity as float between 0 and 1
        """
        # Convert hero hand to treys format
        hero_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in hero_hand]
        board_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in board]
        
        # Parse opponent combo and convert to treys format
        opp_cards = self._parse_combo_to_cards(combo)
        if not opp_cards:
            return 0.5  # Default to 50% if combo parsing fails
        
        # Check if any cards overlap (impossible scenario)
        all_used = set(hero_cards + board_cards + opp_cards)
        if len(all_used) != len(hero_cards) + len(board_cards) + len(opp_cards):
            return 0.5  # Default to 50% if cards overlap
        
        # Calculate showdown equity
        hero_score = self.evaluator.evaluate(board_cards, hero_cards)
        opp_score = self.evaluator.evaluate(board_cards, opp_cards)
        
        if hero_score < opp_score:
            return 1.0  # Hero wins
        elif hero_score > opp_score:
            return 0.0  # Hero loses
        else:
            return 0.5  # Tie
    
    def _parse_combo_to_cards(self, combo: str) -> List[Card]:
        """
        Parse a combo string to treys Card objects.
        
        Args:
            combo: Combo string like 'AKs', 'QQ', 'AJo'
            
        Returns:
            List of Card objects or empty list if parsing fails
        """
        try:
            if len(combo) == 2:  # Pocket pair like 'QQ'
                rank = combo[0]
                return [Card.new(rank + 's'), Card.new(rank + 'h')]
            
            elif len(combo) == 3:  # Suited/offsuit like 'AKs' or 'AJo'
                rank1, rank2, suited = combo[0], combo[1], combo[2]
                
                if suited == 's':  # Suited
                    return [Card.new(rank1 + 's'), Card.new(rank2 + 's')]
                elif suited == 'o':  # Offsuit
                    return [Card.new(rank1 + 's'), Card.new(rank2 + 'h')]
                else:
                    return []
            
            return []
        except:
            return []
    
    def calculate_ev(self, P: float, B: float, FE: float, eq_call: float) -> float:
        """
        Calculate the expected value of a bet.
        
        Args:
            P: Pot before bet
            B: Bet size
            FE: Fold equity (probability opponent folds)
            eq_call: Equity when called
            
        Returns:
            Expected value of the bet
        """
        # EV = FE * P + (1-FE) * (eq_call * (P + 2B) - B)
        ev_called = eq_call * (P + 2 * B) - B
        return FE * P + (1 - FE) * ev_called
    
    def get_break_even_equity(self, P: float, B: float) -> float:
        """
        Calculate the break-even equity when called.
        
        Args:
            P: Pot before bet
            B: Bet size
            
        Returns:
            Minimum equity needed to break even when called
        """
        return B / (P + 2 * B)
    
    def get_required_fold_equity(self, P: float, B: float, eq_call: float) -> float:
        """
        Calculate the minimum fold equity needed for the bet to be profitable.
        
        Args:
            P: Pot before bet
            B: Bet size
            eq_call: Equity when called
            
        Returns:
            Minimum fold equity needed for profitability
        """
        numerator = B - eq_call * (P + 2 * B)
        denominator = (P + B) - eq_call * (P + 2 * B)
        
        if denominator <= 1e-12:
            return 0.0
        
        return max(0.0, numerator / denominator)


# Example usage and helper functions
def create_simple_call_function(bet_size_ratio: float) -> Callable[[str], float]:
    """
    Create a simple call probability function based on bet sizing.
    
    Args:
        bet_size_ratio: Bet size as ratio of pot (0.5 = half pot)
        
    Returns:
        Function that returns call probability for a given combo
    """
    def p_call(combo: str) -> float:
        # Simple heuristic: stronger hands call more often
        # This is a placeholder - in practice, you'd use actual data
        
        # Pocket pairs
        if len(combo) == 2:
            rank = combo[0]
            if rank in ['A', 'K', 'Q', 'J']:
                return 0.9 if bet_size_ratio <= 0.75 else 0.7
            elif rank in ['T', '9', '8']:
                return 0.8 if bet_size_ratio <= 0.5 else 0.4
            else:
                return 0.6 if bet_size_ratio <= 0.33 else 0.2
        
        # Suited/offsuit hands
        elif len(combo) == 3:
            rank1, rank2 = combo[0], combo[1]
            high_ranks = ['A', 'K', 'Q', 'J', 'T']
            
            if rank1 in high_ranks and rank2 in high_ranks:
                return 0.85 if bet_size_ratio <= 0.75 else 0.6
            elif rank1 in high_ranks or rank2 in high_ranks:
                return 0.7 if bet_size_ratio <= 0.5 else 0.3
            else:
                return 0.4 if bet_size_ratio <= 0.33 else 0.1
        
        return 0.5  # Default
    
    return p_call


def create_range_from_actions(actions: List[str], position: str) -> Dict[str, float]:
    """
    Create a range based on betting actions (simplified example).
    
    Args:
        actions: List of actions taken (e.g., ['CALL', 'CALL', 'RAISE'])
        position: Player position
        
    Returns:
        Range dictionary (combo -> weight)
    """
    # This is a simplified example - in practice, you'd use more sophisticated range building
    base_range = {
        'AA': 0.01, 'KK': 0.01, 'QQ': 0.01, 'JJ': 0.01, 'TT': 0.01,
        'AKs': 0.01, 'AQs': 0.01, 'AJs': 0.01, 'ATs': 0.01,
        'AKo': 0.01, 'AQo': 0.01, 'AJo': 0.01, 'ATo': 0.01,
        'KQs': 0.01, 'KJs': 0.01, 'KTs': 0.01,
        'KQo': 0.01, 'KJo': 0.01, 'KTo': 0.01,
        'QJs': 0.01, 'QTs': 0.01, 'QJo': 0.01, 'QTo': 0.01,
        'JTs': 0.01, 'JTo': 0.01, 'T9s': 0.01, 'T9o': 0.01
    }
    
    # Adjust based on actions (simplified)
    if 'RAISE' in actions:
        # Tighten range for raising
        for combo in base_range:
            if combo in ['AA', 'KK', 'QQ', 'AKs', 'AKo']:
                base_range[combo] *= 2
            else:
                base_range[combo] *= 0.5
    
    # Normalize weights
    total_weight = sum(base_range.values())
    return {combo: weight/total_weight for combo, weight in base_range.items()}


# Example usage
if __name__ == "__main__":
    # Example scenario
    P = 100  # Pot before bet
    B = 50   # Half pot bet
    
    # Hero's hand
    hero_hand = [{'rank': 'A', 'suit': 's'}, {'rank': 'K', 'suit': 'h'}]
    
    # Board
    board = [
        {'rank': 'A', 'suit': 'c'}, 
        {'rank': 'K', 'suit': 'd'}, 
        {'rank': 'Q', 'suit': 's'},
        {'rank': 'J', 'suit': 'h'},
        {'rank': 'T', 'suit': 'c'}
    ]
    
    # Opponent's range (simplified)
    R = create_range_from_actions(['CALL', 'CALL'], 'BTN')
    
    # Call probability function
    p_call_func = create_simple_call_function(B/P)
    
    # Classify the bet
    classifier = BluffClassifier()
    result = classifier.classify_river_bet(P, B, R, p_call_func, hero_hand, board)
    
    print(f"Bet classification: {result}")
    print(f"Break-even equity when called: {classifier.get_break_even_equity(P, B):.1%}")
