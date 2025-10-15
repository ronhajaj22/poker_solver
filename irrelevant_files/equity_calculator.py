"""
Advanced Equity Calculator for Poker Analysis

This module provides comprehensive equity calculations for poker hands,
including Monte Carlo simulations and exact calculations using the treys library.
"""

from typing import List, Dict, Tuple, Optional
from treys import Evaluator, Card
from AppUtils.constants import card_to_str, RANKS, SUITS
import itertools
import random
from collections import Counter


class EquityCalculator:
    """
    Advanced equity calculator with multiple calculation methods.
    """
    
    def __init__(self):
        self.evaluator = Evaluator()
    
    def calculate_exact_equity(self, 
                             hero_hand: List[dict], 
                             villain_hand: List[dict], 
                             board: List[dict]) -> float:
        """
        Calculate exact equity between two hands on a given board.
        
        Args:
            hero_hand: Hero's cards [{'rank': 'A', 'suit': 's'}, ...]
            villain_hand: Villain's cards in same format
            board: Board cards in same format
            
        Returns:
            Hero's equity (0.0 to 1.0)
        """
        # Convert to treys format
        hero_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in hero_hand]
        villain_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in villain_hand]
        board_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in board]
        
        # Check for card conflicts
        all_cards = set(hero_cards + villain_cards + board_cards)
        if len(all_cards) != len(hero_cards) + len(villain_cards) + len(board_cards):
            return 0.5  # Default to tie if cards overlap
        
        # Calculate hand strengths
        hero_score = self.evaluator.evaluate(board_cards, hero_cards)
        villain_score = self.evaluator.evaluate(board_cards, villain_cards)
        
        if hero_score < villain_score:
            return 1.0  # Hero wins
        elif hero_score > villain_score:
            return 0.0  # Hero loses
        else:
            return 0.5  # Tie
    
    def calculate_equity_vs_range(self, 
                                 hero_hand: List[dict], 
                                 villain_range: Dict[str, float], 
                                 board: List[dict]) -> float:
        """
        Calculate hero's equity against a weighted range of hands.
        
        Args:
            hero_hand: Hero's cards
            villain_range: Dict mapping combo strings to weights
            board: Board cards
            
        Returns:
            Weighted average equity vs the range
        """
        total_equity = 0.0
        total_weight = 0.0
        
        for combo, weight in villain_range.items():
            villain_hand = self._combo_to_cards(combo)
            if villain_hand:
                equity = self.calculate_exact_equity(hero_hand, villain_hand, board)
                total_equity += equity * weight
                total_weight += weight
        
        return total_equity / total_weight if total_weight > 0 else 0.5
    
    def calculate_equity_vs_calling_range(self, 
                                        hero_hand: List[dict], 
                                        villain_range: Dict[str, float], 
                                        board: List[dict],
                                        call_probabilities: Dict[str, float]) -> float:
        """
        Calculate hero's equity against the portion of villain's range that calls.
        
        Args:
            hero_hand: Hero's cards
            villain_range: Villain's full range
            board: Board cards
            call_probabilities: Probability each combo calls
            
        Returns:
            Equity vs calling range
        """
        calling_equity = 0.0
        calling_weight = 0.0
        
        for combo, range_weight in villain_range.items():
            call_prob = call_probabilities.get(combo, 0.0)
            call_weight = range_weight * call_prob
            
            if call_weight > 0:
                villain_hand = self._combo_to_cards(combo)
                if villain_hand:
                    equity = self.calculate_exact_equity(hero_hand, villain_hand, board)
                    calling_equity += equity * call_weight
                    calling_weight += call_weight
        
        return calling_equity / calling_weight if calling_weight > 0 else 0.5
    
    def monte_carlo_equity(self, 
                          hero_hand: List[dict], 
                          villain_hand: List[dict], 
                          board: List[dict], 
                          iterations: int = 10000) -> float:
        """
        Calculate equity using Monte Carlo simulation for incomplete boards.
        
        Args:
            hero_hand: Hero's cards
            villain_hand: Villain's cards
            board: Current board (can be incomplete)
            iterations: Number of simulations
            
        Returns:
            Hero's equity
        """
        if len(board) >= 5:
            # Board is complete, use exact calculation
            return self.calculate_exact_equity(hero_hand, villain_hand, board)
        
        # Get used cards
        used_cards = set()
        for card in hero_hand + villain_hand + board:
            used_cards.add(Card.new(card_to_str(card)))
        
        # Create remaining deck
        full_deck = [Card.new(rank + suit) for rank in RANKS for suit in SUITS]
        remaining_cards = [card for card in full_deck if card not in used_cards]
        
        # Convert hands to treys format
        hero_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in hero_hand]
        villain_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in villain_hand]
        board_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in board]
        
        wins = 0
        ties = 0
        
        for _ in range(iterations):
            # Complete the board randomly
            needed_cards = 5 - len(board)
            additional_cards = random.sample(remaining_cards, needed_cards)
            complete_board = board_cards + additional_cards
            
            # Evaluate hands
            hero_score = self.evaluator.evaluate(complete_board, hero_cards)
            villain_score = self.evaluator.evaluate(complete_board, villain_cards)
            
            if hero_score < villain_score:
                wins += 1
            elif hero_score == villain_score:
                ties += 1
        
        return (wins + ties * 0.5) / iterations
    
    def calculate_hand_strength_percentile(self, 
                                         hand: List[dict], 
                                         board: List[dict]) -> float:
        """
        Calculate what percentile a hand is in against all possible opponent hands.
        
        Args:
            hand: Hand to evaluate
            board: Board cards
            
        Returns:
            Percentile (0.0 to 1.0, where 1.0 is nuts)
        """
        # Get used cards
        used_cards = set()
        for card in hand + board:
            if isinstance(card, dict):
                used_cards.add(Card.new(card['rank'] + card['suit']))
            else:
                used_cards.add(Card.new(card_to_str(card)))
        
        # Create remaining deck
        full_deck = [Card.new(rank + suit) for rank in RANKS for suit in SUITS]
        remaining_cards = [card for card in full_deck if card not in used_cards]
        
        # Convert hand and board to treys format
        hand_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in hand]
        board_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in board]
        
        # Evaluate hero's hand
        hero_score = self.evaluator.evaluate(board_cards, hand_cards)
        
        # Generate all possible opponent hands
        worse_hands = 0
        total_hands = 0
        
        for opp_hand in itertools.combinations(remaining_cards, 2):
            opp_score = self.evaluator.evaluate(board_cards, list(opp_hand))
            if hero_score < opp_score:
                worse_hands += 1
            total_hands += 1
        
        return worse_hands / total_hands if total_hands > 0 else 0.5
    
    def _combo_to_cards(self, combo: str) -> Optional[List[dict]]:
        """
        Convert a combo string to card objects.
        
        Args:
            combo: Combo string like 'AKs', 'QQ', 'AJo'
            
        Returns:
            List of card dicts or None if invalid
        """
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
    
    def get_nut_equity(self, board: List[dict]) -> Dict[str, float]:
        """
        Calculate equity for the nuts (best possible hand) on the board.
        
        Args:
            board: Board cards
            
        Returns:
            Dict with nut hand type and its equity vs random hands
        """
        # This is a simplified version - in practice, you'd want more sophisticated nut detection
        board_cards = [Card.new(card['rank'] + card['suit']) if isinstance(card, dict) else Card.new(card_to_str(card)) for card in board]
        
        # Get used cards
        used_cards = set()
        for card in board:
            if isinstance(card, dict):
                used_cards.add(Card.new(card['rank'] + card['suit']))
            else:
                used_cards.add(Card.new(card_to_str(card)))
        
        full_deck = [Card.new(rank + suit) for rank in RANKS for suit in SUITS]
        remaining_cards = [card for card in full_deck if card not in used_cards]
        
        best_score = float('inf')
        best_hand = None
        
        # Find the best possible hand
        for hand in itertools.combinations(remaining_cards, 2):
            score = self.evaluator.evaluate(board_cards, list(hand))
            if score < best_score:
                best_score = score
                best_hand = list(hand)
        
        if best_hand:
            # Calculate equity vs random hands
            wins = 0
            total = 0
            
            for opp_hand in itertools.combinations(remaining_cards, 2):
                if list(opp_hand) != best_hand:
                    opp_score = self.evaluator.evaluate(board_cards, list(opp_hand))
                    if best_score < opp_score:
                        wins += 1
                    total += 1
            
            return {
                'hand': best_hand,
                'equity': wins / total if total > 0 else 1.0,
                'score': best_score
            }
        
        return {'hand': None, 'equity': 0.5, 'score': float('inf')}


# Utility functions for range analysis
def create_tight_range() -> Dict[str, float]:
    """Create a tight preflop range."""
    range_dict = {
        'AA': 0.45, 'KK': 0.45, 'QQ': 0.45, 'JJ': 0.45, 'TT': 0.45,
        'AKs': 0.45, 'AQs': 0.45, 'AJs': 0.45, 'ATs': 0.45,
        'AKo': 0.45, 'AQo': 0.45, 'AJo': 0.45, 'ATo': 0.45,
        'KQs': 0.45, 'KJs': 0.45, 'KTs': 0.45,
        'KQo': 0.45, 'KJo': 0.45, 'KTo': 0.45,
        'QJs': 0.45, 'QTs': 0.45, 'QJo': 0.45, 'QTo': 0.45,
        'JTs': 0.45, 'JTo': 0.45, 'T9s': 0.45, 'T9o': 0.45
    }
    
    # Normalize weights
    total = sum(range_dict.values())
    return {combo: weight/total for combo, weight in range_dict.items()}


def create_loose_range() -> Dict[str, float]:
    """Create a loose preflop range."""
    range_dict = {}
    
    # Add all pocket pairs
    for rank in RANKS:
        range_dict[rank + rank] = 0.3
    
    # Add suited connectors and gappers
    for i in range(len(RANKS) - 1):
        for j in range(i + 1, min(i + 5, len(RANKS))):
            rank1, rank2 = RANKS[i], RANKS[j]
            range_dict[rank1 + rank2 + 's'] = 0.3
            range_dict[rank1 + rank2 + 'o'] = 0.2
    
    # Add high card hands
    for rank1 in ['A', 'K', 'Q', 'J', 'T']:
        for rank2 in RANKS:
            if rank1 != rank2:
                combo_s = rank1 + rank2 + 's'
                combo_o = rank1 + rank2 + 'o'
                if combo_s not in range_dict:
                    range_dict[combo_s] = 0.2
                if combo_o not in range_dict:
                    range_dict[combo_o] = 0.15
    
    # Normalize weights
    total = sum(range_dict.values())
    return {combo: weight/total for combo, weight in range_dict.items()}


# Example usage
if __name__ == "__main__":
    calculator = EquityCalculator()
    
    # Example hands
    hero_hand = [{'rank': 'A', 'suit': 's'}, {'rank': 'K', 'suit': 'h'}]
    villain_hand = [{'rank': 'Q', 'suit': 'c'}, {'rank': 'J', 'suit': 'd'}]
    board = [
        {'rank': 'A', 'suit': 'c'}, 
        {'rank': 'K', 'suit': 'd'}, 
        {'rank': 'Q', 'suit': 's'},
        {'rank': 'J', 'suit': 'h'},
        {'rank': 'T', 'suit': 'c'}
    ]
    
    # Calculate exact equity
    equity = calculator.calculate_exact_equity(hero_hand, villain_hand, board)
    print(f"Hero's equity vs villain: {equity:.1%}")
    
    # Calculate vs range
    villain_range = create_tight_range()
    range_equity = calculator.calculate_equity_vs_range(hero_hand, villain_range, board)
    print(f"Hero's equity vs tight range: {range_equity:.1%}")
    
    # Calculate hand strength percentile
    percentile = calculator.calculate_hand_strength_percentile(hero_hand, board)
    print(f"Hero's hand strength percentile: {percentile:.1%}")
