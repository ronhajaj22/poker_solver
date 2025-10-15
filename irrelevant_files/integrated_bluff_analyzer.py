"""
Integrated Bluff Analysis System

This module integrates all bluff analysis components with the existing poker system,
providing a comprehensive solution for bluff detection and analysis.
"""

from typing import Dict, List, Tuple, Optional, Any
from bluff_classifier import BluffClassifier
from equity_calculator import EquityCalculator
from range_builder import RangeBuilder
from call_probability_estimator import CallProbabilityEstimator
from AppUtils.hand_board_calculations import calc_hand_strength, calc_hand_type
from AppUtils.constants import card_to_str
import json


class IntegratedBluffAnalyzer:
    """
    Main class that integrates all bluff analysis components.
    """
    
    def __init__(self):
        self.bluff_classifier = BluffClassifier()
        self.equity_calculator = EquityCalculator()
        self.range_builder = RangeBuilder()
        self.call_probability_estimator = CallProbabilityEstimator()
        
        # Load existing player stats if available
        self.player_stats_cache = {}
        self.load_player_stats()
    
    def analyze_river_bet(self, 
                         hero_hand: List[dict],
                         board: List[dict],
                         pot_size: float,
                         bet_size: float,
                         opponent_position: str = 'BTN',
                         opponent_actions: List[str] = None,
                         opponent_profile: str = 'balanced',
                         opponent_stats: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a river bet for bluff detection.
        
        Args:
            hero_hand: Hero's cards
            board: Board cards
            pot_size: Pot size before bet
            bet_size: Bet size
            opponent_position: Opponent's position
            opponent_actions: List of opponent's actions in the hand
            opponent_profile: Opponent's playing style profile
            opponent_stats: Opponent's statistics
            
        Returns:
            Comprehensive analysis results
        """
        if opponent_actions is None:
            opponent_actions = []
        
        if opponent_stats is None:
            opponent_stats = {}
        
        # 1. Build opponent's range
        opponent_range = self._build_opponent_range(opponent_position, opponent_actions, opponent_profile)
        
        # 2. Update range for board texture
        opponent_range = self.range_builder.update_range_for_board(opponent_range, board)
        
        # 3. Calculate call probabilities
        bet_size_ratio = bet_size / pot_size
        call_probabilities = self.call_probability_estimator.estimate_range_call_probabilities(
            opponent_range, board, bet_size_ratio, opponent_profile, opponent_position
        )
        
        # 4. Create call probability function
        p_call_func = self.call_probability_estimator.create_call_probability_function(
            board, bet_size_ratio, opponent_profile, opponent_position
        )
        
        # 5. Classify the bet
        classification = self.bluff_classifier.classify_river_bet(
            pot_size, bet_size, opponent_range, p_call_func, hero_hand, board
        )
        
        # 6. Calculate additional metrics
        fold_equity = self.call_probability_estimator.get_fold_equity(opponent_range, call_probabilities)
        calling_range = self.call_probability_estimator.get_calling_range_weights(opponent_range, call_probabilities)
        
        # 7. Calculate hero's equity vs calling range
        hero_equity_vs_calling = self.equity_calculator.calculate_equity_vs_calling_range(
            hero_hand, opponent_range, board, call_probabilities
        )
        
        # 8. Calculate expected value
        ev = self.bluff_classifier.calculate_ev(pot_size, bet_size, fold_equity, hero_equity_vs_calling)
        
        # 9. Calculate break-even metrics
        break_even_equity = self.bluff_classifier.get_break_even_equity(pot_size, bet_size)
        required_fold_equity = self.bluff_classifier.get_required_fold_equity(
            pot_size, bet_size, hero_equity_vs_calling
        )
        
        # 10. Generate analysis summary
        analysis = {
            'classification': classification,
            'expected_value': ev,
            'fold_equity': fold_equity,
            'hero_equity_vs_calling': hero_equity_vs_calling,
            'break_even_equity': break_even_equity,
            'required_fold_equity': required_fold_equity,
            'opponent_range_size': len(opponent_range),
            'calling_range_size': len(calling_range),
            'bet_size_ratio': bet_size_ratio,
            'analysis_details': {
                'opponent_range': opponent_range,
                'call_probabilities': call_probabilities,
                'calling_range': calling_range,
                'hero_hand_strength': self.equity_calculator.calculate_hand_strength_percentile(hero_hand, board)
            }
        }
        
        return analysis
    
    def analyze_hand_history(self, hand_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a complete hand for bluff detection opportunities.
        
        Args:
            hand_data: Complete hand data including actions, board, etc.
            
        Returns:
            Analysis of bluff opportunities throughout the hand
        """
        # Extract hand information
        hero_hand = hand_data.get('hero_hand', [])
        board = hand_data.get('board', [])
        actions = hand_data.get('actions', [])
        pot_sizes = hand_data.get('pot_sizes', [])
        bet_sizes = hand_data.get('bet_sizes', [])
        
        analysis_results = []
        
        # Analyze each betting round
        for i, (action, pot_size, bet_size) in enumerate(zip(actions, pot_sizes, bet_sizes)):
            if action == 'BET' or action == 'RAISE':
                analysis = self.analyze_river_bet(
                    hero_hand, board, pot_size, bet_size
                )
                analysis['street'] = self._get_street_from_board_length(len(board))
                analysis['action_index'] = i
                analysis_results.append(analysis)
        
        return {
            'hand_analysis': analysis_results,
            'summary': self._generate_hand_summary(analysis_results)
        }
    
    def load_player_stats(self):
        """Load existing player statistics from the system."""
        try:
            # Try to load from existing stats files
            stats_files = [
                'src/gg_hands_parser/villains_stats_4/Hero_hu_4_stats.txt',
                'src/gg_hands_parser/villains_stats_4/player_4_stats.txt'
            ]
            
            for stats_file in stats_files:
                try:
                    with open(stats_file, 'r') as f:
                        # Parse stats file (simplified)
                        stats = self._parse_stats_file(f.read())
                        player_name = stats_file.split('/')[-1].replace('_stats.txt', '')
                        self.player_stats_cache[player_name] = stats
                except FileNotFoundError:
                    continue
        except Exception as e:
            print(f"Warning: Could not load player stats: {e}")
    
    def _parse_stats_file(self, content: str) -> Dict[str, float]:
        """Parse player stats file (simplified version)."""
        stats = {}
        
        # Simple parsing - in practice, you'd want more sophisticated parsing
        lines = content.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to extract numeric values
                try:
                    if '%' in value:
                        numeric_value = float(value.replace('%', '')) / 100.0
                    else:
                        numeric_value = float(value)
                    stats[key] = numeric_value
                except ValueError:
                    continue
        
        return stats
    
    def _build_opponent_range(self, position: str, actions: List[str], profile: str) -> Dict[str, float]:
        """Build opponent's range based on position and actions."""
        # Start with initial range based on position
        if actions:
            initial_action = actions[0]
        else:
            initial_action = 'CALL'  # Default
        
        range_dict = self.range_builder.build_initial_range(position, initial_action)
        
        # Update range based on subsequent actions
        for action in actions[1:]:
            range_dict = self.range_builder.update_range_for_action(
                range_dict, action, [], None, position
            )
        
        return range_dict
    
    def _get_street_from_board_length(self, board_length: int) -> str:
        """Get street name from board length."""
        if board_length == 0:
            return 'PREFLOP'
        elif board_length == 3:
            return 'FLOP'
        elif board_length == 4:
            return 'TURN'
        elif board_length == 5:
            return 'RIVER'
        else:
            return 'UNKNOWN'
    
    def _generate_hand_summary(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of hand analysis."""
        if not analysis_results:
            return {'total_bets': 0, 'bluff_count': 0, 'value_count': 0, 'thin_count': 0}
        
        total_bets = len(analysis_results)
        bluff_count = sum(1 for analysis in analysis_results if analysis['classification'] == 'BLUFF')
        value_count = sum(1 for analysis in analysis_results if analysis['classification'] == 'VALUE')
        thin_count = sum(1 for analysis in analysis_results if analysis['classification'] == 'THIN')
        
        avg_ev = sum(analysis['expected_value'] for analysis in analysis_results) / total_bets
        avg_fold_equity = sum(analysis['fold_equity'] for analysis in analysis_results) / total_bets
        
        return {
            'total_bets': total_bets,
            'bluff_count': bluff_count,
            'value_count': value_count,
            'thin_count': thin_count,
            'bluff_percentage': bluff_count / total_bets if total_bets > 0 else 0,
            'value_percentage': value_count / total_bets if total_bets > 0 else 0,
            'thin_percentage': thin_count / total_bets if total_bets > 0 else 0,
            'average_ev': avg_ev,
            'average_fold_equity': avg_fold_equity
        }
    
    def get_bluff_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate bluff recommendations based on analysis."""
        recommendations = []
        
        classification = analysis['classification']
        ev = analysis['expected_value']
        fold_equity = analysis['fold_equity']
        hero_equity = analysis['hero_equity_vs_calling']
        
        if classification == 'BLUFF':
            if ev > 0:
                recommendations.append("✅ Good bluff - positive expected value")
            else:
                recommendations.append("⚠️ Bluff with negative EV - consider sizing or timing")
        
        elif classification == 'VALUE':
            if ev > 0:
                recommendations.append("✅ Good value bet - opponent likely to call with worse")
            else:
                recommendations.append("⚠️ Value bet with negative EV - opponent may be too tight")
        
        elif classification == 'THIN':
            recommendations.append("🤔 Thin spot - consider opponent's tendencies")
            if fold_equity > 0.6:
                recommendations.append("💡 High fold equity - could be profitable as bluff")
            elif hero_equity > 0.4:
                recommendations.append("💡 Decent equity - could be profitable as value")
        
        # Additional recommendations
        if fold_equity > 0.8:
            recommendations.append("🎯 Very high fold equity - opponent likely to fold")
        elif fold_equity < 0.3:
            recommendations.append("⚠️ Low fold equity - opponent likely to call")
        
        if hero_equity > 0.7:
            recommendations.append("💪 Strong equity vs calling range")
        elif hero_equity < 0.3:
            recommendations.append("😰 Weak equity vs calling range")
        
        return recommendations
    
    def export_analysis(self, analysis: Dict[str, Any], filename: str):
        """Export analysis results to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"Analysis exported to {filename}")
        except Exception as e:
            print(f"Error exporting analysis: {e}")


# Example usage and integration
if __name__ == "__main__":
    analyzer = IntegratedBluffAnalyzer()
    
    # Example river analysis
    hero_hand = [{'rank': 'A', 'suit': 's'}, {'rank': 'K', 'suit': 'h'}]
    board = [
        {'rank': 'A', 'suit': 'c'}, 
        {'rank': 'K', 'suit': 'd'}, 
        {'rank': 'Q', 'suit': 's'},
        {'rank': 'J', 'suit': 'h'},
        {'rank': 'T', 'suit': 'c'}
    ]
    
    analysis = analyzer.analyze_river_bet(
        hero_hand=hero_hand,
        board=board,
        pot_size=100.0,
        bet_size=50.0,
        opponent_position='BTN',
        opponent_actions=['CALL', 'CALL', 'CALL'],
        opponent_profile='balanced'
    )
    
    print("=== BLUFF ANALYSIS RESULTS ===")
    print(f"Classification: {analysis['classification']}")
    print(f"Expected Value: {analysis['expected_value']:.2f}")
    print(f"Fold Equity: {analysis['fold_equity']:.1%}")
    print(f"Hero Equity vs Calling: {analysis['hero_equity_vs_calling']:.1%}")
    print(f"Break-even Equity: {analysis['break_even_equity']:.1%}")
    print(f"Required Fold Equity: {analysis['required_fold_equity']:.1%}")
    
    print("\n=== RECOMMENDATIONS ===")
    recommendations = analyzer.get_bluff_recommendations(analysis)
    for rec in recommendations:
        print(rec)
    
    # Export analysis
    analyzer.export_analysis(analysis, 'bluff_analysis_example.json')
