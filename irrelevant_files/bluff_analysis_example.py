"""
Practical Example of Bluff Analysis System

This example demonstrates how to use the integrated bluff analysis system
with real poker scenarios.
"""

from integrated_bluff_analyzer import IntegratedBluffAnalyzer
import json


def run_bluff_analysis_examples():
    """Run comprehensive bluff analysis examples."""
    
    analyzer = IntegratedBluffAnalyzer()
    
    print("=== POKER BLUFF ANALYSIS SYSTEM ===\n")
    
    # Example 1: River Bluff Analysis
    print("1. RIVER BLUFF ANALYSIS")
    print("-" * 30)
    
    hero_hand = [{'rank': 'A', 'suit': 's'}, {'rank': 'K', 'suit': 'h'}]
    board = [
        {'rank': 'A', 'suit': 'c'}, 
        {'rank': 'K', 'suit': 'd'}, 
        {'rank': 'Q', 'suit': 's'},
        {'rank': 'J', 'suit': 'h'},
        {'rank': 'T', 'suit': 'c'}
    ]
    
    analysis1 = analyzer.analyze_river_bet(
        hero_hand=hero_hand,
        board=board,
        pot_size=100.0,
        bet_size=50.0,
        opponent_position='BTN',
        opponent_actions=['CALL', 'CALL', 'CALL'],
        opponent_profile='balanced'
    )
    
    print_analysis_results(analysis1, "River Bet Analysis", analyzer)
    
    # Example 2: Different Bet Sizes
    print("\n2. BET SIZE COMPARISON")
    print("-" * 30)
    
    bet_sizes = [25.0, 50.0, 75.0, 100.0]  # 25%, 50%, 75%, 100% pot
    
    for bet_size in bet_sizes:
        analysis = analyzer.analyze_river_bet(
            hero_hand=hero_hand,
            board=board,
            pot_size=100.0,
            bet_size=bet_size,
            opponent_position='BTN',
            opponent_actions=['CALL', 'CALL', 'CALL'],
            opponent_profile='balanced'
        )
        
        print(f"Bet Size: {bet_size} ({bet_size/100.0:.0%} pot)")
        print(f"  Classification: {analysis['classification']}")
        print(f"  EV: {analysis['expected_value']:.2f}")
        print(f"  Fold Equity: {analysis['fold_equity']:.1%}")
        print()
    
    # Example 3: Different Opponent Profiles
    print("3. OPPONENT PROFILE COMPARISON")
    print("-" * 30)
    
    profiles = ['tight', 'loose', 'aggressive', 'passive', 'balanced']
    
    for profile in profiles:
        analysis = analyzer.analyze_river_bet(
            hero_hand=hero_hand,
            board=board,
            pot_size=100.0,
            bet_size=50.0,
            opponent_position='BTN',
            opponent_actions=['CALL', 'CALL', 'CALL'],
            opponent_profile=profile
        )
        
        print(f"Profile: {profile.upper()}")
        print(f"  Classification: {analysis['classification']}")
        print(f"  EV: {analysis['expected_value']:.2f}")
        print(f"  Fold Equity: {analysis['fold_equity']:.1%}")
        print()
    
    # Example 4: Different Board Textures
    print("4. BOARD TEXTURE ANALYSIS")
    print("-" * 30)
    
    boards = [
        # Dry board
        [
            {'rank': 'A', 'suit': 'c'}, 
            {'rank': 'K', 'suit': 'd'}, 
            {'rank': 'Q', 'suit': 's'},
            {'rank': 'J', 'suit': 'h'},
            {'rank': 'T', 'suit': 'c'}
        ],
        # Wet board
        [
            {'rank': '9', 'suit': 'c'}, 
            {'rank': '8', 'suit': 'c'}, 
            {'rank': '7', 'suit': 's'},
            {'rank': '6', 'suit': 'h'},
            {'rank': '5', 'suit': 'c'}
        ],
        # Paired board
        [
            {'rank': 'A', 'suit': 'c'}, 
            {'rank': 'A', 'suit': 'd'}, 
            {'rank': 'K', 'suit': 's'},
            {'rank': 'Q', 'suit': 'h'},
            {'rank': 'J', 'suit': 'c'}
        ]
    ]
    
    board_names = ['Dry Board (AKQJT)', 'Wet Board (98765)', 'Paired Board (AAKQJ)']
    
    for board, name in zip(boards, board_names):
        analysis = analyzer.analyze_river_bet(
            hero_hand=hero_hand,
            board=board,
            pot_size=100.0,
            bet_size=50.0,
            opponent_position='BTN',
            opponent_actions=['CALL', 'CALL', 'CALL'],
            opponent_profile='balanced'
        )
        
        print(f"Board: {name}")
        print(f"  Classification: {analysis['classification']}")
        print(f"  EV: {analysis['expected_value']:.2f}")
        print(f"  Fold Equity: {analysis['fold_equity']:.1%}")
        print()
    
    # Example 5: Complete Hand Analysis
    print("5. COMPLETE HAND ANALYSIS")
    print("-" * 30)
    
    hand_data = {
        'hero_hand': [{'rank': 'A', 'suit': 's'}, {'rank': 'K', 'suit': 'h'}],
        'board': [
            {'rank': 'A', 'suit': 'c'}, 
            {'rank': 'K', 'suit': 'd'}, 
            {'rank': 'Q', 'suit': 's'},
            {'rank': 'J', 'suit': 'h'},
            {'rank': 'T', 'suit': 'c'}
        ],
        'actions': ['BET', 'CALL', 'BET', 'CALL', 'BET'],
        'pot_sizes': [10.0, 20.0, 40.0, 80.0, 160.0],
        'bet_sizes': [10.0, 20.0, 40.0, 80.0, 80.0]
    }
    
    hand_analysis = analyzer.analyze_hand_history(hand_data)
    
    print("Hand Summary:")
    summary = hand_analysis['summary']
    print(f"  Total Bets: {summary['total_bets']}")
    print(f"  Bluff Count: {summary['bluff_count']}")
    print(f"  Value Count: {summary['value_count']}")
    print(f"  Thin Count: {summary['thin_count']}")
    print(f"  Average EV: {summary['average_ev']:.2f}")
    print(f"  Average Fold Equity: {summary['average_fold_equity']:.1%}")
    
    # Example 6: Export Analysis
    print("\n6. EXPORTING ANALYSIS")
    print("-" * 30)
    
    # Export the river analysis
    analyzer.export_analysis(analysis1, 'river_bluff_analysis.json')
    
    # Export the hand analysis
    analyzer.export_analysis(hand_analysis, 'complete_hand_analysis.json')
    
    print("Analysis exported to JSON files!")


def print_analysis_results(analysis, title, analyzer=None):
    """Print analysis results in a formatted way."""
    print(f"{title}:")
    print(f"  Classification: {analysis['classification']}")
    print(f"  Expected Value: {analysis['expected_value']:.2f}")
    print(f"  Fold Equity: {analysis['fold_equity']:.1%}")
    print(f"  Hero Equity vs Calling: {analysis['hero_equity_vs_calling']:.1%}")
    print(f"  Break-even Equity: {analysis['break_even_equity']:.1%}")
    print(f"  Required Fold Equity: {analysis['required_fold_equity']:.1%}")
    print(f"  Opponent Range Size: {analysis['opponent_range_size']} combos")
    print(f"  Calling Range Size: {analysis['calling_range_size']} combos")
    
    # Print recommendations
    if analyzer:
        recommendations = analyzer.get_bluff_recommendations(analysis)
        if recommendations:
            print("  Recommendations:")
            for rec in recommendations:
                print(f"    {rec}")


def demonstrate_mathematical_formulas():
    """Demonstrate the mathematical formulas from the user's specification."""
    
    print("\n=== MATHEMATICAL FORMULAS DEMONSTRATION ===")
    
    analyzer = IntegratedBluffAnalyzer()
    
    # Example scenario from user's specification
    P = 100  # Pot before bet
    B = 50   # Half pot bet
    
    hero_hand = [{'rank': 'A', 'suit': 's'}, {'rank': 'K', 'suit': 'h'}]
    board = [
        {'rank': 'A', 'suit': 'c'}, 
        {'rank': 'K', 'suit': 'd'}, 
        {'rank': 'Q', 'suit': 's'},
        {'rank': 'J', 'suit': 'h'},
        {'rank': 'T', 'suit': 'c'}
    ]
    
    # Build opponent range
    opponent_range = analyzer.range_builder.build_initial_range('BTN', 'CALL')
    opponent_range = analyzer.range_builder.update_range_for_board(opponent_range, board)
    
    # Calculate call probabilities
    call_probs = analyzer.call_probability_estimator.estimate_range_call_probabilities(
        opponent_range, board, B/P, 'balanced', 'BTN'
    )
    
    # Calculate metrics
    CallTotal = sum(weight * call_probs.get(combo, 0.5) for combo, weight in opponent_range.items())
    FE = max(0.0, 1.0 - CallTotal)
    
    # Calculate eq_call
    weighted_eq_sum = 0.0
    for combo, weight in opponent_range.items():
        call_prob = call_probs.get(combo, 0.5)
        call_weight = weight * call_prob
        if call_weight > 0:
            hand = analyzer.range_builder._combo_to_cards(combo)
            if hand:
                equity = analyzer.equity_calculator.calculate_exact_equity(hero_hand, hand, board)
                weighted_eq_sum += call_weight * equity
    
    eq_call = weighted_eq_sum / CallTotal if CallTotal > 0 else 0.5
    
    # Calculate break-even equity
    beq = B / (P + 2 * B)
    
    # Calculate FE_needed
    numerator = B - eq_call * (P + 2 * B)
    denominator = (P + B) - eq_call * (P + 2 * B)
    FE_needed = max(0.0, numerator / denominator) if denominator > 1e-12 else 0.0
    
    # Calculate EV
    ev = analyzer.bluff_classifier.calculate_ev(P, B, FE, eq_call)
    
    print(f"Mathematical Analysis:")
    print(f"  P (Pot): {P}")
    print(f"  B (Bet): {B}")
    print(f"  CallTotal: {CallTotal:.3f}")
    print(f"  FE (Fold Equity): {FE:.1%}")
    print(f"  eq_call: {eq_call:.1%}")
    print(f"  beq (Break-even): {beq:.1%}")
    print(f"  FE_needed: {FE_needed:.1%}")
    print(f"  EV: {ev:.2f}")
    
    # Classification
    if CallTotal == 0:
        classification = "BLUFF"
    elif eq_call >= beq + 0.03:
        classification = "VALUE"
    elif FE >= FE_needed + 0.03:
        classification = "BLUFF"
    else:
        classification = "THIN"
    
    print(f"  Classification: {classification}")
    
    # Verify with our system
    p_call_func = analyzer.call_probability_estimator.create_call_probability_function(
        board, B/P, 'balanced', 'BTN'
    )
    
    system_classification = analyzer.bluff_classifier.classify_river_bet(
        P, B, opponent_range, p_call_func, hero_hand, board
    )
    
    print(f"  System Classification: {system_classification}")
    print(f"  Match: {'✅' if classification == system_classification else '❌'}")


if __name__ == "__main__":
    # Run the examples
    run_bluff_analysis_examples()
    
    # Demonstrate mathematical formulas
    demonstrate_mathematical_formulas()
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("The bluff analysis system is ready for use!")
    print("Key features:")
    print("✅ Mathematical bluff classification")
    print("✅ Equity calculations")
    print("✅ Range building and updating")
    print("✅ Call probability estimation")
    print("✅ Integration with existing system")
    print("✅ Export capabilities")
