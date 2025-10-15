#!/usr/bin/env python3
"""
Script to reorganize open_charts.json by hand type and rank.
Order: pairs, suited aces, offsuit aces, suited kings, offsuit kings, etc.
"""

import json
import sys
from collections import OrderedDict

def get_hand_rank(hand):
    """Get the rank of a hand for sorting purposes."""
    if len(hand) == 2:  # Pair
        rank = hand[0]
        return (0, rank)  # Pairs first
    elif hand.endswith('s'):  # Suited
        rank = hand[:-1]
        return (1, rank)  # Suited second
    else:  # Offsuit
        rank = hand[:-1]
        return (2, rank)  # Offsuit third

def get_rank_value(rank):
    """Convert rank to numeric value for sorting."""
    rank_order = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    try:
        return rank_order.index(rank)
    except ValueError:
        return 999  # Unknown ranks go last

def sort_hands(hands):
    """Sort hands according to the specified order."""
    def sort_key(hand):
        hand_type, rank = get_hand_rank(hand)
        rank_value = get_rank_value(rank)
        return (rank_value, hand_type)  # Group by rank first, then by type
    
    return sorted(hands, key=sort_key)

def reorganize_charts(input_file, output_file):
    """Reorganize the charts file."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        reorganized_data = OrderedDict()
        
        for position, hands in data.items():
            reorganized_position = OrderedDict()
            
            # Sort hands according to the new order
            sorted_hands = sort_hands(hands.keys())
            
            for hand in sorted_hands:
                reorganized_position[hand] = hands[hand]
            
            reorganized_data[position] = reorganized_position
        
        # Write the reorganized data
        with open(output_file, 'w') as f:
            json.dump(reorganized_data, f, indent=2)
        
        print(f"Successfully reorganized {input_file} -> {output_file}")
        
        # Print summary
        for position, hands in reorganized_data.items():
            print(f"\n{position}: {len(hands)} hands")
            # Show first few hands as example
            first_hands = list(hands.keys())[:10]
            print(f"  First 10: {', '.join(first_hands)}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    input_file = "open_charts.json"
    output_file = "open_charts_reorganized.json"
    
    reorganize_charts(input_file, output_file)
