import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import AppUtils.StringUtils as strings
from AppUtils.cards_utils import get_all_combinations
from AppUtils.constants import get_general_position

PRE_FLOP_CHART_CACHE: Optional[Dict] = None
HU_PRE_FLOP_CHART_CACHE: Optional[Dict] = None

def load_preflop_range_chart(is_hu: bool) -> Dict:
    """Load the preflop range chart JSON file and cache its contents."""
    global PRE_FLOP_CHART_CACHE
    global HU_PRE_FLOP_CHART_CACHE
    requsted_chart = PRE_FLOP_CHART_CACHE if not is_hu else HU_PRE_FLOP_CHART_CACHE
    if requsted_chart is None: 
        base_dir = Path(__file__).parent.parent.parent
        chart_path = base_dir / 'data' / 'pre_flop_charts' / (('hu_' if is_hu else '') + 'preflop_range.json')
        with chart_path.open(encoding='utf-8') as f:
            requsted_chart = json.load(f)
    return requsted_chart

def resolve_section_key(is_hu: bool, num_raises: int, is_ip: bool, position: str) -> str:
    """Translate table coordinates into the relevant JSON section key."""
    
    if is_hu:
        hu_section_map = {
            0: 'open' if position == 'SB' else 'vs_limp',
            1: 'vs_open',
            2: 'vs_3bet',
            3: 'vs_4bet',
            4: 'vs_5bet',
        }
        return hu_section_map.get(num_raises, 'vs_5bet')
    
    position = position.upper()

    if num_raises == 0:
        if position == 'BB':
            return 'vs_limp'
        else:
            return f"{position}_RFI"
    elif num_raises > 4:
        num_raises = 4
    
    section_map = {
        1: ('vs_open_ip', 'vs_open_bb'),
        2: ('vs_3bet_ip', 'vs_3bet_oop'),
        3: ('vs_4bet_ip', 'vs_4bet_oop'),
        4: ('vs_5bet_ip', 'vs_5bet_oop'),
    }

    ip_key, oop_key = section_map[min(num_raises, 4)]
 
    return ip_key if is_ip else oop_key


def get_preflop_cell(is_hu: bool, num_raises: int, is_ip: bool, position: str, action: str, amount=0) -> Optional[Dict[str, float]]:
    """
    Return the table cell (dictionary of hands -> frequency) for the given coordinates.
    """
    chart = load_preflop_range_chart(is_hu)
    section_key = resolve_section_key(is_hu, num_raises, is_ip, position)
    section = chart.get(section_key)

    if section is None:
        return None
    
    action_key = action.lower()
    
    # Note: this is a very special case for HU games when the size is decided from preflop chart
    if is_hu and position == 'BB' and num_raises == 0 and action == strings.RAISE:
        if amount > 3:
            action_key = 'raise_big'

    if isinstance(section, dict) and action_key in section and isinstance(section[action_key], dict):
        return section[action_key]

    # Some legacy sections may already be flat dictionaries
    return section if num_raises == 0 else None


def create_range(action, is_hu: bool, is_ip, position, num_raises, raise_is_not_possible=False, amount=0):
    range = {}
    position = get_general_position(position)
    hands_probability_map = get_preflop_cell(is_hu, num_raises, is_ip, position, action, amount)

    # Note; this doesn't really matter. when raise is not possible, the range doesn't matter
    if (raise_is_not_possible):
        pass
    sum_values = 0
    if not hands_probability_map:
        print("Error: hands probability map is empty")
        return range

    for hand_str, probability in hands_probability_map.items():
        if probability and isinstance(probability, (float, int)) and probability > 0:
            for combination in get_all_combinations(hand_str):
                range[combination] = (probability, 0)  # (prob, strength) - strength is 0 until calculated
                sum_values += probability

    if sum_values == 0:
        return range

    # Normalize probabilities
    for combination in range:
        old_prob, strength = range[combination]
        range[combination] = (old_prob / sum_values, strength)

    return range