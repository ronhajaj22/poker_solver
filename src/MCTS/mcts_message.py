import AppUtils.StringUtils as strings
from AppUtils.cards_utils import get_trey_card_rank, get_trey_card_suit, pretty_suit

def print_main_player_range(range):
    return
    if range is not None and len(range) > 0:
        print() 
        index = 0;
        for hand, (prob, strength) in range.items():
            card1 = get_trey_card_rank(hand[0]) + pretty_suit(get_trey_card_suit(hand[0]))
            card2 = get_trey_card_rank(hand[1]) + pretty_suit(get_trey_card_suit(hand[1]))
            if (index%20 == 0 or index < 15):
                print(f"{card1}{card2} | prob: {(prob * 100):.2f}%", end="\n")
            index+=1
        print()  # New line after all hands


def create_river_message(root, best_action, hand_strength, position):
    chosen_action = root.children[best_action]
    print(f"MCTS decided {best_action} here because:")
    for action, child in chosen_action.children.items():
        from MCTS.mcts import child_expected_value_for_parent
        expected_score = child_expected_value_for_parent(root, child, position)
        print(f"you: {action} in probability: {child.prior_prob:.3f} expected score: {expected_score:.2f} visit count: {child.visit_count}")
    
    for action, child in chosen_action.children.items():
        from MCTS.mcts import child_expected_value_for_parent
        expected_score = child_expected_value_for_parent(root, child, position)
        if best_action == strings.RAISE:
            if  action == strings.CALL:
                if expected_score < 0:
                    print("Bluff")
                elif expected_score > 1.5:
                    print("Value bet")
                elif expected_score > 0.95:
                    print("Thin value")
                else:
                    print("blocking bet")
        if best_action == strings.CHECK:
            if expected_score > 1 and action == strings.RAISE:
                print("Trap")
            elif expected_score > 0.5 and action == strings.CHECK:
                print("showdown value")
        if best_action == strings.CALL:
            if hand_strength > 0.5:
                print("bluff catcher")
            else:
                print("easy call")