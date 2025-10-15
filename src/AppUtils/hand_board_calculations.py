from collections import Counter
from treys import Evaluator, Card
from itertools import combinations
from AppUtils.constants import TURN
import pdb
from AppUtils.cards_utils import card_to_str, trey_to_str, RANKS, SUITS, CARD_RANK_TO_INT_VALUE

# TODO - make sute the hand is sorted for all function
# This function gets a hand and a board and return how strong this hand against all other combination
# format: Card: {rank: 2, suit: 's'}
# the higher the score (0-100), the stronger the hand
def calc_hand_strength(hand = [], board = [], opponent_range = []):
    evaluator = Evaluator()
    
    # Convert hero hand and board to treys format
    hero_cards = [Card.new(card_to_str(card)) for card in hand]
    board_cards = [Card.new(card_to_str(card)) for card in board]
    
    # Create full deck in treys format
    full_deck = [Card.new(rank + suit) 
                for rank in RANKS 
                for suit in SUITS]
    
    # Filter out used cards
    used_cards = set(hero_cards + board_cards)
    available_cards = [c for c in full_deck if c not in used_cards]
    
    # Generate all possible opponent hands
    all_opponent_hands = list(combinations(available_cards, 2))

    if opponent_range == []:
        # If no opponent range specified, use all possible hands
        opponent_updated_range = all_opponent_hands
    else:
        # Filter hands to only include those in the opponent range
        opponent_updated_range = []
        for hand_combo in all_opponent_hands:
            hand_str = trey_to_str(hand_combo)
            if hand_str in opponent_range:
                opponent_updated_range.append(hand_combo)

    hero_score = evaluator.evaluate(board_cards, hero_cards)
    
    total = 0
    worse = 0
    for opp_hand in opponent_updated_range:
        opp_score = evaluator.evaluate(board_cards, list(opp_hand))
        if hero_score < opp_score:
            worse += 1
        total += 1

    if total == 0:
        return 0  # Avoid division by zero
    
    percentile = 100 * worse / total
    
    return round(percentile, 2)


# gets hand cards, board cards and returns hand type and hand ranks (for example: "Straight Flush", "Four of a Kind", "Full House", "Flush", "Straight", "Three of a Kind", "Two Pair", "Pair", "High Card")
# format: Card: {rank: 2, suit: 's'}
def calc_hand_type(hand_cards, board_cards):
    all_cards = hand_cards + board_cards
    
    # Get unique ranks and count them
    rank_counts = Counter(card.rank for card in all_cards)
    sorted_ranks = sorted(rank_counts.keys(), key=lambda x: RANKS.index(x), reverse=True)
    
    # Get unique suits and count them
    suit_counts = Counter(card.suit for card in all_cards)
    
    # Check for pairs, three of a kind, four of a kind
    four_of_kind_rank = None
    three_of_kind_rank = None
    pair_ranks = []
    
    for rank, count in rank_counts.items():
        if count == 4:
            four_of_kind_rank = rank
        elif count == 3:
            three_of_kind_rank = rank
        elif count == 2:
            pair_ranks.append(rank)
    
    # Check for flush
    flush_suit = None
    for suit, count in suit_counts.items():
        if count >= 5:
            flush_suit = suit
            break
    
    # Check for straight
    check_straight, straight_ranks = is_straight(hand_cards, board_cards)

    # Check for straight flush
    is_straight_flush = False
    if check_straight and flush_suit:
        # Check if the straight cards are all the same suit
        straight_cards = [card for card in all_cards if card.rank in straight_ranks]
        if len(straight_cards) >= 5:
            flush_straight_cards = [card for card in straight_cards if card.suit == flush_suit]
            if len(flush_straight_cards) >= 5:
                is_straight_flush = True
    
    # Determine hand type and return
    if is_straight_flush:
        return "Straight Flush", straight_ranks
    elif four_of_kind_rank:
        return "Four of a Kind", [four_of_kind_rank]
    elif three_of_kind_rank and pair_ranks:
        return "Full House", [three_of_kind_rank, pair_ranks[0]]
    elif flush_suit:
        flush_cards = [card.rank for card in all_cards if card.suit == flush_suit]
        return "Flush", sorted(flush_cards, key=lambda x: RANKS.index(x), reverse=True)[:5]
    elif check_straight:
        return "Straight", straight_ranks
    elif three_of_kind_rank:
        return "Three of a Kind", [three_of_kind_rank]
    elif len(pair_ranks) >= 2:
        return "Two Pair", pair_ranks[:2]
    elif len(pair_ranks) == 1:
        return "Pair", [pair_ranks[0]]
    else:
        return "High Card", sorted_ranks[:5]

# CHECK DRAW FUNCTIONS

# this is the main straight draw function - check if there is a straight (TRUE/FALSE)
# working for all streets
# format: Card: {rank: 'A', suit: 's'}
def is_straight(hand_cards, board_cards):
    # Get all unique ranks sorted
    all_cards = hand_cards + board_cards
    is_straight = False
    straight_ranks = []
    unique_ranks = sorted(set(card.rank for card in all_cards), key=lambda x: RANKS.index(x))
    
    # Check for regular straights
    for i in range(len(unique_ranks) - 4):
        potential_straight = unique_ranks[i:i+5]
        if is_straight_group(potential_straight):
            is_straight = True
            straight_ranks = potential_straight
            break
    
    # Check for A2345 straight (wheel)
    if not is_straight and len(unique_ranks) >= 5:
        wheel_ranks = ['2', '3', '4', '5', 'A']
        if all(rank in unique_ranks for rank in wheel_ranks):
            is_straight = True
            straight_ranks = wheel_ranks
    
    return is_straight, straight_ranks

# sub-function for is_straight - check if a group of cards (len 5) is a straight
def is_straight_group(group):
    group = sorted(group, key=lambda x: RANKS.index(x))
    
    # Check for A2345 straight (wheel) first
    if len(group) == 5 and group[0] == '2' and group[1] == '3' and group[2] == '4' and group[3] == '5' and group[-1] == 'A':
        return True
        
    # Check for regular consecutive straights
    for i in range(len(group) - 1):
        if RANKS.index(group[i]) - RANKS.index(group[i + 1]) != 1:
            return False
    return True

# this function check if we have any kind of straight DRAW
# woking mainly for flop, maybe for turn (no need for river)
# format: Card: {rank: 'A', suit: 's'}
def calculate_straight_draw(hand_cards, flop_cards, turn_card = None):
    STRIGHT = 1
    OVER_CARDS_STRIGHT_DRAW = 0.9 # 78 on 56A
    OPEN_ENDED_STRAIGHT_DRAW = 0.85 # 79 on 8T2
    LOWER_CARDS_STRIGHT_DRAW = 0.75 # 34 on 56A
    ONE_OVERCARD_STRAIGHT_DRAW = 0.65 # 7 on 456
    ONE_LOWER_CARD_STRIGHT_DRAW = 0.45 # 4 on 567
    GUT_SHOT_HIGH_CARD_DRAW = 0.35 # 9 on 567 board or 9 in 578
    GUT_SHOT_STRAIGHT_DRAW = 0.3 # 8 on 569 board
    GUT_SHOT_LOW_CARD_DRAW = 0.1 # 2 on 456 board
    NO_STRAIGHT_DRAW = 0

    all_cards = hand_cards + flop_cards + ([turn_card] if turn_card else [])

    # Check if we already have a straight
    if is_straight(hand_cards, all_cards)[0]:
        return STRIGHT

    all_ranks_numbers = [CARD_RANK_TO_INT_VALUE[card.rank] for card in all_cards]
    unique_ranks = sorted(list(set(all_ranks_numbers)))
    board_unique_ranks = list(set([CARD_RANK_TO_INT_VALUE[card.rank] for card in flop_cards]))

    if len(unique_ranks) < 4:
        return NO_STRAIGHT_DRAW

    # special case - stright is option with 3 cards on board
    if len(board_unique_ranks) == 3 and board_unique_ranks[0] + 2 == board_unique_ranks[2]: # i.e 3 4 5
        if (calc_prev_card(board_unique_ranks[0]) in [card.rank for card in hand_cards]):
            return ONE_LOWER_CARD_STRIGHT_DRAW
        elif (board_unique_ranks[2] != 14 and board_unique_ranks[2]+1 in [card.rank for card in hand_cards]):
            return ONE_OVERCARD_STRAIGHT_DRAW
        elif (board_unique_ranks[2] < 13 and board_unique_ranks[2]+2 in [card.rank for card in hand_cards]):
            return GUT_SHOT_HIGH_CARD_DRAW
        elif (board_unique_ranks[0] > 2 and calc_prev_card(board_unique_ranks[0]-1) in [card.rank for card in hand_cards]):
            return GUT_SHOT_LOW_CARD_DRAW
    
    # Handle Ace-low straights (A2345)
    if 14 in all_ranks_numbers:  # If we have an Ace
        if (unique_ranks[2] == 5 or unique_ranks[2] == 4): # 235 or 2 3 4
            if unique_ranks[2] == 5 and unique_ranks[2] in [card.rank for card in hand_cards]: # 5 in hero's hans
                return GUT_SHOT_HIGH_CARD_DRAW
            else:
                return GUT_SHOT_STRAIGHT_DRAW

    gap = 3
    # Open ended straight draw
    found = False
    for i in range(len(unique_ranks) - gap): # 0 1 2 for 3, 0 1 for 4
        if unique_ranks[i] + gap == unique_ranks[i+gap]:
            found = True
            break
    if found:
        if unique_ranks[len(unique_ranks)-1] == 14:
            return GUT_SHOT_STRAIGHT_DRAW 
        else:
            return OPEN_ENDED_STRAIGHT_DRAW
    
    gap += 1
    # check gut shot straight draw
    for i in range(len(unique_ranks) - 3): # 0 1 2 for 3, 0 1 for 4
        if unique_ranks[i] + gap == unique_ranks[i+3]:
            found = True
            break
    if found:
        return GUT_SHOT_STRAIGHT_DRAW 

    return NO_STRAIGHT_DRAW

# sub-function for calculate_straight_draw
def calc_prev_card(rank):
    if rank == 2:
        return 14
    else:
        return rank - 1

# this function check if we have any kind of flush draw
# working only in the flop
# format: Card: {rank: 'A', suit: 's'}
def calculate_flush_draw(hand_cards, flop_cards):
    NO_FLUSH_DRAW = 0
    TWO_ON_BOARD_ONE_IN_HAND_BACKDOOR_FLUSH_DRAW = 0.2
    ONE_ON_BOARD_TWO_IN_HAND_BACKDOOR_FLUSH_DRAW = 0.3
    THREE_ON_BOARD_ONE_IN_HAND_FLUSH_DRAW = 0.65
    TWO_ON_BOARD_TWO_IN_HAND_FLUSH_DRAW = 0.8
    FLUSH = 1.0
    
    hand_cards = sorted(hand_cards, key=lambda x: RANKS.index(x.rank), reverse=True)
    suit_counts = Counter(c.suit for c in hand_cards + flop_cards)
    ans = 0
    if max(suit_counts.values()) < 3:
        ans = NO_FLUSH_DRAW
    elif max(suit_counts.values()) == 5:
        ans = FLUSH
    elif max(suit_counts.values()) == 4:
        if hand_cards[0].suit == hand_cards[1].suit:
            ans = TWO_ON_BOARD_TWO_IN_HAND_FLUSH_DRAW
            if hand_cards[0].rank == 'A':
                ans += 0.1 # nuts flush draw with A
            elif hand_cards[0].rank == 'K':
                flop_cards_of_suit = [c for c in flop_cards if c.suit == hand_cards[0].suit]
                highest_card_in_board = max(c.rank for c in flop_cards_of_suit)
                if highest_card_in_board == 'A':
                    ans += 0.5 # nuts flush draw with K
            else:
                # TODO - not accurate - need to improve
                ans -= (RANKS.index('A') - RANKS.index(hand_cards[0].rank))*0.0125 # flush draw
        else:
            ans = THREE_ON_BOARD_ONE_IN_HAND_FLUSH_DRAW
            board_suit = flop_cards[0].suit
            card_value = hand_cards[0].rank if hand_cards[0].suit == board_suit else hand_cards[1].rank
            if card_value == 'A':
                ans += 0.1 # nuts flush draw with A
            else:
                board_card_to_rank = [c.rank for c in flop_cards]
                highest_card_in_board = max(RANKS.index(rank) for rank in board_card_to_rank)
                highest_possible = 'A'
                if highest_card_in_board == 'A':
                    if 'K' in board_card_to_rank:
                        highest_possible = 'Q' if 'Q' in board_card_to_rank else 'J'
                    else:
                        highest_possible = 'K'
                ans -= (highest_possible - card_value)*0.06 # nuts flush draw with K
                ans = min(ans, 0.2)

    elif suit_counts[hand_cards[0].suit] == 3 or suit_counts[hand_cards[1].suit] == 3:
        suited_card = hand_cards[0] if suit_counts[hand_cards[0].suit] == 3 else hand_cards[1]
        board_card_to_rank = [c.rank for c in flop_cards if c.suit == suited_card.suit]
        highest_card_in_board = max(board_card_to_rank)
        highest_possible = 'K' if highest_card_in_board == 'A' else 'A'
        if hand_cards[0].suit == suited_card.suit:
            ans = ONE_ON_BOARD_TWO_IN_HAND_BACKDOOR_FLUSH_DRAW
        else:
            ans = TWO_ON_BOARD_ONE_IN_HAND_BACKDOOR_FLUSH_DRAW
        if suited_card.rank == highest_possible:
            ans += 0.1 # nuts flush draw with A
        else:
            ans -= (RANKS.index(highest_possible) - RANKS.index(suited_card.rank))*0.02
    else:
        ans = NO_FLUSH_DRAW
    
    flush_draw = round(ans, 2)
    
    return flush_draw

# format: Card: {rank: 'A', suit: 's'}
# working for only for turn
def calculate_flush_draw_on_turn(hand_cards, board_cards):
    counter = 0;

    if hand_cards[0].suit == hand_cards[1].suit:
        highest_card = hand_cards[0].rank;
        suit = hand_cards[0].suit
        for card in board_cards:
            if card.suit == suit:
                counter += 1
                # Compare indices to find the highest rank, but keep highest_card as string
                if RANKS.index(card.rank) > RANKS.index(highest_card):
                    highest_card = card.rank
        if (counter == 2):
            if (hand_cards[0].rank == 'A' or (highest_card == 'A' and hand_cards[0].rank == 'K')):
                return 0.8
            else:
                return 0.5
        else:
            return 0
        
    else:
        counter1 = 0
        counter2 = 0
        for card in board_cards:
            if card.suit == hand_cards[0].suit:
                counter1 += 1
            elif card.suit == hand_cards[1].suit:
                counter2 += 1
        if (counter1 == 3 or counter2 == 3):
            return 0.5
        else:
            return 0

#  when we are in turn/river, check if flush hit
# format: Card: {rank: 'A', suit: 's'}
def is_flush_hit(street, flop_cards, turn_card, river_card = None):
    board_before = flop_cards if street == TURN else flop_cards + [turn_card]
    max_before = max(Counter(c.suit for c in board_before).values())
    board_now = flop_cards + [turn_card] if street == TURN else flop_cards + [turn_card] + [river_card]
    max_now = max(Counter(c.suit for c in board_now).values())
    
    if (2<=max_before<4):
        return 1 if max_now == max_before+1 else 0
    else:
        return 0
    
def calc_draw_on_turn(hand_cards, board_cards):
    flush_draw = calculate_flush_draw_on_turn(hand_cards, board_cards)
    stright_draw = calculate_straight_draw(hand_cards, board_cards[:3], board_cards[3])/0.6

    return flush_draw+stright_draw

def calc_board_dynamic(board_cards):
    FLUSH_DANGER = 0.4
    STRAIGHT_DRAW_DANGER = 0.3
    BACK_DOOR_STRAIGHT_DANGER = 0.2
    ACE_TO_COME_DANGER = 0.2
    K_TO_COME_DANGER = 0.05
    Q_TO_COME_DANGER = 0.05

    ans = 0
    ranks = sorted([card.rank for card in board_cards])
    suits = [card.suit for card in board_cards]
    numeric_ranks = sorted([RANKS.index(r) for r in ranks])
    suit_counts = Counter(suits)

    if numeric_ranks[-1] - numeric_ranks[0] <= 4:
        ans += STRAIGHT_DRAW_DANGER
    elif numeric_ranks[1] - numeric_ranks[0] <= 2 or numeric_ranks[2] - numeric_ranks[1] <=2:
        ans += BACK_DOOR_STRAIGHT_DANGER

    flush_count = max(suit_counts.values())
    if flush_count >= 2:
        ans += FLUSH_DANGER * (flush_count - 1)

    high_cards = {'A': ACE_TO_COME_DANGER, 'K': K_TO_COME_DANGER, 'Q': Q_TO_COME_DANGER}
    for card, penalty in high_cards.items():
        if card not in ranks:
            ans += penalty
        else:
            break

    return min(round(ans, 2), 1)

# This function is used to calculate the strength of a hand before the flop - naive approach
# expect to get T as '10'
def calc_strength_pre (cards):
    RANKS = '23456789TJQKA'
    RANK_VALUES = {
        'A': 35, 'K': 28, 'Q': 20, 'J': 15, 'T': 13, '9': 12,
        '8': 10, '7': 9, '6': 8, '5': 7.5, '4': 7.25, '3': 7, '2': 6.8
    }
    PAIR_STRENGTH = {
        'A': 100, 'K': 98, 'Q': 92, 'J': 85, 'T': 80, '9': 75,
        '8': 70, '7': 67, '6': 64, '5': 60, '4': 56, '3': 54, '2': 52
    }

    r1, s1 = cards[0].rank, cards[0].suit
    r2, s2 = cards[1].rank, cards[1].suit

    if r1 == r2:
        return PAIR_STRENGTH[r1]

    rank = float(RANK_VALUES[r1] + RANK_VALUES[r2])    

    idx1, idx2 = RANKS.index(r1), RANKS.index(r2)
    diff = abs(idx1 - idx2)

    if diff == 1 or (r1, r2) in [('A', '2'), ('2', 'A')]:
        rank += 10
    elif diff == 2 or (r1, r2) in [('A', '3'), ('3', 'A')]:
        rank += 5
    elif diff == 3 or (r1, r2) in [('A', '4'), ('4', 'A')]:
        rank += 2
    
    if s1 == s2:
        rank *= 1.25

    return int(min(rank, 100))

