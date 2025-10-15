RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']

CARD_RANK_TO_INT_VALUE = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

ALL_POCKET_PAIRS = ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22']
SUITED_CARDS = ['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'KQs', 'KJs', 'KTs', 'K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s', 'QJs', 'QTs', 'Q9s', 'Q8s', 'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s', 'JTs', 'J9s', 'J8s', 'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s', 'T9s', 'T8s', 'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s', '98s', '97s', '96s', '95s', '94s', '93s', '92s', '87s', '86s', '85s', '84s', '83s', '82s', '76s', '75s', '74s', '73s', '72s', '65s', '64s', '63s', '62s', '54s', '53s', '52s', '43s', '42s', '32s']
UN_SUITED_CARDS = ['AKo', 'AQo', 'AJo', 'ATo', 'A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o', 'KQo', 'KJo', 'KTo', 'K9o', 'K8o', 'K7o', 'K6o', 'K5o', 'K4o', 'K3o', 'K2o', 'QJo', 'QTo', 'Q9o', 'Q8o', 'Q7o', 'Q6o', 'Q5o', 'Q4o', 'Q3o', 'Q2o', 'JTo', 'J9o', 'J8o', 'J7o', 'J6o', 'J5o', 'J4o', 'J3o', 'J2o', 'T9o', 'T8o', 'T7o', 'T6o', 'T5o', 'T4o', 'T3o', 'T2o', '98o', '97o', '96o', '95o', '94o', '93o', '92o', '87o', '86o', '85o', '84o', '83o', '82o', '76o', '75o', '74o', '73o', '72o', '65o', '64o', '63o', '62o', '54o', '53o', '52o', '43o', '42o', '32o']
ALL_CARD_COMBINATIONS = ALL_POCKET_PAIRS + SUITED_CARDS + UN_SUITED_CARDS


pocket_pairs = ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22']
a_high_suited = ['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A5s', 'A4s', 'A3s', 'A7s', 'A2s', 'A6s']
off_premiun = ['AKo', 'AQo', 'KQo', 'AJo', 'ATo', 'KJo']
high_suited = ['KQs', 'KJs', 'KTs', 'K9s', 'QTs', 'Q9s', 'J9s']
suited_connected = ['QJs', 'JTs', 'T9s', '98s', '87s', '65s', '54s']
weak_a_high = ['A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o']
weak_k_high = ['KTo', 'K9o', 'K8s', 'K7s']
suited_connected_with_gap = ['T8s', '97s', '86s', '75s', '64s', '43s', '32s']
ok_cards = ['QJo', 'QTo', 'T9o', 'K9o', 'K6s', 'K5s', 'K4s']
utg_open_range = pocket_pairs + a_high_suited + off_premiun + suited_connected + high_suited
co_open_range = utg_open_range + weak_a_high + weak_k_high + suited_connected_with_gap
btn_open_range = co_open_range + ok_cards

def card_str_to_int_rank(card):
    return CARD_RANK_TO_INT_VALUE[card[0]]

def card_to_str(card):
    return card.rank + card.suit

# get list of cards as object return list of strings (result will look like ['As', 'Jd])
def cards_str(cards):
    return [card.rank + card.suit for card in cards]

# get a player's hand when each card is a treys object return it as str (like 'AKo')
def trey_to_str(hand):
    from treys import Card

    def get_card_rank(card):
        rank_int = Card.get_rank_int(card)
        return RANKS[rank_int]
    
    def get_card_suit(card):
        suit_int = Card.get_suit_int(card)
        suit_map = {1: 's', 2: 'h', 4: 'd', 8: 'c'}
        return suit_map[suit_int]

    # Sort cards by rank strength (highest first)
    sorted_hand = sorted(hand, key=lambda card: RANKS.index(get_card_rank(card)), reverse=True)
    
    rank1 = get_card_rank(sorted_hand[0])
    rank2 = get_card_rank(sorted_hand[1])
    suit1 = get_card_suit(sorted_hand[0])
    suit2 = get_card_suit(sorted_hand[1])
    
    if (rank1 == rank2):
        return rank1 + rank2
    else:
        return rank1 + rank2 + ('o' if suit1 != suit2 else 's')

# get a player's hand when each card is a Card object and return it as str (like 'AKo')
def hand_to_str(hand):
    # Sort cards by rank strength (highest first)
    sorted_hand = sorted(hand, key=lambda card: RANKS.index(card.rank), reverse=True)
    
    if (sorted_hand[0].rank == sorted_hand[1].rank):
        return sorted_hand[0].rank + sorted_hand[1].rank
    else:
        return sorted_hand[0].rank + sorted_hand[1].rank + ('o' if sorted_hand[0].suit != sorted_hand[1].suit else 's')


