from collections import Counter

class JsonFormatter:
    def encode_card(self, card):
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
                    '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9,
                    'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

        rank_vec = [0] * 13
        suit_vec = [0] * 4

        rank = rank_map[card[0]]
        suit = suit_map[card[1]]

        rank_vec[rank] = 1
        suit_vec[suit] = 1

        return rank_vec + suit_vec

    def format_cards_to_vector(self, cards):
        return [self.encode_card(card) for card in cards]
    
    def format_cards_to_rank(self, cards):
        ans = []
        for card in cards:
            rank = self.card_to_rank(card)
            #ans.append(round(rank-1/13, 2))
            ans.append(rank)
        return ans
    
    # TODO - replace to more generic function
    def card_to_rank(self, card):
        """Convert card to rank value (2-14, where 14 is Ace)"""
        rank = card[0]
        if rank == 'T':
            return 10
        elif rank == 'J':
            return 11
        elif rank == 'Q':
            return 12
        elif rank == 'K':
            return 13
        elif rank == 'A':
            return 14
        return int(rank)
    
    def encode_list_cards_as_matrix(self, cards, street):
        """
        Encode cards as ?×17 matrix
        Each card: 13 ranks + 4 suits
        """
        # Initialize 5×17 matrix with zeros
        board_matrix = [[0] * 17 for _ in range(street+2)]
        
        if not cards:
            return board_matrix
        
        # Split board cards into individual cards
        cards = [cards[i:i+2] for i in range(0, len(cards), 2)]
        
        for i, card in enumerate(cards):
            if i < 5 and len(card) == 2:  # Max 5 cards
                rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
                           '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9,
                           'Q': 10, 'K': 11, 'A': 12}
                suit_map = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
                
                rank = rank_map.get(card[0], 0)
                suit = suit_map.get(card[1], 0)
                
                # Set rank (position 0-12)
                board_matrix[i][rank] = 1
                # Set suit (position 13-16)
                board_matrix[i][13 + suit] = 1
        
        return board_matrix
