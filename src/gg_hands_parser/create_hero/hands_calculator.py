from AppUtils.constants import ALL_POSITIONS_POST_FLOP, PREFLOP, FLOP, TURN, RIVER
from AppUtils.cards_utils import CARD_RANK_TO_INT_VALUE
from AppUtils import agent_utils

HEADS_UP_POSITIONS = {'BB': 0, 'SB': 1}

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    
    def __str__(self):
        return f"{self.rank}{self.suit}"

class HandData:
    def __init__(self, hand_id, bb_size, hero_cards, flop_cards, players_num, turn_card = None, river_card = None):
        self.hand_id = hand_id
        self.bb_size = bb_size
        self.players_num = players_num

        self.hero_cards =  sorted([Card(card[0], card[1]) for card in hero_cards], key=lambda x: x.rank, reverse=True)
        self.flop_cards = sorted([Card(card[0], card[1]) for card in flop_cards], key=lambda x: x.rank, reverse=True)

        self.board_cards = self.flop_cards.copy()

        self.is_hand_suited = self.hero_cards[0].suit == self.hero_cards[1].suit

        self.pot_size = [[0], [], [], []]

        self.hero = None
        self.sum_action = [[0], [], [], []]

        self.turn_card = None
        self.river_card = None

    def set_turn_card(self, turn_card):
        self.turn_card = Card(turn_card[0], turn_card[1])
        self.board_cards.append(self.turn_card)

    def set_river_card(self, river_card):
        if river_card:
            self.river_card = Card(river_card[0], river_card[1])
            self.board_cards.append(self.river_card)
    
    def calc_flop_action(self, players, pot_size, num_actions):
        self.pot_size[PREFLOP][0] = pot_size
        
        self.calc_action(players, num_actions, FLOP)

    def calc_turn_action(self, players, num_actions):
        self.calc_action(players, num_actions, TURN)
    
    def calc_river_action(self, players, num_actions):
        self.calc_action(players, num_actions, RIVER)
    
    # in this function we calculate the general actions strength before each hero decision
    # we also update the pot size and the stack of each player before the decision
    # actions that wete made after hero's last decision will be ignored - the pot size will not be updated and the stacks will not be updated
    def calc_action(self, players, hero_num_actions, street): 
        sorted_players = sorted([player for player in players if not player.is_folded], key=lambda x: ALL_POSITIONS_POST_FLOP.index(x.position) if self.players_num > 2 else HEADS_UP_POSITIONS[x.position])
        
        action_num = 0
        max_actions = max(len(player.actions[street]) for player in players)
       # self.sum_action[street] = [0] * max_actions

        if (street > PREFLOP and self.pot_size[street] == []):
            self.pot_size[street].append(self.pot_size[street-1][-1])
            self.sum_action[street].append(0)
            for player in players:
                player.stack[street].append(player.stack[street-1][-1])
        i = 0;
        while (action_num < max_actions):
            for player in sorted_players:
                if player.is_folded or action_num >= len(player.actions[street]):
                    continue

                if (player.name == 'Hero'):
                    i+=1
                    if i == hero_num_actions and (street == RIVER or player.actions[street][i-1] == 'FOLD'):
                        break;
                    else:
                        self.sum_action[street].append(self.sum_action[street][i-1])
                        self.pot_size[street].append(self.pot_size[street][i-1])

                if (i > 0 and len(player.stack[street]) == i):
                    player.stack[street].append(player.stack[street][i-1])

                if player.actions[street][action_num] == 'RAISE':
                    self.sum_action[street][i] = max(player.action_sizes[street][action_num]/self.pot_size[street][i], self.sum_action[street][i] * 3)
                    self.pot_size[street][i] += player.action_sizes[street][action_num]
                    player.stack[street][i] -= player.action_sizes[street][action_num]
                elif player.actions[street][action_num] == 'FOLD':
                    player.is_folded = True
                elif player.actions[street][action_num] == 'CALL':
                    self.sum_action[street][i] += 0.1
                    self.pot_size[street][i] += player.action_sizes[street][action_num]
                    player.stack[street][i] -= player.action_sizes[street][action_num]
                agent_utils.update_action_strength(self.sum_action[street][i], player.actions[street][action_num], player.action_sizes[street][action_num], self.players_num, self.pot_size[street][i], street)           
            action_num += 1

    def calc_spr(self, villians, street, stage):
        relevant_stack = self.hero.stack[street][stage]
        rel_villians = [v for v in villians if v.name != 'Hero' and 'FOLD' not in v.actions[street][:stage]]

        villian_stacks = [v.stack[street][0] for v in rel_villians]
        if (len(villian_stacks) == 1):
            villian_stack = villian_stacks[0]
            relevant_stack = min(relevant_stack, villian_stack)
        return round(relevant_stack/self.pot_size[street][stage], 2)

    