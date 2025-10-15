import AppUtils.StringUtils as strings

class Player:
    def __init__(self, name = '', stack = 100, position = None, cards = [], is_folded = False, chips_in_pot = 0, action = 'CHECK'):
        self.name = name
        self.position = position
        self.cards = cards
        self.is_folded = is_folded
        self.is_hero = name == 'Hero'
        
        self.stack = [[stack], [], [], []]
        self.chips_in_pot = [[chips_in_pot], [], [], []]

        self.hand_strength = 0
        self.flush_draw = 0
        self.straight_draw = 0
        
        self.aggression = [0, 0, 0, 0]

        self.actions = [[action], [], [], []] # pre-flop -> not real! just check/fold and for hero we have raise if her's the last raise
        self.action_sizes = [[], [], [], []] # this is array in size 4 (pre_flop, flop, turn, river)
    
    def set_cards(self, cards):
        self.cards = cards
    
    def set_chips_in_pot(self, chips_in_pot):
        self.chips_in_pot += chips_in_pot

    def set_pre_flop_actions(self, action, size = None, bb_size = 2):
        self.add_action(0, action, size, bb_size)

    def set_flop_actions(self, action, size = None, bb_size = 2):
        self.add_action(1, action, size, bb_size)

    def set_turn_actions(self, action, size = None, bb_size = 2):
        self.add_action(2, action, size, bb_size)
    
    def set_river_actions(self, action, size = None, bb_size = 2):
        self.add_action(3, action, size, bb_size)
    
    def set_is_last_raiser(self):
        self.actions[0][-1] = strings.RAISE

    def add_action(self, street, action, size = None, bb_size = 2):
        self.actions[street].append(action)
        if size:
            size = round(float(size)/bb_size, 2)
        self.action_sizes[street].append(size)
    
    def calc_aggression(self, data, street, stage):
        sum_action = 0
        for index, action in enumerate(self.actions[street]):
            if (stage < 0 or index < stage):
                if (action == strings.RAISE):
                    sum_action =  3 if index == 0 and data.pot_size[street-1][0] == data.pot_size[street][0] else max(sum_action + 3, 5)
                elif (action == strings.CALL):
                    sum_action += 1
                elif (action == strings.CHECK):
                    sum_action -= 0.5

        self.aggression[street] = sum_action
        return sum_action

