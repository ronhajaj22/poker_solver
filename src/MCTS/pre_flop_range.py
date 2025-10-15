pocket_pairs = ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22']
a_high_suited = ['AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A5s', 'A4s', 'A3s', 'A7s', 'A2s', 'A6s']
off_premiun = ['AKo', 'AQo', 'KQo', 'AJo', 'ATo', 'KJo']
high_suited = ['KQs', 'KJs', 'KTs', 'K9s', 'QTs', 'QTs', 'J9s']
suited_connected = ['QJs', 'JTs', 'T9s', '98s', '87s', '65s', '54s']

four_bet_range = ['AA', 'KK', 'AKs', 'AKo', 'QQ'] # 38 combos
strong_three_bet_range = ['AQs', 'JJ', 'TT', 'AQo'] # 32 combos
three_bet_range = ['AJs', 'ATs', 'KQs', 'KJs', '99', '88', 'QJs', 'JTs', 'T9s', '98s', '87s']
utg_open_range = four_bet_range + strong_three_bet_range + three_bet_range + a_high_suited + off_premiun
def get_range(num_raises, position, pfr):
    return [];






