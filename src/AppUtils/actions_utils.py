from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..game import Game
    from ..player import Player

import AppUtils.StringUtils as strings
from typing import List

bet_sizing_categories = {
    'small': (0.0, 0.35),
    'medium': (0.35, 0.67),
    'large': (0.67, 1.0),
    'overbet': (1.0, float('inf')),
}

def get_bet_size_category(bet_size: float) -> str:
    for category, (min_size, max_size) in bet_sizing_categories.items():
        if min_size <= bet_size < max_size:
            return category
    return 'overbet'

def find_max_size_in_live_play(board: Game, player: Player) -> float:
    last_bet_size = board.last_bet_size
    stack_size = player.stack_size + player.chips_in_pot
    villains_stacks = [villain.stack_size + villain.chips_in_pot for villain in board.get_active_players_in_hand() if player.position != villain.position]
    return find_max_size_in_live_play_with_stack_sizes(last_bet_size, stack_size, villains_stacks)

def find_max_size_in_live_play_with_stack_sizes(last_bet_size: float, stack_size: float, villains_stacks: List[float]) -> float:
    if stack_size > last_bet_size:
        if len(villains_stacks) > 0:
            return min(stack_size, max(villains_stacks))
        else:
            return last_bet_size
    else:
        return stack_size

def find_possible_actions_in_live_play(board: Game, player: Player) -> List[str]:
    last_bet_size = board.last_bet_size
    stack_size = player.stack_size + player.chips_in_pot
    villains_stacks = [villain.stack_size + villain.chips_in_pot for villain in board.get_active_players_in_hand() if player.position != villain.position]
    possible_actions = find_possible_actions_with_stack_sizes(last_bet_size, stack_size, villains_stacks)
    return possible_actions

def find_possible_actions_with_stack_sizes(last_bet_size: float, stack_size: float, villains_stacks: List[float]) -> List[str]:
    possible_actions = []
    if last_bet_size > 0:
        possible_actions.append(strings.FOLD)
        possible_actions.append(strings.CALL)
    else:
        possible_actions.append(strings.CHECK)
        possible_actions.append(strings.RAISE)
        return possible_actions

    if is_raise_possible_in_live_play(last_bet_size, stack_size, villains_stacks):
        possible_actions.append(strings.RAISE)
    return possible_actions

def is_raise_possible_in_live_play(last_bet_size: float, stack_size: float, villains_stacks: List[float]) -> bool:
    if stack_size > last_bet_size:
        if any(stack > last_bet_size for stack in villains_stacks):
            return True
    return False
    