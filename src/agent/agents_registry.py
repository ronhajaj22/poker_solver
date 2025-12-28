"""
Agents Registry - Singleton module for poker agents and MCTS.

Usage:
    from agents_registry import get_hero_agents, get_villain_agents, get_mcts

FUNCTIONS (signatures):
- initialize_agents() - Initialize all agents once
- load_agent(model_path, input_dim) - Load a single agent model
- get_hero_agents() - Get hero agents
- get_villain_agents() - Get villain agents
- get_mcts() - Get MCTS model
"""

import torch
from AppUtils.files_utils import find_agent_path
from AppUtils.constants import FLOP, TURN, RIVER


# Singleton instances
hero_agents = None
club_agents = None
mcts = None
_initialized = False


def initialize_agents():
    global hero_agents, club_agents, mcts, _initialized
    
    if _initialized:
        return
        
    from agent.agent_utils import FLOP_FEATURES_COUNT, TURN_FEATURES_COUNT, RIVER_FEATURES_COUNT, BLIND_FLOP_FEATURES_COUNT, BLIND_TURN_FEATURES_COUNT, BLIND_RIVER_FEATURES_COUNT
    hero_config = {
        FLOP: (find_agent_path(FLOP), FLOP_FEATURES_COUNT),
        TURN: (find_agent_path(TURN), TURN_FEATURES_COUNT),
        RIVER: (find_agent_path(RIVER), RIVER_FEATURES_COUNT)
    }
    
    villain_config = {
        FLOP: (find_agent_path(FLOP, is_blind_agent=True), BLIND_FLOP_FEATURES_COUNT),
        TURN: (find_agent_path(TURN, is_blind_agent=True), BLIND_TURN_FEATURES_COUNT),
        RIVER: (find_agent_path(RIVER, is_blind_agent=True), BLIND_RIVER_FEATURES_COUNT)
    }
    
    hero_agents = {}
    club_agents = {}
    
    # Load hero agents
    for street, (model_path, input_dim) in hero_config.items():
        try:
            hero_agents[street] = load_agent(model_path, input_dim)
            print(f"  ✓ Loaded hero agent for street {street}")
        except Exception as e:
            print(f"  ✗ Failed to load hero agent for street {street}: {e}")
    
    # Load villain agents
    for street, (model_path, input_dim) in villain_config.items():
        try:
            club_agents[street] = load_agent(model_path, input_dim)
            print(f"  ✓ Loaded club agent for street {street}")
        except Exception as e:
            print(f"  ✗ Failed to load club agent for street {street}: {e}")
    
    _initialized = True
    
    from MCTS.mcts import MCTS
    mcts = MCTS()
    
    print(f"🎯 Agents registry initialized: {len(hero_agents)} hero, {len(club_agents)} villain")

def load_agent(model_path, input_dim):
    from agent.supervised_poker_agent import SupervisedPokerAgent
    agent = SupervisedPokerAgent(input_dim=input_dim, num_actions=4)
    checkpoint = torch.load(model_path, map_location='cpu')
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    return agent

def get_hero_agents():
    if not _initialized:
        initialize_agents()
    return hero_agents

def get_villain_agents():
    if not _initialized:
        initialize_agents()
    return club_agents

def get_mcts():
    if not _initialized:
        initialize_agents()

    return mcts
