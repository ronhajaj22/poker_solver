from api import run_api
from pre_flop_charts.create_pre_flop_chart import create_pre_flop_charts
from gg_hands_parser.create_hero.gg_to_json import parse_sessions, players_map
from agent.supervised_poker_agent import main as train_supervised_agent
from agent.poker_agent_new import start_model

def main():
    # Uncomment for additional functionality
    #
    # parse_sessions(collect_data=True)
    
    #create_pre_flop_charts()

    # Train suppervised poker agent for all streets (flop, turn, river)
    # for street in range(1, 4):
    #     print(f"Training for street {street}")
    #     start_model(street)
  #  generated_players_map = parse_sessions(collect_data=True)
    run_api()

if __name__ == "__main__":
    main()