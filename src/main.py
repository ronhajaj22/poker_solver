from agent.supervised_poker_agent import main as train_supervised_agent
from gg_hands_parser.gg_to_json import parse_sessions
from agent.parse_data_to_agent import save_training_data_to_file

from api import run_api


def main():
    # Uncomment for additional functionality
    parse_sessions()
    save_training_data_to_file("data/training_data.pkl")
    train_supervised_agent()  # This will automatically load from file if it exists
    run_api()

if __name__ == "__main__":
    main()