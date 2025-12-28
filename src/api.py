from flask import Flask, request, jsonify
from flask_cors import CORS
from AppUtils.cards_utils import pretty_suit
from game import Game

app = Flask(__name__)
CORS(app)

# Global game object
@app.route('/api/game/new', methods=['POST'])
def new_game():
    print("new_game")
    global current_game
    data = request.get_json()
    num_players = data.get('num_players', 2)
    stack_size = data.get('stack_size', 100)
    club_mode = data.get('club_mode', False)
    if not 2 <= num_players <= 9:
        return jsonify({'error': 'Number of players must be between 2 and 9'}), 400
    main_player_seat = num_players #random.randint(1, num_players)
    current_game = Game(num_players, main_player_seat, stack_size, club_mode=club_mode)

    players = [
        {
            'position': player.position,
            'position_index': player.position_index,
            'name': player.name,
            'is_main_player': player.is_main_player,
            'stack_size': player.stack_size,
            'chips_in_pot': player.chips_in_pot,
            'is_folded': player.is_folded,
            'is_need_to_act': player.is_need_to_act,
            'hand': [{'rank': card.rank, 'suit': pretty_suit(card.suit)} for card in player.hand.cards] if hasattr(player, 'hand') and player.hand else [],
        } for player in current_game.players
    ]
    
   # def start_game_background(current_game):
    #    if current_game:
    #        current_game.start_game()
    
   # game_thread = threading.Thread(target=start_game_background)
   # game_thread.daemon = True
   # game_thread.start()
    
    return jsonify({
        'message': 'Game created',
        'players': players
    })

@app.route('/api/game/state', methods=['GET'])
def get_state():
    global current_game
    if not current_game:
        return jsonify({'error': 'No active game'}), 404
    players = [
        {
            'name': player.name,
            'position': player.position,
            'is_player': player.is_main_player,
            'chips': player.stack_size,
            'hand': [{'rank': card.rank, 'suit': pretty_suit(card.suit)} for card in player.hand.cards] if hasattr(player, 'hand') and player.hand else [],
            'folded': False
        } for player in current_game.players
    ]
    return jsonify({
        'players': players,
        'board': {
            'pot_size': round(getattr(current_game, 'pot_size', 0), 2),
            'community_cards': getattr(current_game, 'community_cards', [])
        },
        'active_player_seat': getattr(current_game, 'active_player_seat', None)
    })

@app.route('/api/game/deal_next_cards', methods=['POST'])
def deal_next_cards():
    data = request.get_json()
    stage = data.get('stage')
    current_game.get_ready_for_next_street()
    players_to_act = [player.name for player in current_game.get_players_to_act_after_deal_cards()]
    print("players_to_act: ", players_to_act)
    
    return jsonify({
        'community_cards': [{'rank': card.rank, 'suit': pretty_suit(card.suit)} for card in current_game.community_cards] if current_game.community_cards else [],
        'players_to_act': players_to_act
    })

@app.route('/api/game/player_action', methods=['POST'])
def send_player_action():
    data = request.get_json()
    player_name = data.get('player')
    ans = current_game.actPlayer(player_name)
    action, total_bet_size, added_amount, stack_size = ans

    return jsonify({
        'message': 'Player action sent',
        'action': action,
        'total_bet_size': round(total_bet_size, 2) if total_bet_size is not None else 0,
        'added_amount': round(added_amount, 2) if added_amount is not None else 0,
        'stack_size': round(stack_size, 2) if stack_size is not None else 0,
        'players_to_act': [p.name for p in current_game.get_players_to_act_by_index(player_name)]
    })

@app.route('/api/game/send_main_player_action', methods=['POST'])
def get_main_player_action():
    data = request.get_json()
    if current_game.street == 0:
        check_action = current_game.check_action(data.get('action'), data.get('bet_size'))
    else:
        if False: # not current_game.club_mode
            categories_range = current_game.get_action_category_range(data.get('action'), data.get('bet_size'))
            check_action = categories_range
        else:
            check_action = None
    ans = current_game.act_after_human_choice(current_game.get_main_player(), data.get('action'), data.get('bet_size'))

    action, total_bet_size, added_amount, stack_size = ans
    
    return jsonify({
        'message': 'Player action sent',
        'action': action,
        'total_bet_size': round(total_bet_size, 2) if total_bet_size is not None else 0,
        'added_amount': round(added_amount, 2) if added_amount is not None else 0,
        'stack_size': round(stack_size, 2) if stack_size is not None else 0,
        'players_to_act': [player.name for player in current_game.get_players_to_act_by_index(current_game.get_main_player().name)], # todo - fix later
        'check_action': check_action
    })

@app.route('/api/game/show_down', methods=['POST'])
def show_down():
    data = request.get_json()
    winners, winning_hands = current_game.find_winners()
   
    # Convert Hand objects to dictionaries with card information
    winning_hands_dict = []
    for winning_hand in winning_hands:
        if winning_hand and hasattr(winning_hand, 'cards'):
            winning_hands_dict.append({
                'hand': [{'rank': card.rank, 'suit': pretty_suit(card.suit)} for card in winning_hand.cards]
            })
        else:
            winning_hands_dict.append(None)
        
    return jsonify({
        'message': 'Hand finished',
        'winners': [winner.name for winner in winners],
        'winning_hands': winning_hands_dict
    })

@app.route('/api/game/start_new_hand', methods=['POST'])
def start_new_hand():
    data = request.get_json()
    current_game.start_new_hand()
    return jsonify({
        'message': 'New hand started',
        'players': [
            {
                'name': player.name,
                'position': player.position,
                'is_main_player': player.is_main_player,
                'stack_size': player.stack_size,
                'chips_in_pot': player.chips_in_pot,
                'is_folded': player.is_folded,
                'is_need_to_act': player.is_need_to_act,
                'position_index': player.position_index,
                'hand': [{'rank': card.rank, 'suit': pretty_suit(card.suit)} for card in player.hand.cards] if hasattr(player, 'hand') and player.hand else [],
            } for player in current_game.players
        ]
        
    })
def run_api():
    import sys
    import os
    # Disable reloader when running in debugger (VS Code sets this)
    # Also check if we're in VS Code debug mode
    is_debugging = sys.gettrace() is not None or os.environ.get('VSCODE_DEBUG', '') == '1'
    use_reloader = not is_debugging
    app.run(debug=True, use_reloader=use_reloader, port=5000, use_debugger=is_debugging) 

     