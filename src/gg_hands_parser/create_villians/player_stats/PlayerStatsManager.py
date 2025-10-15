from .PlayerStats import PlayerStats
from AppUtils.files_utils import find_players_stats_path
import pdb
reg_player_names = ['Hero', 'Nagasti_Bagamba', 'Mazal_Ohayon', 'Titia', 'THE_JUDGE', 'RiverRoullette', 'Development',
                    'kadimmaa', 'HadagNahash', 'Luckbox222', 'Tandorban1']

class PlayerStatsManager:
    """
    Manages multiple PlayerStats objects for different players.
    """
    cash_player_types = ['HU', 'Normal', 'Bombpot']
    cash_bb_sizes = [2, 4, 6, 8, 10]

    def __init__(self):
        self.temp_players = {} 
        self.const_players = {}
    
    # get the player from players map and if it isn't exist, create it
    def get_player(self, player_name: str, bb_size: float, is_heads_up: bool) -> PlayerStats:
        
        name_with_details = self.create_general_name(player_name, bb_size, is_heads_up)
        
        # if it is a reg player/temp player that already exists -> return the player
        if name_with_details in self.const_players:
            return self.const_players[name_with_details]
        elif name_with_details in self.temp_players:
            return self.temp_players[name_with_details]
        
        # if the player isn't exist yet        
        else:
            # if it is a reg player -> create a const player (for example, first time playing HU)
            if player_name in reg_player_names:
                self.const_players[name_with_details] = PlayerStats(player_name, name_with_details, False, None, is_heads_up)
                print("created const reg player", name_with_details)
                return self.const_players[name_with_details]
            
            # if it is a temp player -> create a temp player
            else:
                general_name = self.create_general_name('player', bb_size, is_heads_up)
                # the basic player exist - we just create a temp player that points to it
                if general_name in self.const_players:
                    self.temp_players[name_with_details] = PlayerStats(player_name, name_with_details, True, self.const_players[general_name], is_heads_up)
                else:
                    # if the basic player doesn't exist -> create a const player
                    general_player = PlayerStats(general_name, general_name, False, None, is_heads_up)
                    print("created const player", general_name)
                    self.const_players[general_name] = general_player
                    self.temp_players[name_with_details] = PlayerStats(player_name, name_with_details, True, general_player, is_heads_up)
                return self.temp_players[name_with_details]
    
    # add '-_hu_' if it is heads up and bb_size
    def create_general_name(self, player_name: str, bb_size: float, is_heads_up: bool) -> str:
        name =  player_name + ('_hu_' if is_heads_up else '_')  + str(int(bb_size))
        
        return name

    def get_all_summaries(self) -> dict:
        print("getting all summaries")
        print(self.const_players.keys())
        for player in self.const_players.values():
            self.calc_avgs(player)
            self.print_player(player)
        
        print(f"All player statistics saved to folder!")

    def calc_won_avgs(self, player):
        total_win_percent = 0
        won_with_no_showdown_percent = 0
        won_with_showdown_percent = 0

        if player.pre_flop_stats.total_hands != 0:
            total_win_percent = round(float(player.hands_won*100 / player.pre_flop_stats.total_hands), 2)
            if player.hands_won != 0:
                won_with_no_showdown_percent = round(float(player.wnsd*100 / player.hands_won), 2)
                won_with_showdown_percent = round(float(player.wsd*100 / player.hands_won), 2)

        return total_win_percent, won_with_showdown_percent, won_with_no_showdown_percent

    def print_player(self, player):
        import os
        bb_size = int(player.player_general_name[-1])
        # Create villains_stats folder if it doesn't exist
        folder_name = find_players_stats_path(bb_size)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        filename = f"{folder_name}/{player.player_general_name}_stats.txt"
        print('PLAYER NAME:', player.player_general_name)
        with open(filename, 'w') as f:
            f.write(f"PLAYER: {player.player_general_name}\n")
            f.write("="*60 + "\n")
            

            # General Stats
            f.write("\nGENERAL STATS:\n")
            f.write(f"  Total Hands: {player.pre_flop_stats.total_hands}\n")
            total_win_percent, won_with_showdown_percent, won_with_no_showdown_percent = self.calc_won_avgs(player)
            f.write(f"  Hands Won ({player.hands_won}): {total_win_percent}%\n")
            f.write(f"  Won With Showdown: {won_with_showdown_percent}%")
            f.write(f", Won With No Showdown: {won_with_no_showdown_percent}%\n")
            f.write(f"  VPIP: {player.pre_flop_stats.vpip}%\n")


            zero_values_filter = lambda x: {k: v for k, v in x.items() if v > 0}

            # Print stats for each street
            f.write("\n" + "="*60 + "\n")
            player.pre_flop_stats.print_stats(f, zero_values_filter)
            player.flop_stats.print_all_stats(f)
            player.turn_stats.print_stats(f)
            player.river_stats.print_all_stats(f)
            
            print(f"Created {filename}")

    def calc_avgs(self, player):
        #calc_avg = lambda x:  sum(x) / len(x) if len(x) > 0 else 0
        
        player.pre_flop_stats.calc_avgs()