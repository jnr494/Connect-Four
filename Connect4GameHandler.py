import numpy as np
import copy
import numba

from Connect4Game import Connect4
from IPlayer import IPlayer

#class for handling gameplay
class Connect4GameHandler:

    game: Connect4
    rewards: list[int]
    save_info: bool

    actions: list[int]
    player_turns: list[int]

    def __init__(self, game: Connect4, player0: IPlayer, player1: IPlayer, save_info: bool = True, rewards: list[int] = [0.5,-1]):
        self.game = game
        self.game_size = self.game.no_rows * self.game.no_cols
        
        self.players = [player0,player1]
        self.no_players = len(self.players)
        self.player_values = [1,-1]

        self.save_info = save_info
        self.rewards = rewards
        
        self.reset_game()
        
    
    def reset_game(self) -> None:
        self.game.reset()
        #variables to save game states and actions
        self.actions = []
        self.states = [self.game.Board]
        self.player_turns = []
        self.action_probabilities = []
        self.winner = 0
        for player in self.players:
            player.reset()
        
    def adjust_board_state(self, board_state, player_turn,flip):
        return adjust_board_state_numba(board_state, self.player_values[player_turn], self.game.no_cols, flip)

    
    def adjust_action(self, action, flipped_bool):
        if flipped_bool:
            action = (action-(self.game.no_cols-1))*(-1)
        return action
    
    def play_game(self,player_turn=0, plot_primo_states = False, flip = False):
        for _ in range(self.game_size):
            #plot board state if enabled
            if plot_primo_states:
                self.game.plot_board_state()
            
            #get player value and next player turn
            current_player_value = self.player_values[player_turn]
            next_player_turn = (player_turn + 1) % self.no_players
            next_player_value = self.player_values[next_player_turn]
            
            #update player turns
            self.player_turns.append(current_player_value)
            
            #get board state
            #board_state, flipped_bool = self.adjust_board_state(self.game.Board, player_turn, flip)
            
            #get available (clever) actions
            #clever_available_actions = self.adjust_action(self.game.get_clever_available_actions(current_player_value,next_player_value),flipped_bool)
            clever_available_actions = self.game.get_clever_available_actions(current_player_value,next_player_value)
            #get new action
            new_action = self.players[player_turn].make_action(self.game, clever_available_actions)
            
            #save action probabilities (only if new_action is tuple)
            if type(new_action) is tuple:
                self.action_probabilities.append(new_action[1])
                new_action = new_action[0]
            else:
                self.action_probabilities.append(None)
            
            #adjust and save action
            #new_action = self.adjust_action(new_action, flipped_bool)
            self.actions.append(new_action)

            #perform new action and save state
            is_game_won = self.game.place_disc(new_action,current_player_value)
            self.states.append(copy.deepcopy(self.game.Board))
            
            #check if game is done
            #is_game_won = self.game.check_four_in_a_row(new_action,new_row_height,current_player_value)
            if is_game_won:
                self.winner = self.game.winner
                break

            #update player turn
            player_turn = next_player_turn
     
    def play_n_games(self,no_games,plot = False, flip = False):
        player_turn = 0
        winners = []
        
        
        for _ in range(no_games):
            print('Game ',_)
            self.reset_game()
            self.play_game(player_turn, flip = flip)
            winners.append(self.winner)
            player_turn = (player_turn + 1) % 2
            if plot:
                self.game.plot_board_state()
        return winners
       
        
    def save_data(self):
        pass
        
@numba.njit
def adjust_board_state_numba(board_state, player_value,no_cols, flip):
    board_state = board_state * player_value # adjust board so player is always 1 and opponent -1
    
    if flip:
        #flip board so most discs are on the left side
        half_no_cols = int(no_cols/2)
        no_discs_left_side = np.count_nonzero(board_state[:,:half_no_cols])
        no_discs_right_side = np.count_nonzero(board_state[:,-half_no_cols:])
        if no_discs_right_side > no_discs_left_side:
            board_state = np.fliplr(board_state)
            flipped_bool = True
        else:
            flipped_bool = False
    else:
        flipped_bool = False
        
    return board_state, flipped_bool