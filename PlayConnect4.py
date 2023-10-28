# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:48:08 2022

@author: mrgna
"""

import Connect4Game
import Connect4Players
import sys
import time
import numpy as np

class PlayConnect4:
    
    game = None
    human_color_wish: int = 1 #Red
    human_start_wish: int = 1 #Not Starting
    difficulty: str = ""
    _quick_start: bool = False
    _debug: bool = False
    
    def __init__(self, game):
        self.game = game
    
    def setup_game(self):

        self._does_human_want_to_play()
        
        #Get Human game specifications
        if not self._quick_start:
            self._get_human_start_wish()
            self._get_human_color_wish()
        self._get_human_difficulty_wish()
        
        #Create opponent player        
        self._create_player()
    
    def prepare_game(self):
        self._player_turn = 0
        if self.human_start_wish == self._player_turn:
            self._player_values = [self.human_color_wish, self.human_color_wish*-1]
        else:
            self._player_values = [self.human_color_wish*-1, self.human_color_wish]
            
        self.game.reset()
        
    def start_game(self):
        #init plot of game
        self.game.plot_board_state(update = True)
        
        while True:
            player_value = self._player_values[self._player_turn]
            self._next_player_turn = (self._player_turn + 1) % 2
            next_player_value = self._player_values[self._next_player_turn]
            
            #if human turn then ask human for action else ask AI player
            if self._player_turn == self.human_start_wish:
                action = self._get_human_action()
            else:
                clever_available_actions = self.game.get_clever_available_actions(player_value, next_player_value)
                action = self.player.make_action(self.game, clever_available_actions)
                self._log_player_action(action)
            
            #perform action and plot game
            self.game_over = self.game.place_disc(action, player_value)
            self.game.plot_board_state(update = True)
            
            #Check if game is over
            self._check_game_over()
            
            self._player_turn = self._next_player_turn
        
        print("We are draw... Good game.")
        input("Press Escape to stop the game: ")
        sys.exit()
        
    def _does_human_want_to_play(self) -> None:
        answer = input('Hello human! Do you want to play a game? [yes,no]: ')
        time.sleep(0.25)
        if type(answer) is not str:
            print("Answer not valid... idiot.")
            time.sleep(3)
            sys.exit()
        elif answer.lower() in ['no','n']:
            print('Pussy...')
            time.sleep(3)
            sys.exit()
        elif answer.lower() in ['quick','q']:
            print("Quick Start")
            self._quick_start = True
        elif answer.lower() not in ['yes','y']:
            print('Answer is neither yes or no... idiot.')
            time.sleep(3)
            sys.exit()
            
    #Asks human for color wish
    def _get_human_start_wish(self) -> None:
        time.sleep(0.25)
        human_start_wish = input('Do you wish to start? [yes,no]: ')
        
        while type(human_start_wish) is not str or human_start_wish.lower() not in ['yes','no','y','n']:
            time.sleep(0.25)
            human_start_wish = input("Answer not valid. Do you wish to start? [yes,no]: ")
        
        human_start_wish = human_start_wish.lower()

        time.sleep(0.25)

        if human_start_wish == 'yes' or human_start_wish == 'y':
            human_start_wish = 0
        else:
            human_start_wish = 1

        self.human_start_wish = human_start_wish
    
    #Asks human for color wish
    def _get_human_color_wish(self) -> None:
        human_color_wish = input('Do you wish to be red or yellow? [red,yellow]: ')
        while type(human_color_wish) is not str or human_color_wish.lower() not in ['red','yellow','r','y']:
            time.sleep(0.25)
            human_color_wish = input("Answer not valid. Do you wish to be red or yellow? [red,yellow]: ")
        
        human_color_wish = human_color_wish.lower()
        
        if human_color_wish in ['red','r']:
            human_color_wish = 1
        else:
            human_color_wish = -1

        self.human_color_wish = human_color_wish
    
    #Asks human for difficulty
    def _get_human_difficulty_wish(self) -> None:
        time.sleep(0.25)
        difficulty_question: str = 'Choose a difficulty [easy,normal,hard,god]: '
        difficulty: str = input(difficulty_question)
        
        while type(difficulty) is not str or difficulty.lower() not in ['easy','normal','hard','god','e','n','h','g']:
            time.sleep(0.25)
            difficulty = input("Answer not valid. Please choose a valid difficulty [easy,normal,hard,god]: ")    
        
        self.difficulty: str = difficulty.lower()
    
    #Create AI player based on difficulty
    def _create_player(self):
        time.sleep(0.25)
        if self.difficulty in ['easy','e']:
            self.player = Connect4Players.RandomPlayer()
            print("Ok... pussy. Let's play...")
        elif self.difficulty in ['normal','n']:
            game = Connect4Game.Connect4()
            self.player = Connect4Players.MCTSPlayer(game = game, 
                                                player = self.human_color_wish*-1, 
                                                next_player = self.human_color_wish, 
                                                max_count = 5e2, 
                                                max_depth = 2, 
                                                confidence_value = 2**2,
                                                rave_param=None)
            print("Ok. Let's see what you can do.")
        elif self.difficulty in ['hard','h']:
            game = Connect4Game.Connect4()
            self.player = Connect4Players.MCTSPlayer(game = game, 
                                                player = self.human_color_wish*-1, 
                                                next_player = self.human_color_wish, 
                                                max_count = 2e3, 
                                                max_depth = 5, 
                                                confidence_value = 2**2,
                                                rave_param=2**0) 
            print("You are brave. Let's go!")
        elif self.difficulty in ['god','g']:
            game = Connect4Game.Connect4()
            self.player = Connect4Players.MCTSPlayer(game = game, 
                                                player = self.human_color_wish*-1, 
                                                next_player = self.human_color_wish, 
                                                max_count = 1e4,
                                                max_depth = 100, 
                                                confidence_value = 2**2,
                                                rave_param=2**0)
            print("You are a dead man... Let's go!")
        else:
            raise
        
    #Asks human for action
    def _get_human_action(self) -> int:
        available_actions = self.game.get_available_actions()
        available_actions = np.array(available_actions)+1 #adjust for be 1-based
        
        human_action = input('Your turn human. Choose a column to play ' + str(available_actions) + ": ")
        
        try:
            human_action = int(human_action)
        except:
            human_action = None
        
        #check if chosen action is valid
        while human_action not in available_actions:
            human_action = input("Choice not valid. Please choose a valid column "+ str(available_actions) + ": ") 
            try:
                human_action = int(human_action)
            except:
                human_action = None
           
        return int(human_action)-1 #adjsut to be 0-based

    #Check if game is over and who has won
    def _check_game_over(self) -> None:
        if not self.game_over:
            return
        
        if self._player_turn != self.human_start_wish:
            print("YOU SUCK! I knew it...")
        else:
            print("God damn it... You are the master.")
        input("Press Escape to stop the game: ")
        sys.exit()
    
    def _log_player_action(self, action: int):
        print('I will play column',action+1)
        if hasattr(self.player, 'winning_probability'):
            print('I estimate that my probability of winning is: ',str(round(self.player.winning_probability,4)*100)+'%')
            if self._debug:
                optimal_actions, q_values,amaf_q_values,nodes = self.player.get_optimal_actions_qvalues()
                print('Optimal actions:',[i+1 for i in optimal_actions])
                print('Q values:',[round(i,4) for i in q_values])
                print('AMAF Q values:',[round(i,4) for i in amaf_q_values])
                for idx, node in enumerate(nodes):
                    print('Number of visits '+str(idx)+':',node['no_visits_actions'],'Q:',tuple(np.round(node['q_values'],3)),'AA:',node['actions']+1)
    

def main():
    game = Connect4Game.Connect4()
    playConnect4 = PlayConnect4(game)
    playConnect4.setup_game()
    playConnect4.prepare_game()
    playConnect4.start_game()
    
    
if __name__ == '__main__':
    main()