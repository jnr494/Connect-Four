# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:48:08 2022

@author: magnus
"""

import sys
import time
import numpy as np

import Connect4Game
import Connect4Players
import GameTurnHandler
from Logger import Logger

class PlayConnect4:
    
    _game: Connect4Game.Connect4
    _game_turn_handler: GameTurnHandler.GameTurnHandler
    human_color_wish: int = 1 #Red
    human_start_wish: int = 1 #Not Starting
    difficulty: str
    _quick_start: bool = False
    _debug: bool = False
    
    def __init__(self, game: Connect4Game.Connect4, game_turn_handler: GameTurnHandler.GameTurnHandler, logger):
        self._game = game
        self._game_turn_handler = game_turn_handler
        self._logger = logger
        
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
        #Prepare GameTurnHandler
        player_values: list[int]    
        if self.human_start_wish == 0:
            player_values = [self.human_color_wish, self.human_color_wish*-1]
        else:
            player_values = [self.human_color_wish*-1, self.human_color_wish]
        
        self._game_turn_handler.setup(player_values)
            
        #Reset Game
        self._game.reset()
        
    def start_game(self):
        action: int
        
        self._logger.info("Start game.")
        
        #init plot of game
        self._game.plot_board_state(update = True)
        
        for _ in range(self._game.no_cols*self._game.no_rows):
            
            #if human turn then ask human for action else ask AI player
            if self._game_turn_handler.get_current_player_value() == self.human_color_wish:
                action = self._get_human_action()
            else:
                action = self._get_nonhuman_player_action()
                self._log_player_action(action)
                
            #perform action and plot game
            self.game_over = self._game.place_disc_using_turn_handler(action)
            self._game.plot_board_state(update = True)
            
            #Check if game is over
            self._check_game_over()
            
            #Go to next turn
            self._game_turn_handler.next_turn()
        
        self._message_to_human_player("We are draw... Good game.")
        self._question_to_human_player("Press Escape to stop the game: ")
        sys.exit()
        
    def _does_human_want_to_play(self) -> None:
        answer = self._question_to_human_player('Hello human! Do you want to play a game? [yes,no]: ')
        
        time.sleep(0.25)
        if type(answer) is not str:
            self._message_to_human_player("Answer not valid... idiot.")
            time.sleep(3)
            sys.exit()
        elif answer.lower() in ['no','n']:
            self._message_to_human_player('Pussy...')
            time.sleep(3)
            sys.exit()
        elif answer.lower() in ['quick','q']:
            self._message_to_human_player("Quick Start")
            self._quick_start = True
        elif answer.lower() not in ['yes','y']:
            self._message_to_human_player("Answer is neither yes or no... idiot.")
            time.sleep(3)
            sys.exit()
            
    #Asks human for color wish
    def _get_human_start_wish(self) -> None:
        time.sleep(0.25)
        human_start_wish = self._question_to_human_player('Do you wish to start? [yes,no]: ')
        
        while type(human_start_wish) is not str or human_start_wish.lower() not in ['yes','no','y','n']:
            time.sleep(0.25)
            human_start_wish = self._question_to_human_player("Answer not valid. Do you wish to start? [yes,no]: ")
        
        human_start_wish = human_start_wish.lower()

        time.sleep(0.25)

        if human_start_wish == 'yes' or human_start_wish == 'y':
            human_start_wish = 0
        else:
            human_start_wish = 1

        self.human_start_wish = human_start_wish
    
    #Asks human for color wish
    def _get_human_color_wish(self) -> None:
        human_color_wish = self._question_to_human_player('Do you wish to be red or yellow? [red,yellow]: ')
        while type(human_color_wish) is not str or human_color_wish.lower() not in ['red','yellow','r','y']:
            time.sleep(0.25)
            human_color_wish = self._question_to_human_player("Answer not valid. Do you wish to be red or yellow? [red,yellow]: ")
        
        human_color_wish = human_color_wish.lower()
        
        if human_color_wish in ['red','r']:
            human_color_wish = 1
        else:
            human_color_wish = -1

        self.human_color_wish = human_color_wish
    
    #Asks human for difficulty
    def _get_human_difficulty_wish(self) -> None:
        time.sleep(0.25)
        difficulty: str = self._question_to_human_player('Choose a difficulty [easy,normal,hard,god]: ')
        
        while type(difficulty) is not str or difficulty.lower() not in ['easy','normal','hard','god','e','n','h','g']:
            time.sleep(0.25)
            difficulty = self._question_to_human_player("Answer not valid. Please choose a valid difficulty [easy,normal,hard,god]: ")    
        
        self.difficulty: str = difficulty.lower()
    
    #Create AI player based on difficulty
    def _create_player(self):
        time.sleep(0.25)
        if self.difficulty in ['easy','e']:
            self.player = Connect4Players.RandomPlayer()
            self._message_to_human_player("Ok... pussy. Let's play...")
        elif self.difficulty in ['normal','n']:
            game = Connect4Game.Connect4()
            self.player = Connect4Players.MCTSPlayer(game = game, 
                                                player = self.human_color_wish*-1, 
                                                next_player = self.human_color_wish, 
                                                max_count = 5e2, 
                                                max_depth = 2, 
                                                confidence_value = 2**2,
                                                rave_param=None)
            self._message_to_human_player("Ok. Let's see what you can do.")
        elif self.difficulty in ['hard','h']:
            game = Connect4Game.Connect4()
            self.player = Connect4Players.MCTSPlayer(game = game, 
                                                player = self.human_color_wish*-1, 
                                                next_player = self.human_color_wish, 
                                                max_count = 2e3, 
                                                max_depth = 5, 
                                                confidence_value = 2**2,
                                                rave_param=2**0) 
            self._message_to_human_player("You are brave. Let's go!")
        elif self.difficulty in ['god','g']:
            game = Connect4Game.Connect4()
            self.player = Connect4Players.MCTSPlayer(game = game, 
                                                player = self.human_color_wish*-1, 
                                                next_player = self.human_color_wish, 
                                                max_count = 1e4,
                                                max_depth = 100, 
                                                confidence_value = 2**2,
                                                rave_param=2**0)
            self._message_to_human_player("You are a dead man... Let's go!")
        else:
            raise
        
    #Asks human for action
    def _get_human_action(self) -> int:
        available_actions = self._game.get_available_actions()
        available_actions = np.array(available_actions)+1 #adjust for be 1-based
        
        human_action = self._question_to_human_player('Your turn human. Choose a column to play ' + str(available_actions) + ": ")
        
        try:
            human_action = int(human_action)
        except:
            human_action = None
        
        #check if chosen action is valid
        while human_action not in available_actions:
            human_action = self._question_to_human_player("Choice not valid. Please choose a valid column "+ str(available_actions) + ": ") 
            try:
                human_action = int(human_action)
            except:
                human_action = None
           
        return int(human_action)-1 #adjsut to be 0-based
    
    #Get action from non-human player
    def _get_nonhuman_player_action(self) -> int:
        clever_available_actions = self._game.get_clever_available_actions_using_turn_handler()
        action = self.player.make_action(self._game, clever_available_actions)
        return action
    
    #Check if game is over and who has won
    def _check_game_over(self) -> None:
        if not self.game_over:
            return
        
        if self._game.get_winner() == self.human_color_wish:
            self._message_to_human_player("God damn it... You are the master.")
        else:
            self._message_to_human_player("YOU SUCK! I knew it...")
        self._question_to_human_player("Press Escape to stop the game: ")
        sys.exit()
    
    def _log_player_action(self, action: int):
        self._message_to_human_player(f"I will play column {action+1}")
        if hasattr(self.player, 'winning_probability'):
            self._message_to_human_player("I estimate that my probability of winning is: " + str(round(self.player.winning_probability,4)*100) + "%")
            if self._debug:
                optimal_actions, q_values,amaf_q_values,nodes = self.player.get_optimal_actions_qvalues()
                self._logger.debug('Optimal actions:',[i+1 for i in optimal_actions])
                self._logger.debug('Q values: ' +[round(i,4) for i in q_values])
                self._logger.debug('AMAF Q values: ' + [round(i,4) for i in amaf_q_values])
                for idx, node in enumerate(nodes):
                    self._logger.debug('Number of visits '+str(idx)+': '+ node['no_visits_actions'] + ' Q: ' + tuple(np.round(node['q_values'],3)) + ' AA: ',node['actions']+1)
    
    def _question_to_human_player(self, question: str) -> str:
        self._logger.info(f"Question to player: '{question}'")
        answer = input(question)
        self._logger.info(f"Answer from player: '{answer}'")
        return answer
    
    
    def _message_to_human_player(self, message: str) -> None:
        self._logger.info(f"Message to player: '{message}'")
        print(message)

def main():
    playConnect4_logger = Logger.create_PlayConnect4Logger()
    game_turn_handler = GameTurnHandler.GameTurnHandler()
    game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)
    playConnect4 = PlayConnect4(game, game_turn_handler, playConnect4_logger)
    
    playConnect4.setup_game()
    playConnect4.prepare_game()
    playConnect4.start_game()
    
    
if __name__ == '__main__':
    main()