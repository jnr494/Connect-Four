# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 10:26:39 2023

@author: magnus
"""

import itertools

class GameTurnHandler:
    
    _current_player_value: int
    _next_player_value: int
    _player_values: list[int]
    
    def __init__(self):
        pass
    
    def reset(self, player_values: list[int], starting_position: int = 0):
        self._player_values = player_values
        
        self._player_turn_cycle = itertools.cycle(range(len(self._player_values)))
        self._player_turn_cycle = itertools.islice(self._player_turn_cycle, starting_position, None)
        
        self._current_player_turn = next(self._player_turn_cycle)
        self._next_player_turn = next(self._player_turn_cycle)
        
        self._current_player_value = self._player_values[self._current_player_turn]
        self._next_player_value = self._player_values[self._next_player_turn]
        
    def get_current_player_value(self) -> int:
        return self._current_player_value
    
    def get_next_player_value(self) -> int:
        return self._next_player_value
    
    def next_turn(self) -> None:
        self._current_player_turn = self._next_player_turn
        self._current_player_value = self._next_player_value
        
        self._next_player_turn = next(self._player_turn_cycle)
        self._next_player_value = self._player_values[self._next_player_turn]
    
    def copy(self):
        new_game_handler = GameTurnHandler()
        new_game_handler.reset(self._player_values, self._current_player_turn)
        return new_game_handler
    
if __name__ == '__main__':
    game_turn_handler = GameTurnHandler()
    game_turn_handler.reset([4,5,6,7,9],2)
    print(game_turn_handler.get_current_player_value(),game_turn_handler.get_next_player_value())
    game_turn_handler.next_turn()
    print(game_turn_handler.get_current_player_value(),game_turn_handler.get_next_player_value())
    game_turn_handler.next_turn()
    print(game_turn_handler.get_current_player_value(),game_turn_handler.get_next_player_value())
    
    new_game_turn_handler = game_turn_handler.copy()
    print("Copy")
    print(new_game_turn_handler.get_current_player_value(),new_game_turn_handler.get_next_player_value())
    new_game_turn_handler.next_turn()
    print(new_game_turn_handler.get_current_player_value(),new_game_turn_handler.get_next_player_value())
    game_turn_handler.next_turn()
    