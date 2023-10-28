# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 10:26:39 2023

@author: magnus
"""

import itertools

class GameTurnHandler:
    
    _current_player_value: int
    _next_player_value: int
    
    def __init__(self):
        pass
    
    def reset(self, player_values: list[int]):
        self._player_values_cycle = itertools.cycle(player_values)
        self.current_player_value = next(self._player_values_cycle)
        self.next_player_value = next(self._player_values_cycle)
        
    def get_current_player_value(self) -> int:
        return self.current_player_value
    
    def get_next_player_value(self) -> int:
        return self.next_player_value
    
    def next_turn(self) -> None:
        self.current_player_value = self.next_player_value
        self.next_player_value = next(self._player_values_cycle)
    
    
if __name__ == '__main__':
    mylist = [1,2,3]
    myGameTurnHandler = GameTurnHandler(mylist)
    print(myGameTurnHandler.get_current_player_value(), myGameTurnHandler.get_next_player_value())
    myGameTurnHandler.next_turn()
    print(myGameTurnHandler.get_current_player_value(), myGameTurnHandler.get_next_player_value())
    myGameTurnHandler.next_turn()
    print(myGameTurnHandler.get_current_player_value(), myGameTurnHandler.get_next_player_value())
    myGameTurnHandler.next_turn()
    print(myGameTurnHandler.get_current_player_value(), myGameTurnHandler.get_next_player_value())
    myGameTurnHandler.next_turn()