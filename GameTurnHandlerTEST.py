# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:09:18 2023

@author: magnus
"""

import GameTurnHandler

def test_creation():
    game_turn_handler = GameTurnHandler.GameTurnHandler()
    game_turn_handler.reset([4,5,6,7,9],3)
    assert game_turn_handler.get_current_player_value() == 7
    assert game_turn_handler.get_next_player_value() == 9
    
def test_next_turn():
    game_turn_handler = GameTurnHandler.GameTurnHandler()
    game_turn_handler.reset([4,5,6,7,9])
    game_turn_handler.next_turn()
    assert game_turn_handler.get_current_player_value() == 5
    assert game_turn_handler.get_next_player_value() == 6
    
def test_copy():
    game_turn_handler = GameTurnHandler.GameTurnHandler()
    game_turn_handler.reset([4,5,6,7,9])
    game_turn_handler.next_turn()
    
    next_game_turn_handler = game_turn_handler.copy()
    next_game_turn_handler.next_turn()
    assert next_game_turn_handler.get_current_player_value() == 6
    assert next_game_turn_handler.get_next_player_value() == 7
    
    #Also check that the original game_turn_handler did not go to the next turn
    assert game_turn_handler.get_current_player_value() == 5
    assert game_turn_handler.get_next_player_value() == 6
    
def run_GameTurnHandler_tests():
    test_creation()
    test_next_turn()
    test_copy()

if __name__ == "__main__":
    run_GameTurnHandler_tests()
    print("Connect4Game tests passed")