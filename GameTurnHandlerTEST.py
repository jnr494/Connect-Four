"""
Created on Sun Oct 29 18:09:18 2023

@author: magnus
"""

import GameTurnHandler

def test_creation():
    game_turn_handler = GameTurnHandler.GameTurnHandler([4,5,6,7,9],3)
    assert game_turn_handler.get_current_player_value() == 7
    assert game_turn_handler.get_next_player_value() == 9
    
def test_empty_creation():
    game_turn_handler = GameTurnHandler.GameTurnHandler()
    assert game_turn_handler.get_current_player_value() == 0
    assert game_turn_handler.get_next_player_value() == 0
    
def test_next_turn():
    game_turn_handler = GameTurnHandler.GameTurnHandler([4,5,6,7,9])
    game_turn_handler.next_turn()
    assert game_turn_handler.get_current_player_value() == 5
    assert game_turn_handler.get_next_player_value() == 6
    
def test_copy():
    game_turn_handler = GameTurnHandler.GameTurnHandler([4,5,6,7,9])
    game_turn_handler.next_turn()
    next_game_turn_handler = game_turn_handler.copy()
    next_game_turn_handler.next_turn()
    
    assert next_game_turn_handler.get_current_player_value() == 6
    assert next_game_turn_handler.get_next_player_value() == 7
    
    #Also check that the original game_turn_handler did not go to the next turn
    assert game_turn_handler.get_current_player_value() == 5
    assert game_turn_handler.get_next_player_value() == 6

def run_test(test_function):
    try:
        test_function()
        return True
    except AssertionError as ex:
        template = "An exception of type {0} occurred in test: '{1}'"
        message = template.format(type(ex).__name__, test_function.__name__)
        print (message)
        return False
    
def run_gameturnhandler_tests():
    tests = [test_creation, test_empty_creation, test_next_turn, test_copy]
    result = True
    for test in tests:
        result = result & run_test(test)
        
    return result

if __name__ == "__main__":
    if run_gameturnhandler_tests():
        print("GameTurnHandler tests passed")
    else:
        print("GameTurnHandler tests failed")
        