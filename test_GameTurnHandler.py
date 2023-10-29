"""
Created on Sun Oct 29 18:09:18 2023

@author: magnus
"""
import unittest
import GameTurnHandler

class GameTurnHandlerTests(unittest.TestCase):

    def test_creation(self):
        game_turn_handler = GameTurnHandler.GameTurnHandler([4,5,6,7,9],3)
        self.assertEqual(game_turn_handler.get_current_player_value(), 7)
        self.assertEqual(game_turn_handler.get_next_player_value(), 9)
        
    def test_empty_creation(self):
        game_turn_handler = GameTurnHandler.GameTurnHandler()
        self.assertEqual(game_turn_handler.get_current_player_value(), 0)
        self.assertEqual(game_turn_handler.get_next_player_value(), 0)
        
    def test_next_turn(self):
        game_turn_handler = GameTurnHandler.GameTurnHandler([4,5,6,7,9])
        game_turn_handler.next_turn()
        self.assertEqual(game_turn_handler.get_current_player_value(), 5)
        self.assertEqual(game_turn_handler.get_next_player_value(), 6)
        
    def test_copy(self):
        game_turn_handler = GameTurnHandler.GameTurnHandler([4,5,6,7,9])
        game_turn_handler.next_turn()
        next_game_turn_handler = game_turn_handler.copy()
        next_game_turn_handler.next_turn()
        
        self.assertEqual(next_game_turn_handler.get_current_player_value(), 6)
        self.assertEqual(next_game_turn_handler.get_next_player_value(), 7)
        
        #Also check that the original game_turn_handler did not go to the next turn
        self.assertEqual(game_turn_handler.get_current_player_value(), 5)
        self.assertEqual(game_turn_handler.get_next_player_value(), 6)