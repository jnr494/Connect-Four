import unittest

import numpy as np

import Connect4Game
import ConfigHandler
import LoggerHandler
import Connect4Players
import MCTSPlayerFactory
import GameTurnHandler
import Connect4GameHandler

class Connect4GameTests(unittest.TestCase):
    def test_random_vs_normal_mcts(self: "Connect4GameTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()
        logger_handler = LoggerHandler.LoggerHandler(config_handler)

        game_turn_handler = GameTurnHandler.GameTurnHandler([1,-1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)

        player0 = Connect4Players.RandomPlayer()
        player1 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(game, -1, 1, "normal", config_handler)
        game_handler = Connect4GameHandler.Connect4GameHandler(game, player0, player1,logger_handler,config_handler)
        number_of_games = 10
        winners = game_handler.play_n_games(number_of_games)

        self.assertAlmostEqual(np.sum(winners),-1*number_of_games)
    
    def test_normal_vs_hard_mcts(self: "Connect4GameTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()
        logger_handler = LoggerHandler.LoggerHandler(config_handler)

        game_turn_handler = GameTurnHandler.GameTurnHandler([1,-1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)

        player0 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(game, 1, -1, "normal", config_handler)
        player1 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(game, -1, 1, "god", config_handler)
        game_handler = Connect4GameHandler.Connect4GameHandler(game, player0, player1,logger_handler,config_handler)
        number_of_games = 10
        winners = game_handler.play_n_games(number_of_games)

        self.assertAlmostEqual(np.sum(winners),-1*number_of_games)
