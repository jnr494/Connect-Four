import unittest

import numpy as np

import ConfigHandler
import Connect4Game
import Connect4GameHandler
import Connect4Players
import GameTurnHandler
import LoggerHandler
import MCTSPlayerFactory


class GameHandlerTests(unittest.TestCase):
    def test_random_vs_normal_mcts(self: "GameHandlerTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()
        logger_handler = LoggerHandler.LoggerHandler(config_handler)

        game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)

        player0 = Connect4Players.RandomPlayer()
        player1 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(
            game,
            -1,
            1,
            "normal",
            config_handler,
            logger_handler,
        )
        game_handler = Connect4GameHandler.Connect4GameHandler(game, player0, player1, logger_handler, config_handler)
        number_of_games = 10
        winners = game_handler.play_n_games(number_of_games)

        self.assertAlmostEqual(np.sum(winners), -1 * number_of_games)

    def test_normal_vs_hard_mcts(self: "GameHandlerTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()
        logger_handler = LoggerHandler.LoggerHandler(config_handler)

        game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)

        player0 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(
            game,
            1,
            -1,
            "normal",
            config_handler,
            logger_handler,
        )
        player1 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(game, -1, 1, "hard", config_handler, logger_handler)
        game_handler = Connect4GameHandler.Connect4GameHandler(game, player0, player1, logger_handler, config_handler)
        number_of_games = 10

        np.random.seed(420)  # noqa: NPY002
        winners = game_handler.play_n_games(number_of_games)

        self.assertLess(np.sum(winners), -number_of_games/4)

    def test_normal_vs_god_mcts(self: "GameHandlerTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()
        logger_handler = LoggerHandler.LoggerHandler(config_handler)

        game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)

        player0 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(
            game,
            1,
            -1,
            "normal",
            config_handler,
            logger_handler,
        )
        player1 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(game, -1, 1, "god", config_handler, logger_handler)
        game_handler = Connect4GameHandler.Connect4GameHandler(game, player0, player1, logger_handler, config_handler)
        number_of_games = 2
        winners = game_handler.play_n_games(number_of_games)

        self.assertAlmostEqual(np.sum(winners), -1 * number_of_games)

    def test_god_vs_god_mcts_play_many_rounds(self: "GameHandlerTests") -> None:
        config_handler = ConfigHandler.ConfigHandler()
        logger_handler = LoggerHandler.LoggerHandler(config_handler)

        game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)

        player0 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(game, 1, -1, "god", config_handler, logger_handler)
        player1 = MCTSPlayerFactory.MCTSPlayerFactory.create_player(game, -1, 1, "god", config_handler, logger_handler)
        game_handler = Connect4GameHandler.Connect4GameHandler(game, player0, player1, logger_handler, config_handler)
        number_of_games = 1

        np.random.seed(72)  # noqa: NPY002

        #play game
        game_handler.play_n_games(number_of_games)
        rounds = game_handler.rounds

        self.assertGreater(np.average(rounds),30)