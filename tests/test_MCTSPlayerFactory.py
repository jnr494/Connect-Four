import unittest

import numpy as np

import Connect4Game
import GameTurnHandler
from ConfigHandler import ConfigHandler
from LoggerHandler import LoggerHandler
from MCTSPlayerFactory import MCTSPlayerFactory, MCTSPlayerNames


class MCTSPlayerFactoryTests(unittest.TestCase):
    def test_creation_of_normal_player(self: "MCTSPlayerFactoryTests") -> None:
        game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)
        config_handler = ConfigHandler()
        logger_handler = LoggerHandler(config_handler)
        player = MCTSPlayerFactory.create_player(game, 1, -1, MCTSPlayerNames.normal, config_handler, logger_handler)
        self.assertIsNone(player._mcts_config.rave_param)
        self.assertEqual(player._mcts_config.max_depth, 2)
        self.assertEqual(player._mcts_config.max_count, 500)
        available_actions = game.get_available_actions()
        action = player.make_action(game, available_actions)
        self.assertIn(action, available_actions)
        self.assertIsNotNone(player.winning_probability)
        if player.winning_probability is not None:
            self.assertGreaterEqual(player.winning_probability, 0.5)

    def test_creation_of_hard_player(self: "MCTSPlayerFactoryTests") -> None:
        game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)
        config_handler = ConfigHandler()
        logger_handler = LoggerHandler(config_handler)
        player = MCTSPlayerFactory.create_player(game, 1, -1, MCTSPlayerNames.hard, config_handler, logger_handler)
        self.assertIsNotNone(player._mcts_config.rave_param)
        self.assertEqual(player._mcts_config.max_depth, 5)
        self.assertEqual(player._mcts_config.max_count, 2000)
        available_actions = game.get_available_actions()
        action = player.make_action(game, available_actions)
        self.assertIn(action, available_actions)
        self.assertIsNotNone(player.winning_probability)
        if player.winning_probability is not None:
            self.assertGreaterEqual(player.winning_probability, 0.5)

    def test_creation_of_god_player(self: "MCTSPlayerFactoryTests") -> None:
        game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)
        config_handler = ConfigHandler()
        logger_handler = LoggerHandler(config_handler)
        player = MCTSPlayerFactory.create_player(game, 1, -1, MCTSPlayerNames.god, config_handler, logger_handler)
        self.assertIsNotNone(player._mcts_config.rave_param)
        self.assertEqual(player._mcts_config.max_depth, 100)
        self.assertEqual(player._mcts_config.max_count, 10000)
        available_actions = game.get_available_actions()
        action = player.make_action(game, available_actions)
        self.assertIn(action, available_actions)
        self.assertIsNotNone(player.winning_probability)
        if player.winning_probability is not None:
            self.assertGreaterEqual(player.winning_probability, 0.5)

    def test_mcts_player_reset(self: "MCTSPlayerFactoryTests") -> None:
        game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
        game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)
        config_handler = ConfigHandler()
        logger_handler = LoggerHandler(config_handler)
        player = MCTSPlayerFactory.create_player(game, 1, -1, MCTSPlayerNames.normal, config_handler, logger_handler)
        available_actions = game.get_available_actions()
        _ = player.make_action(game, available_actions)
        player.reset()
        self.assertIsNone(player._tree)
        self.assertIsNone(player.winning_probability)

    def test_god_player_first_few_moves(self: "MCTSPlayerFactoryTests") -> None:

        for seed in range(420,423):
            np.random.seed(seed)  # noqa: NPY002
            #Create game and god player
            game_turn_handler = GameTurnHandler.GameTurnHandler([1, -1])
            game = Connect4Game.Connect4(game_turn_handler=game_turn_handler)
            config_handler = ConfigHandler()
            logger_handler = LoggerHandler(config_handler)
            player = MCTSPlayerFactory.create_player(game, 1, -1, MCTSPlayerNames.god, config_handler, logger_handler)

            #First action should be 3
            action = player.make_action(game, game.get_clever_available_actions())
            self.assertEqual(action, 3)
            game.place_disc(action)
            game.next_turn()

            #Manual action for opponent
            action = 3
            game.place_disc(action)
            game.next_turn()

            #Second action should also be 3
            action = player.make_action(game, game.get_clever_available_actions())
            self.assertEqual(action, 3)
            game.place_disc(action)
            game.next_turn()

            #Manual action for opponent
            action = 3
            game.place_disc(action)
            game.next_turn()


