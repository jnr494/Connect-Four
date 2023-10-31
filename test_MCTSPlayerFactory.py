import unittest

import Connect4Game
from ConfigHandler import ConfigHandler
from MCTSPlayerFactory import MCTSPlayerFactory, MCTSPlayerNames


class MCTSPlayerFactoryTests(unittest.TestCase):
    def test_creation_of_normal_player(self: "MCTSPlayerFactoryTests") -> None:
        game = Connect4Game.Connect4()
        config_handler = ConfigHandler()
        player = MCTSPlayerFactory.create_player(game, 1, -1, MCTSPlayerNames.normal, config_handler)
        available_actions = game.get_available_actions()
        action = player.make_action(game, available_actions)
        self.assertIn(action, available_actions)
        self.assertGreaterEqual(player.winning_probability, 0.5)
        player.reset()
        self.assertIsNone(player._tree)
        self.assertIsNone(player._tree)


    def test_creation_of_hard_player(self: "MCTSPlayerFactoryTests") -> None:
        game = Connect4Game.Connect4()
        config_handler = ConfigHandler()
        player = MCTSPlayerFactory.create_player(game, 1, -1, MCTSPlayerNames.hard, config_handler)
        available_actions = game.get_available_actions()
        action = player.make_action(game, available_actions)
        self.assertIn(action, available_actions)
        self.assertGreaterEqual(player.winning_probability, 0.5)

    def test_creation_of_god_player(self: "MCTSPlayerFactoryTests") -> None:
        game = Connect4Game.Connect4()
        config_handler = ConfigHandler()
        player = MCTSPlayerFactory.create_player(game, 1, -1, MCTSPlayerNames.god, config_handler)
        available_actions = game.get_available_actions()
        action = player.make_action(game, available_actions)
        self.assertIn(action, available_actions)
        self.assertGreaterEqual(player.winning_probability, 0.5)

    def test_mcts_player_reset(self: "MCTSPlayerFactoryTests") -> None:
        game = Connect4Game.Connect4()
        config_handler = ConfigHandler()
        player = MCTSPlayerFactory.create_player(game, 1, -1, MCTSPlayerNames.normal, config_handler)
        available_actions = game.get_available_actions()
        _ = player.make_action(game, available_actions)
        player.reset()
        self.assertIsNone(player._tree)
        self.assertIsNone(player.winning_probability)