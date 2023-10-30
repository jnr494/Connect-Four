import unittest

import Connect4Game
from MCTSPlayerFactory import MCTSPlayerFactory


class MCTSPlayerFactoryTests(unittest.TestCase):
    def test_creation_of_normal_player(self: "MCTSPlayerFactoryTests") -> None:
        game = Connect4Game.Connect4()
        player = MCTSPlayerFactory.create_normal_player(game, 1, -1)
        available_actions = game.get_available_actions()
        action = player.make_action(game, available_actions)
        self.assertIn(action, available_actions)
        self.assertGreaterEqual(player.winning_probability, 0.5)

    def test_creation_of_hard_player(self: "MCTSPlayerFactoryTests") -> None:
        game = Connect4Game.Connect4()
        player = MCTSPlayerFactory.create_hard_player(game, 1, -1)
        available_actions = game.get_available_actions()
        action = player.make_action(game, available_actions)
        self.assertIn(action, available_actions)
        self.assertGreaterEqual(player.winning_probability, 0.5)

    def test_creation_of_god_player(self: "MCTSPlayerFactoryTests") -> None:
        game = Connect4Game.Connect4()
        player = MCTSPlayerFactory.create_god_player(game, 1, -1)
        available_actions = game.get_available_actions()
        action = player.make_action(game, available_actions)
        self.assertIn(action, available_actions)
        self.assertGreaterEqual(player.winning_probability, 0.5)
