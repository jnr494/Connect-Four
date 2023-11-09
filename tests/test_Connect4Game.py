import unittest

import numpy as np

from Connect4Game import Connect4
from GameTurnHandler import GameTurnHandler


class Connect4GameTests(unittest.TestCase):
    def test_init_empty_game(self: "Connect4GameTests") -> None:
        game = Connect4()

        # Assert board is array of zeros of correct size
        self.assertEqual(np.array_equal(game._get_board(), np.zeros((game._no_rows, game._no_cols))), True)

    def test_place_discs(self: "Connect4GameTests") -> None:
        game_turn_handler = GameTurnHandler([1, -1])
        game = Connect4(game_turn_handler=game_turn_handler)

        self.assertIsNone(game._winner)
        game.place_disc(3)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(3)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(0)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(6)
        self.assertIsNone(game._winner)

        self.assertEqual(game._get_board()[0, 3], 1)
        self.assertEqual(game._get_board()[1, 3], -1)
        self.assertEqual(game._get_board()[0, 0], 1)
        self.assertEqual(game._get_board()[0, 6], -1)

    def test_simple_win(self: "Connect4GameTests") -> None:
        game_turn_handler = GameTurnHandler([1, -1])
        game = Connect4(game_turn_handler=game_turn_handler)

        game.place_disc(3)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(2)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(3)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(2)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(3)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(2)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(3)
        self.assertEqual(game._winner, 1)

    def test_less_simple_win(self: "Connect4GameTests") -> None:
        game_turn_handler = GameTurnHandler([1, -1])
        game = Connect4(game_turn_handler=game_turn_handler)

        game.place_disc(1)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(0)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(4)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(0)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(2)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(0)
        self.assertIsNone(game._winner)
        game.next_turn()
        game.place_disc(3)
        self.assertEqual(game._winner, 1)

    def test_copy(self: "Connect4GameTests") -> None:
        game_turn_handler = GameTurnHandler([1, -1])
        game = Connect4(game_turn_handler=game_turn_handler)

        game.place_disc(3)
        game.next_turn()

        game_copy = game.copy()

        game_copy.place_disc(3)
        game_copy.next_turn()

        self.assertEqual(game._get_board()[0, 3], 1)
        self.assertEqual(game._get_board()[1, 3], 0)
        self.assertEqual(game.get_current_player(), -1)

        self.assertEqual(game_copy._get_board()[0, 3], 1)
        self.assertEqual(game_copy._get_board()[1, 3], -1)
        self.assertEqual(game_copy.get_current_player(), 1)
