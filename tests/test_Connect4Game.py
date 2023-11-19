import unittest

import numpy as np

from Connect4Game import Connect4
from GameTurnHandler import GameTurnHandler


class Connect4GameTests(unittest.TestCase):
    def test_init_empty_game(self: "Connect4GameTests") -> None:
        game = Connect4()

        # Assert board is array of zeros of correct size
        self.assertTrue(np.array_equal(game._get_board(), np.zeros((game._no_rows, game._no_cols))))

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
        self.assertEqual(game.get_last_player(),game_copy.get_last_player())

        game_copy.place_disc(3)
        game_copy.next_turn()

        self.assertEqual(game._get_board()[0, 3], 1)
        self.assertEqual(game._get_board()[1, 3], 0)
        self.assertEqual(game.get_current_player(), -1)

        self.assertEqual(game_copy._get_board()[0, 3], 1)
        self.assertEqual(game_copy._get_board()[1, 3], -1)
        self.assertEqual(game_copy.get_current_player(), 1)

    def test_reset(self: "Connect4GameTests") -> None:
        game_turn_handler = GameTurnHandler([1, -1])
        game = Connect4(game_turn_handler=game_turn_handler)

        game.place_disc(3)
        game.next_turn()

        game.reset()

        self.assertIsNone(game.get_last_player())
        self.assertIsNone(game.get_winner())
        self.assertTrue(np.array_equal(game._get_board(), np.zeros((game._no_rows, game._no_cols))), True)
        self.assertTrue(np.array_equal(game.next_row_height, np.zeros((game._no_cols,), dtype=int)), True)
        self.assertEqual(game_turn_handler.get_current_player_value(), 1)
        self.assertEqual(game.get_round(), 0)

    def test_last_player(self: "Connect4GameTests") -> None:
        game_turn_handler = GameTurnHandler([1, -1])
        game = Connect4(game_turn_handler=game_turn_handler)

        self.assertIsNone(game.get_last_player())

        game.place_disc(3)
        game.next_turn()

        self.assertEqual(game.get_last_player(), 1)

        game.place_disc(3)
        game.next_turn()

        self.assertEqual(game.get_last_player(), -1)

        game.next_turn()

        self.assertEqual(game.get_last_player(), -1)

    def test_get_round(self: "Connect4GameTests") -> None:
        game_turn_handler = GameTurnHandler([1, -1])
        game = Connect4(game_turn_handler=game_turn_handler)

        self.assertEqual(game.get_round(), 0)

        game.place_disc(3)
        game.next_turn()

        self.assertEqual(game.get_round(), 1)

        game.place_disc(3)
        game.next_turn()

        self.assertEqual(game.get_round(), 2)

    def test_is_draw(self: "Connect4GameTests") -> None:
        game_turn_handler = GameTurnHandler([1, -1])
        game = Connect4(game_turn_handler=game_turn_handler)

        self.assertFalse(game.is_draw())

        #First column
        for _ in range(6):
            game.place_disc(0)
            game.next_turn()
            self.assertFalse(game.is_draw())

        #Second column
        for _ in range(6):
            game.place_disc(1)
            game.next_turn()
            self.assertFalse(game.is_draw())

        #Third column
        for _ in range(6):
            game.place_disc(2)
            game.next_turn()
            self.assertFalse(game.is_draw())

        #First player plays column 6 to avoid winning
        game.place_disc(6)
        game.next_turn()
        self.assertFalse(game.is_draw())

        #Fourth column
        for _ in range(6):
            game.place_disc(3)
            game.next_turn()
            self.assertFalse(game.is_draw())

        #Fifth column
        for _ in range(6):
            game.place_disc(4)
            game.next_turn()
            self.assertFalse(game.is_draw())

        #Sixth column
        for _ in range(6):
            game.place_disc(5)
            game.next_turn()
            self.assertFalse(game.is_draw())

        #Last column
        for _ in range(5):
            self.assertFalse(game.is_draw())
            game.place_disc(6)
            game.next_turn()

        self.assertTrue(game.is_draw())
