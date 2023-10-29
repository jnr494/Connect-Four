import unittest
import numpy as np

import Connect4Game


class Connect4GameTests(unittest.TestCase):

    def test_init_empty_game(self):
        game = Connect4Game.Connect4()
        
        #Assert board is array of zeros of correct size
        self.assertEqual(np.array_equal(game.Board, np.zeros((game.no_rows,game.no_cols))),True)
