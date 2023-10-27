# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 23:26:27 2023

@author: mrgna
"""

import Connect4Game

import numpy as np

def test_init_empty_game():
    game = Connect4Game.Connect4()
    
    #Assert board is array of zeros of correct size
    assert np.array_equal(game.Board, np.zeros((game.no_rows,game.no_cols)))

def run_Connect4Game_tests():
    test_init_empty_game()

if __name__ == "__main__":
    run_Connect4Game_tests()
    print("Connect4Game tests passed")