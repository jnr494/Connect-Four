# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:57:25 2022

@author: mrgna
"""

import numpy as np
import numba

from IPlayer import IPlayer
import MonteCarloTreeSearch as MCTS
from Connect4Game import Connect4

class RandomPlayer(IPlayer):
    def __init__(self):
        pass
    
    def make_action(self, game, available_actions):
        return make_random_choice(available_actions)
    
    def reset(self):
        pass

class DQPlayer(IPlayer):
    def __init__(self, QModel, epsilon):
        self.QModel = QModel
        self.epsilon = epsilon
        self.RandomPlayer = RandomPlayer()
        
    def make_action(self, game, available_actions):
        uniform = np.random.uniform()
        if uniform < self.epsilon:
            return self.RandomPlayer.make_action(game, available_actions)
        else:
            q_values = self.QModel.predict(np.expand_dims(game.Board.flatten(),axis=0))
            q_values_available = q_values[0,available_actions]
            action = available_actions[np.argmax(q_values_available)]
            return action

    def reset(self):
        pass

class MCTSPlayer(IPlayer):
    def __init__(self, game: Connect4, player: int, next_player: int, max_count: int, max_depth: int, 
                 confidence_value: float, rave_param: float = None, reuse_tree: bool = True, randomize_action: bool = False):
        self.game = game
        self.player = player
        self.next_player = next_player
        self.max_count = max_count
        self.max_depth = max_depth
        self.confidence_value = confidence_value
        self.rave_param = rave_param
        self.RandomPlayer = RandomPlayer()
        self.reuse_tree = reuse_tree
        self.randomize_action = randomize_action
        
        self.reset()
        
    def make_action(self, game, available_actions):  
        self.game = game
        #important that env_state comes from game.
        best_action, tree, winning_probability = MCTS.MonteCarloTreeSearch(self.game, self.player, self.next_player, self.max_count, self.max_depth,
                                         self.confidence_value, self.rave_param, self.RandomPlayer, self.tree)
        
        self.winning_probability = winning_probability
        
        if self.reuse_tree:
            self.tree = tree
            
        if self.randomize_action:
            action_probabilities = MCTS.get_action_probabilities(self.game, tree, temperature=1)
            action = np.random.choice(self.game.no_cols,p=action_probabilities)
            return (action,action_probabilities)
        else: 
            return best_action
    
    def reset(self):
        self.tree = None
        self.winning_probability = None
        
    def get_optimal_actions_qvalues(self):
        best_actions, q_values,amaf_q_values,nodes = MCTS.get_optimal_tree_actions(self.game,self.tree,self.player,self.next_player)
        return best_actions, q_values,amaf_q_values,nodes
    
    
@numba.njit
def make_random_choice(available_actions):
    return np.random.choice(available_actions)