# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:57:25 2022

@author: mrgna
"""

import numpy as np
import MonteCarloTreeSearch as MCTS
import copy

class RandomPlayer:
    def __init__(self):
        pass
    
    def make_action(self, env_state, available_actions):
        return np.random.choice(available_actions)
    
    def reset(self):
        pass

class DQPlayer:
    def __init__(self, QModel, epsilon):
        self.QModel = QModel
        self.epsilon = epsilon
        self.RandomPlayer = RandomPlayer()
        
    def make_action(self, env_state, available_actions):
        uniform = np.random.uniform()
        if uniform < self.epsilon:
            return self.RandomPlayer.make_action(env_state, available_actions)
        else:
            q_values = self.QModel.predict(np.expand_dims(env_state.flatten(),axis=0))
            q_values_available = q_values[0,available_actions]
            action = available_actions[np.argmax(q_values_available)]
            return action

    def reset(self):
        pass

class MCTSPlayer:
    def __init__(self, game, player, next_player, max_count, max_depth, confidence_value, rave_param = None, reuse_tree = True):
        self.game = copy.deepcopy(game)
        self.player = player
        self.next_player = next_player
        self.max_count = max_count
        self.max_depth = max_depth
        self.confidence_value = confidence_value
        self.rave_param = rave_param
        self.RandomPlayer = RandomPlayer()
        self.tree = None
        self.reuse_tree = reuse_tree
        self.winning_probability = None
        
    def make_action(self, env_state, available_actions):  
        self.game.reset(env_state)
        #best_action, new_root = MCTS.MonteCarloTreeSearch(self.game, self.player, self.next_player, self.max_count, self.max_depth,
        #                                 self.confidence_value, self.RandomPlayer, None)################################################
        best_action, tree, winning_probability = MCTS.MonteCarloTreeSearch(self.game, self.player, self.next_player, self.max_count, self.max_depth,
                                         self.confidence_value, self.rave_param, self.RandomPlayer, self.tree)
        
        self.winning_probability = winning_probability
        
        if self.reuse_tree:
            self.tree = tree
        return best_action
    
    def reset(self):
        self.root_node = None
        
    def get_optimal_actions_qvalues(self):
        best_actions, q_values,amaf_q_values,nodes = MCTS.get_optimal_tree_actions(self.game,self.tree,self.player,self.next_player)
        return best_actions, q_values,amaf_q_values,nodes