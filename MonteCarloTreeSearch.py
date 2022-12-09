# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:57:27 2022

@author: magnus
inspiared by: https://webdocs.cs.ualberta.ca/~hayward/396/jem/mcts.html
"""

import copy
import numpy as np
import numba

class Tree:
    def __init__(self):
        self.nodes = {}
    
    def new_node(self,state_hash, available_actions,next_row_height):
        new_node = {'no_visits':                0,
                    'actions':                  available_actions,
                    'actions_idx':              {action:idx for idx, action in enumerate(available_actions)},
                    'next_row_height':          next_row_height,
                    'no_visits_actions':        np.zeros(len(available_actions),dtype=np.int64),
                    'q_values':                 np.zeros(len(available_actions)),
                    'amaf_q_values':            np.zeros(len(available_actions)),
                    'amaf_no_visits_actions':   np.zeros(len(available_actions),dtype=np.int64),
                    'next_state_hash':          np.zeros(len(available_actions),dtype=np.int64),
                    }
                             
        self.nodes[state_hash] = new_node
        
        return new_node
    
    def is_node_in_tree(self, state_hash):
        if state_hash in self.nodes:
            return True
        else:
            return False
    
    def get_node(self, state_hash):
        return self.nodes[state_hash]
    
    def update_node(self, state_hash, action, reward, following_actions, following_row_heights):
        node = self.nodes[state_hash]
        
        action_idx = node['actions_idx'][action]

        node['no_visits'] += 1
        node['no_visits_actions'][action_idx] += 1
        node['q_values'][action_idx] += (reward - node['q_values'][action_idx]) / node['no_visits_actions'][action_idx]
    
        #update amaf
        for node_action in node['actions']:
            first_following_action_idx = np.where(following_actions==node_action)[0]
            if len(first_following_action_idx)>0:
                first_following_action_idx = first_following_action_idx[0]
                f_row_height = following_row_heights[first_following_action_idx]
                action_idx = node['actions_idx'][node_action]
                if f_row_height == node['next_row_height'][action_idx]:
                    node['amaf_no_visits_actions'][action_idx] += 1
                    node['amaf_q_values'][action_idx] += (reward - node['amaf_q_values'][action_idx]) / node['amaf_no_visits_actions'][action_idx]
                
        
    
    def update_next_state_hash(self,prev_state_hash,prev_action,current_state_hash):
        prev_node = self.get_node(prev_state_hash)
        prev_action_idx = prev_node['actions_idx'][prev_action]
        prev_node['next_state_hash'][prev_action_idx] = current_state_hash
        
    

def check_game_over(game, action, row, player):
    game_won = game.check_four_in_a_row(col = action, row = row, player = player)
    last_player = player
    if game_won:
        terminal_bool = True
        last_player_reward = 1
    else:
        #check if game is draw
        if len(game.get_available_actions())==0: 
            terminal_bool = True
            last_player_reward = 0.5
        else:
            terminal_bool = False
            last_player_reward = 0
    
    return terminal_bool, last_player, last_player_reward

def select_node_action_ucb1(node, confidence_value, rave_param, max_bool = True):    
    if rave_param is None:
        rave_bool = False
        rave_param = 0
    else:
        rave_bool = True
        
    return select_node_action_ucb1_numba(node['q_values'],
                                         node['no_visits'], 
                                         node['no_visits_actions'], 
                                         confidence_value, 
                                         node['amaf_q_values'],
                                         node['amaf_no_visits_actions'],
                                         rave_bool,
                                         rave_param,
                                         node['actions'], 
                                         max_bool)

@numba.njit
def select_node_action_ucb1_numba(q_values, no_visits, no_visits_actions, confidence_value,
                                  amaf_q_values, amaf_no_visits_actions,rave_bool,rave_param, 
                                  actions, max_bool):
    
    exploration_term = np.sqrt(np.log(no_visits) / (no_visits_actions+1e-8))    
    
    if rave_bool:
        beta = amaf_no_visits_actions / (no_visits_actions + amaf_no_visits_actions + 4 * no_visits_actions * amaf_no_visits_actions * rave_param**2)
        adjusted_q_values = (1 - beta) * q_values + beta * amaf_q_values
    else:
        adjusted_q_values =  q_values       
    
    if max_bool:
        ucb1_vals = adjusted_q_values + confidence_value * exploration_term
        select_action_idx = np.argmax(ucb1_vals)
    else:
        ucb1_vals = adjusted_q_values - confidence_value * exploration_term
        select_action_idx = np.argmin(ucb1_vals)
        
    select_action = actions[select_action_idx]
    
    return select_action

def get_optimal_action_and_next_state_hash(node,max_bool):
    best_action = select_node_action_ucb1(node,0,None,max_bool=max_bool)
    best_action_idx = node['actions_idx'][best_action]
    q_value = node['q_values'][best_action_idx]
    amaf_q_value = node['amaf_q_values'][best_action_idx]
    next_state_hash = node['next_state_hash'][best_action_idx]
    return best_action, q_value,amaf_q_value, next_state_hash

def get_optimal_tree_actions(game, tree, player, next_player):
    start_state_hash = hash(game.Board.tobytes())
    current_node = tree.get_node(start_state_hash)
    
    players = [player, next_player]
    player_turn = 0
    
    nodes = []
    state_hashes = []
    q_values = []
    best_actions = []
    amaf_q_values = []
    
    while True:
        best_action, q_value,amaf_q_value, next_state_hash = get_optimal_action_and_next_state_hash(current_node,players[player_turn] == player)
        nodes.append(current_node)
        state_hashes.append(next_state_hash)
        q_values.append(q_value)
        amaf_q_values.append(amaf_q_value)
        best_actions.append(best_action)
        if next_state_hash == 0:
            return best_actions, q_values, amaf_q_values, nodes
        
        player_turn = (player_turn + 1) % 2
        current_node = tree.get_node(next_state_hash)
        

def MonteCarloTreeSearch(game, player, next_player, max_count, max_depth, confidence_value, rave_param, rollout_player, tree = None):
    if tree is None:
        tree = Tree()

    counter = 0
    
    players = [player, next_player]
    game_copy = copy.deepcopy(game)
    
    while counter < max_count:
        game_copy.reset(game.Board)
        visited_state_hashes = []
        no_visited_states = len(visited_state_hashes)
        actions = []
        new_row_heights = []
        terminal_bool = False
        player_turn = 0
        
        ##selection
        while (not terminal_bool) and len(visited_state_hashes) <= max_depth:
            next_player_turn = (player_turn + 1) % 2
            
            current_state_hash = hash(game_copy.Board.tobytes())
            visited_state_hashes.append(current_state_hash)
            no_visited_states = len(visited_state_hashes)
            
            ##expansion and stop selection
            if not tree.is_node_in_tree(current_state_hash):
                clever_available_actions = game_copy.get_clever_available_actions(players[player_turn],players[next_player_turn])
                next_row_heights = game_copy.next_row_height[clever_available_actions]
                tree.new_node(current_state_hash,clever_available_actions,next_row_heights)
                if no_visited_states > 1:
                    tree.update_next_state_hash(visited_state_hashes[-2],actions[-1],current_state_hash)
                break
                
            if no_visited_states > 1:
                tree.update_next_state_hash(visited_state_hashes[-2],actions[-1],current_state_hash)
                
            
            ##selection continued
            #get node and find ucb1 optimal action
            current_node = tree.get_node(current_state_hash)
            selected_action = select_node_action_ucb1(current_node, confidence_value, rave_param, max_bool = players[player_turn] == player)
            actions.append(selected_action)
            
            #perform action
            new_row_height = game_copy.place_disc(selected_action, players[player_turn])  
            new_row_heights.append(new_row_height)
            #check if game is over
            terminal_bool, last_player, last_player_reward = check_game_over(game_copy, selected_action, new_row_height, players[player_turn])
            #update player turn
            player_turn = next_player_turn
            
        ##simulation        
        while not terminal_bool:
            next_player_turn = (player_turn + 1) % 2
            #get available actions    
            clever_available_actions = game_copy.get_clever_available_actions(players[player_turn],players[next_player_turn])
            #get and simulate action
            sim_action = rollout_player.make_action(game_copy.Board, clever_available_actions)
            actions.append(sim_action)
            new_row_height = game_copy.place_disc(sim_action, players[player_turn])
            new_row_heights.append(new_row_height)
            #check if game is over
            terminal_bool, last_player, last_player_reward = check_game_over(game_copy, sim_action, new_row_height, players[player_turn])
            #update player turn
            player_turn = next_player_turn
        
        ##backprogation
        if last_player == player: #from input
            player_reward = last_player_reward
        else:
            player_reward = 1 - last_player_reward
        
        #update player rewards
        for idx in range(0,len(visited_state_hashes)):
            tree.update_node(visited_state_hashes[idx],
                             actions[idx],
                             player_reward,
                             following_actions=actions[idx+2::2],
                             following_row_heights=new_row_heights[idx+2::2])   
        
        counter +=1
    
    #find best action
    start_node = tree.get_node(visited_state_hashes[0])
    best_root_action = select_node_action_ucb1(start_node,0,None)
    winning_probability = start_node['q_values'][start_node['actions_idx'][best_root_action]]
    return best_root_action, tree, winning_probability