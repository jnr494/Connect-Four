# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:14:33 2022

@author: mrgna
"""

import numpy as np
import matplotlib.pyplot as plt
import numba
import copy

class Connect4():
    def __init__(self):
        self.no_rows = 6
        self.no_cols = 7
        self.blank_value = 0
        
        self.fig = None
        
        self.reset()
        
    def place_disc(self, col, player):
        #place disc
        row = self.next_row_height[col]
        self.Board[row,col] = player
        #update next_row_height
        self.next_row_height[col] += 1
        return row
    
    
    def reset(self, board = None):
        if board is None:
            self.Board = np.zeros((self.no_rows,self.no_cols))
            self.next_row_height = np.zeros((self.no_cols,),dtype=int)
        else:
            self.Board = copy.deepcopy(board)
            self.next_row_height = np.append((self.Board==self.blank_value),np.ones((1,self.no_cols)),axis=0).argmax(axis=0)
            
    
    def get_available_actions(self):
        board_top_row = self.Board[-1,:]
        available_actions = np.where(board_top_row==0)[0]
        return available_actions
    
    def get_clever_available_actions(self, player, next_player):
        winning_actions = self.get_winning_actions(player = player)
        if len(winning_actions) > 0:
            return winning_actions
        
        must_block_actions = self.get_winning_actions(player = next_player)
        if len(must_block_actions) > 0:
            return must_block_actions
        
        all_available_actions = self.get_available_actions()
        must_avoid_actions = self.get_winning_actions(player = next_player, use_2x_next_row_height=True)
        if len(must_avoid_actions) > 0:
            filtered_available_actions = np.setdiff1d(all_available_actions,must_avoid_actions)
            if len(filtered_available_actions) > 0: 
                return filtered_available_actions
            else: #game is lost if opponent plays optimally
                return all_available_actions
        else:
            return all_available_actions
    
    
    def get_winning_actions(self, player, use_2x_next_row_height = False):
        if use_2x_next_row_height:
            return find_available_winning_actions(self.Board, self.next_row_height+1, player, self.no_cols, self.no_rows)
        else:
            return find_available_winning_actions(self.Board, self.next_row_height, player, self.no_cols, self.no_rows)
    
    def check_four_in_a_row(self,col,row,player):
        return check_four_in_a_row_numba(col,row,player,self.Board,self.no_cols,self.no_rows)
        
    
    def plot_board_state(self, board_state = None, update = False):
        
        if board_state is None:
            board_state = self.Board
            
        no_rows, no_cols = board_state.shape
        
        #set x and y values and set radius    
        x = np.array(list(range(no_cols))*no_rows)
        y = np.array([list(range(no_rows)) for _ in range(no_cols)]).flatten('F')
        r = [1000]*len(x)
        
        #find colors
        color_map = {1:'r',-1:'y',0:'w'}
        colors = [color_map[v] for v in board_state.flatten()]
    
        #Create figure
        if self.fig is None or update is False:
            plt.ion()
            self.fig = plt.figure(figsize=(no_cols*2,no_rows))
            self.ax = self.fig.add_subplot(111)
            self.scatter = self.ax.scatter(x+0.5, y+0.5, s=r, c = colors)
            self.ax.set_facecolor("blue")
            plt.xticks(ticks = np.arange(7)+0.5, labels = np.arange(1,self.no_cols+1))
            self.ax.get_yaxis().set_visible(False)
            plt.xlim([0,no_cols])
            plt.ylim([0,no_rows])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        else:        
            self.scatter = self.ax.scatter(x+0.5, y+0.5, s=r, c = colors)
        
        if update is False:
            plt.show()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        

@numba.njit
def check_four_in_a_row_numba(col,row,player,Board,no_cols,no_rows):
    ##check up/down
    up_down = 1
    #check down 
    if row > 0:
        for offset in range(1,row+1):
            pass
            if Board[row-offset,col] == player:
                up_down += 1
                if up_down == 4:
                    return True
                continue
            else:
                break
    
    #check up. note we actually don't have to check up for newly places discs
    if row < no_rows -1:
        for offset in range(1,no_rows-row):
            if Board[row+offset,col] == player:
                up_down += 1
                if up_down == 4:
                    return True
                continue
            else:
                break
    
    ##check side2side
    side2side = 1
    #check left
    if col > 0:
        for offset in range(1,col+1):
            if Board[row,col-offset] == player:
                side2side += 1
                if side2side == 4:
                    return True
                continue
            else:
                break
    #check right
    if col < no_cols-1:
        for offset in range(1,no_cols-col):
            if Board[row,col+offset] == player:
                side2side += 1
                if side2side == 4:
                    return True
                continue
            else:
                break
    
    ##check diagonal piover4
    diag_piover4 = 1
    #check down/left
    max_rc_offset = np.minimum(row,col)
    if max_rc_offset > 0:
        for rc_offset in range(1,max_rc_offset+1):
            if Board[row-rc_offset,col-rc_offset] == player:
                diag_piover4 += 1
                if diag_piover4 == 4:
                    return True
                continue
            else:
                break
    #check up/right
    max_rc_offset = np.minimum(no_rows-1-row, no_cols-1-col)
    if max_rc_offset > 0:
        for rc_offset in range(1,max_rc_offset+1):
            if Board[row+rc_offset,col+rc_offset] == player:
                diag_piover4 += 1
                if diag_piover4 == 4:
                    return True
                continue
            else:
                break
            
    ##check diagonal 7piover4
    diag_7piover4 = 1
    #check up/left
    max_rc_offset = np.minimum(no_rows-1-row, col)
    if max_rc_offset > 0:
        for rc_offset in range(1,max_rc_offset+1):
            if Board[row+rc_offset,col-rc_offset] == player:
                diag_7piover4 += 1
                if diag_7piover4 == 4:
                    return True
                continue
            else:
                break
    #check down/right
    max_rc_offset = np.minimum(row, no_cols-1-col)
    if max_rc_offset > 0:
        for rc_offset in range(1,max_rc_offset+1):
            if Board[row-rc_offset,col+rc_offset] == player:
                diag_7piover4 += 1
                if diag_7piover4 == 4:
                    return True
                continue
            else:
                break
    
    return False

@numba.njit
def find_available_winning_actions(board, next_row_height, player, no_cols, no_rows):
    winning_moves = np.zeros(no_cols) #formatted as vector of 0 or 1 (1 being winning move)
    
    for col in range(no_cols):
        if next_row_height[col] < no_rows:
            winning_moves[col] = check_four_in_a_row_numba(col = col, 
                                                           row = next_row_height[col], 
                                                           player = player, 
                                                           Board = board, 
                                                           no_cols = no_cols, 
                                                           no_rows = no_rows)
    return np.where(winning_moves==1)[0]
    

if __name__ == '__main__':
    import time
    game = Connect4()
    game.plot_board_state()
    game.place_disc(2, 1)
    input('you ready to see some more? ')
    game.plot_board_state()
    time.sleep(3)
    
# =============================================================================
#     def defineplot():
#         x = np.linspace(0, 10*np.pi, 100)
#         y = np.sin(x)
#           
#         plt.ion()
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         sc = ax.scatter(x, y)
#         return fig, sc
#     
#     def updateplot(fig, sc, phase) :  
#         x = np.linspace(0, 10*np.pi, 100)
#         sc.set_offsets(np.c_[x,np.sin(0.5 * x + phase)])
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#     
#     fig,sc = defineplot()
#     for phase in np.linspace(0, 10*np.pi, 100):
#         updateplot(fig,sc,phase)
#     input('do you want to see more?')
#     for phase in np.linspace(0, 10*np.pi, 100):
#         updateplot(fig,sc,phase)
# =============================================================================
