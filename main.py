import numpy as np
import Connect4Game as C4Game
import Connect4Players as C4Players
import Connect4GameHandler as C4GameHandler
import DeepQModel
import time

def create_training_data(Handler,no_games=1,possible_rewards=[1,-1]):
    player_focus = 1 #player0 has player_value=1 and is red
    
    init_states = []
    actions = []
    rewards = []
    next_states = []
    terminations = []
    
    player_turn = np.random.choice(2) #the first player is random
    for _ in range(no_games):
        #play game
        Handler.reset_game()
        Handler.play_game(player_turn)
        
        #analyze game
        for idx in range(len(Handler.player_turns)):
            player_turn = Handler.player_turns[idx]
            if player_turn == player_focus:
                init_state = Handler.states[idx].flatten()
                action = Handler.actions[idx]
                if idx==len(Handler.states)-2:#winning move
                    next_state = np.zeros((init_state.shape))
                    reward = possible_rewards[0]
                    termination = 1
                elif idx==len(Handler.states)-3:#losing
                    next_state = np.zeros((init_state.shape))
                    reward = possible_rewards[1]
                    termination = 1
                else:
                    next_state = Handler.states[idx+2].flatten()
                    reward = 0
                    termination = 0
                
                init_states.append(init_state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                terminations.append(termination)
                
                player_turn = (player_turn + 1) % 2  #which player starts first
    
    training_data = [init_states,actions,rewards,next_states,terminations]
    return [np.array(data) for data in training_data]

def combine_training_data(data0, data1, max_elements = int(1e5)):
    new_data = [np.concatenate((tmpdata0,tmpdata1),axis=0) for tmpdata0, tmpdata1 in zip(data0,data1) ]
    len_data = new_data[0].shape[0]
    if len_data > max_elements:
        new_data = [data[-max_elements:] for data in new_data]
    return new_data

def create_batch(training_data, batch_size):
    len_training_data = len(training_data[0])
    batch_samples_idx = np.random.choice(len_training_data,size = batch_size, replace = False)
    batch = [data[batch_samples_idx] for data in training_data]
    return batch

def main0():
    player0 = C4Players.RandomPlayer()
    player1 = C4Players.RandomPlayer()
    
    game = C4Game.Connect4()
    
    #Create handler
    Handler = C4GameHandler.Connect4GameHandler(game, player0, player1)
    
    #play game
    np.random.seed(1002)
    Handler.play_game(0, True)
    game.plot_board_state()
    
    #play 100 games
    np.random.seed(None)
    Handler.reset_game()
    winners = Handler.play_n_games(1000,False)

def main1():
    player0 = C4Players.RandomPlayer()
    player1 = C4Players.RandomPlayer()
    
    game = C4Game.Connect4()
    
    #Create handler
    Handler = C4GameHandler.Connect4GameHandler(game, player0, player1)
    
    #deep Q model
    QModel = DeepQModel.DeepQModel(input_dim = game._no_rows * game._no_cols, 
                                   output_dim=game._no_cols, 
                                   no_layers = 6, 
                                   units = 15)
    
    #DP player
    DQPlayer0 = C4Players.DQPlayer(QModel.model,epsilon = 0.2)   
    DQPlayer1 = C4Players.DQPlayer(QModel.model,epsilon = 0)
    
    #DQ player vs random
    HandlerDvD = C4GameHandler.Connect4GameHandler(game, DQPlayer0, DQPlayer1)
    HandlerDvD.reset_game()
    
    HandlerDvR = C4GameHandler.Connect4GameHandler(game, DQPlayer1, player0)
    HandlerDvR.reset_game()
    
    Handlers = [HandlerDvD]
    
    batch_size = 32
    
    training_data = create_training_data(Handler,no_games = 100)
    winners = []
    for _ in range(1000):
        for __ in range(20):
            for tmp_handler in Handlers:
                new_training_data = create_training_data(tmp_handler,no_games = 1)
                print(_,__,tmp_handler.winner,len(tmp_handler.actions))
                winners.append(tmp_handler.winner)
                training_data = combine_training_data(training_data,new_training_data,max_elements=int(5e3))
                #get batch
                init_states, actions, rewards, next_states, terminations = create_batch(training_data,batch_size)
                #do gradient step
                QModel.do_gradient_step(init_states, actions, rewards, next_states, terminations)
        
        QModel.update_target_model()
        print('avg winner:',np.average(winners[-100:]))
    
    HandlerDvD.reset_game()
    HandlerDvD.play_n_games(10,True)
    
    HandlerDvR.reset_game()
    winners = HandlerDvR.play_n_games(100,False)
    print(np.average(winners))

def main2():
    random_player = C4Players.RandomPlayer()
    game = C4Game.Connect4()
    
    player = 1
    next_player = -1
    confidence_value = 1
    max_count = 1e2
    max_depth = 100
    
    rave0 = 2**0
    rave1 = 2**0
    cv0 = 2**2
    cv1 = 2**2
    tree0 = True
    tree1 = True
    
    seed = int(time.time())
    np.random.seed(seed)
    print('Seed',seed) 
    
    print('1 - rave:',rave0,'cv:',cv0,'tree:',tree0)
    print('-1 - rave:',rave1,'cv:',cv1,'tree:',tree1)
    
    MCTSPlayer0 = C4Players.MCTSPlayer(game,
                                       player = 1, 
                                       next_player = -1, 
                                       max_count = max_count, 
                                       max_depth = max_depth, 
                                       confidence_value = cv0,
                                       rave_param=rave0,
                                       reuse_tree=tree0,
                                       randomize_action = False,)
    MCTSPlayer1 = C4Players.MCTSPlayer(game, 
                                       player = -1, 
                                       next_player = 1, 
                                       max_count = max_count, 
                                       max_depth = max_depth, 
                                       confidence_value = cv1,
                                       rave_param=rave1,
                                       reuse_tree=tree1,
                                       randomize_action = False,)
   
    Handler = C4GameHandler.Connect4GameHandler(game, MCTSPlayer0, MCTSPlayer1)
    
    #Handler.play_game(0, True)
    #game.plot_board_state()
    
    winners = Handler.play_n_games(10,True,flip=False)
    seed += 1
    np.random.seed(seed)
    print(winners)
    print(np.average(winners))
    
    #best_action = MCTS.MonteCarloTreeSearch(game, player, next_player, 1e3, confidence_value, random_player)
    
if __name__ == '__main__':
    main2()
    
    

    
    
    

    
    
