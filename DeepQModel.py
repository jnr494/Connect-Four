# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:30:05 2022

@author: mrgna
"""

#Used inpiration from https://keras.io/examples/rl/deep_q_network_breakout/

import tensorflow as tf

def create_NN(input_dim, output_dim, no_layers, units, activation = 'elu'):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units, input_shape=(input_dim,),activation=activation,kernel_regularizer='l2'))
        for _ in range(no_layers-1):
            model.add(tf.keras.layers.Dense(units, activation=activation,kernel_regularizer='l2'))

        model.add(tf.keras.layers.Dense(output_dim,activation='softsign',kernel_regularizer='l2'))
    
        return model

class DeepQModel:
    def __init__(self,input_dim, output_dim, no_layers, units, activation = 'elu',learning_rate = 0.001):
        self.output_dim = output_dim
        
        self.model = create_NN(input_dim, output_dim, no_layers, units, activation)
        self.model_target = create_NN(input_dim, output_dim, no_layers, units, activation)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.loss_funciton = tf.keras.losses.MeanSquaredError()
    
    def do_gradient_step(self,init_states, actions, rewards, next_states, terminations):
        #calculate targets from rewards and 'future rewards' from target_model
        targets = rewards + (terminations-2)*(-1)*tf.reduce_max(self.model_target.predict(next_states),axis=1)
        
        masks = tf.one_hot(actions, self.output_dim)
        
        with tf.GradientTape() as tape:
            #find model q values and calculate loss bassed on targets
            q_values = self.model(init_states)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_funciton(targets, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        
    def update_target_model(self):
        self.model_target.set_weights(self.model.get_weights())