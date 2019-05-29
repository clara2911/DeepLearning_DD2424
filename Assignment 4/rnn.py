"""
Deep Learning in Data Science Assignment 4
Text synthesis using a vanilla RNN

Author: Clara Tump
Last updated: 30-05-2019

RNN class.
Initializes and trains the RNN.

"""

import numpy as np

class RNN:

    def __init__(self):
        self.m = 100               #hidden layer length
        self.k = 80                # number of unique characters
        self.eta = 0.1             # learning rate
        self.seq_length = 25       # length of the input sequences
        self.b = np.zeros((self.m,1))   # bias vector
        self.c = np.zeros((self.k,1))   # ?
        self.sig = 0.01
        
        # TODO check if u w v is indeed the right order and it shouldnt be u v w
        self.u = np.random.rand(self.m, self.k)*self.sig    # weight matrix 1
        self.w = np.random.rand(self.m, self.m)*self.sig    # weight matrix 2
        self.v = np.random.rand(self.k, self.m)*self.sig    # weight matrix 3
        self.grad_u = 0
        self.grad_w = 0
        self.grad_v = 0
        
    def generate(self, h0, x0, n, unique_chars):
      """
      generate a sequence of n one_hot encoded characters based on initial 
      hidden state h0 and input character x0
      
      Return: n*k matrix of n generated chars encoded as one-hot vectors
      """
      # TODO make sure that this predicts the next char not the current one
      char_seq = np.zeros((n, self.k))
      h_prev = h0
      x_prev = x0
      # for each of the timesteps in our n-length sequence
      for t in range(n):
        # compute probability of current character based on previous character
        act_curr = np.dot(self.w, h_prev) + np.dot(self.u, x_prev)
        h_curr = np.tanh(act_curr)
        o_curr = np.dot(self.v, h_curr) + self.c
        prob = self.softmax(o_curr)
        prob = prob.reshape(prob.shape[0],)
        
        # randomly choose a character for this timestep weighted by the probs
        predicted_char = self.select_char(prob, unique_chars)
        char_seq[t] = predicted_char
        
        # predict char t+1 based on the h and predicted character of timestep t
        h_prev = h_curr
        x_prev = predicted_char.reshape((predicted_char.shape[0],1))
        
      return char_seq
    
    def softmax(self, Y_pred_lin):
      """
      compute softmax activation, used in evaluating the prediction of the model
      """
      ones = np.ones(Y_pred_lin.shape[0])
      Y_pred = np.exp(Y_pred_lin) / np.dot(ones.T, np.exp(Y_pred_lin))
      return Y_pred
    
    def select_char(self, prob, unique_chars):
      """
      Use the conditional probabilities of a character
      to generate a one_hot character based on a prob-weighted random choice
      """
      # draw an int in [0,k]
      indices = list(range(self.k))
      int_draw = int(np.random.choice(indices, 1, p=prob)) 
      
      # convert int to one-hot 
      one_hot_draw = np.zeros(self.k)
      one_hot_draw[int_draw] = 1    
      return one_hot_draw
