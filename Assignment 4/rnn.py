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
        self.u = np.random.rand(self.m, self.k)*self.sig    # weight matrix 1
        self.w = np.random.rand(self.m, self.m)*self.sig    # weight matrix 2
        self.v = np.random.rand(self.k, self.m)*self.sig    # weight matrix 3
        

    def generate(self, h0, x0, n, unique_chars):
      """
      generate a sequence of n characters based on initial hidden state h0 and
      input character x0
      """
      char_seq = []
      h_prev = h0
      x_prev = x0
      for t in range(1,n+1):
        act_curr = np.dot(self.w, h_prev) + np.dot(self.u, x_prev)
        h_curr = np.tanh(act_curr)
        o_curr = np.dot(self.v, h_curr) + self.c
        prob = self.softmax(o_curr)
        prob = prob.reshape(prob.shape[0],)
        next_char = self.select_char(prob, unique_chars)
        char_seq.append(next_char)
        
        h_prev = h_curr
        x_prev = x0
        
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
      Use the conditional probabilities of each character at each timestep (probs)
      to randomly generate a sequence of characters
      """
      draw = str(np.random.choice(unique_chars, 1,
              p=prob))
      return draw

