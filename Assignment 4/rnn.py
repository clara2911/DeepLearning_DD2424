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
        self.seq_length = 0       # length of the input sequences
        
        
        
        # TODO check if u w v is indeed the right order and it shouldnt be u v w
        self.sig = 0.01
        self.b = np.zeros((self.m,1))   # bias vector
        self.c = np.zeros((self.k,1))   # another bias vector
        self.u = np.random.rand(self.m, self.k)*self.sig    # weight matrix 1
        self.w = np.random.rand(self.m, self.m)*self.sig    # weight matrix 2
        self.v = np.random.rand(self.k, self.m)*self.sig    # weight matrix 3
        self.grad_u = 0
        self.grad_w = 0
        self.grad_v = 0
        self.grad_b = 0
        self.grad_c = 0
        
    def train(self, X, Y, epochs=1):
      """
      Train the RNN using backpropagation through time (bptt)
      """
      # now putting this twice. need to find a smarter way
      self.seq_length = X.shape[1]
      for i in range(epochs):
        p, a, h = self.evaluate(X)
        loss = compute_loss(p, Y)
        compute_grads = (p, Y, h, a)
        update_params()
        print("epoch ", i, " // loss: ", loss)
        
    def evaluate(self, X):
      """
      evaluates a sequence of one-hot encoded characters X and outputs a 
      probability vector at each X_t representing the predicted probs
      for the next character.
      Used as forward pass of the backpropagation through time (bptt)
      """
      # can ignore this when training. or need to find a better place for it
      self.seq_length = X.shape[1]
      
      p = np.zeros((self.seq_length, self.k))
      a = np.zeros((self.seq_length, self.m))
      h = np.zeros((self.seq_length, self.m))
      
      h_prev = np.zeros((self.m,1))
      
      for t in range(self.seq_length):
        xt = X[:,t].reshape((self.k, 1)) # reshape from (k,) to (k,1)
        a_curr = np.dot(self.w, h_prev) + np.dot(self.u, xt) + self.b
        h_curr = np.tanh(a_curr)
        o_curr = np.dot(self.v, h_prev) + self.c
        p_curr = self.softmax(o_curr)
        
        a[t] = a_curr.reshape(self.m) #reshape from (m,1) to (m,)
        h[t] = h_curr.reshape(self.m) #reshape from (m,1) to (m,)
        p[t] = p_curr.reshape(self.k) #reshape from (k,1) to (k,)
        
        h_prev = h_curr
        
      return p, a, h
    
    def compute_loss(p, Y):
      """
      Compute the cross entropy loss between:
      - a (seq_len x k) matrix of predicted probabilities
      - a (k x seq _len) matrix of true one_hot encoded characters
      
      """
      # todo: check if not one of them needs to be transposed
      # TODO: this can be more efficient without the for loop
      loss = 0
      for t in range(self.seq_length):
        loss += -np.log(np.dot(Y[:,t].T, p[t])
      return loss
    
    def softmax(self, Y_pred_lin):
      """
      compute softmax activation, used in evaluating the prediction of the model
      """
      ones = np.ones(Y_pred_lin.shape[0])
      Y_pred = np.exp(Y_pred_lin) / np.dot(ones.T, np.exp(Y_pred_lin))
      return Y_pred
    
    def compute_grads(self, p, Y, a, h):
      pass
    
    def update_params():
      pass
      
        
        
    def generate(self, X, unique_chars):
      """
      generate a sequence of n one_hot encoded characters based on initial 
      hidden state h0 and input character x0
      
      Return: n*k matrix of n generated chars encoded as one-hot vectors
      """
      p, a, h = self.evaluate(X)
      char_seq = np.array([self.select_char(pt, unique_chars) for pt in p])
      return char_seq
    
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
