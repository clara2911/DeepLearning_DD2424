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

        self.params = self.init_params()
        self.grads = {}
        
    def init_params(self):
      params = {}
      self.sig = 0.01
      params['b'] = np.zeros((self.m,1))   # bias vector
      params['c'] = np.zeros((self.k,1))   # another bias vector
      params['u'] = np.random.rand(self.m, self.k)*self.sig    # weight matrix 1
      params['w'] = np.random.rand(self.m, self.m)*self.sig    # weight matrix 2
      params['v'] = np.random.rand(self.k, self.m)*self.sig    # weight matrix 3
      return params
      
        
    def train(self, X, Y, epochs=1):
      """
      Train the RNN using backpropagation through time (bptt)
      """
      # now putting this twice. need to find a smarter way
      self.seq_length = X.shape[1]
      for i in range(epochs):
        p, a, h = self.evaluate(X)
        loss = self.compute_loss(p, Y)
        self.compute_grads(X, Y, p, a, h)
        #update_params()
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
        a_curr = np.dot(self.params['w'], h_prev) + np.dot(self.params['u'], xt) + self.params['b']
        h_curr = np.tanh(a_curr)
        o_curr = np.dot(self.params['v'], h_prev) + self.params['c']
        p_curr = self.softmax(o_curr)
        
        a[t] = a_curr.reshape(self.m) #reshape from (m,1) to (m,)
        h[t] = h_curr.reshape(self.m) #reshape from (m,1) to (m,)
        p[t] = p_curr.reshape(self.k) #reshape from (k,1) to (k,)
        
        h_prev = h_curr
        
      return p, a, h
    
    def compute_loss(self, p, Y):
      """
      Compute the cross entropy loss between:
      - a (seq_len x k) matrix of predicted probabilities
      - a (k x seq _len) matrix of true one_hot encoded characters
      
      """
      # todo: check if not one of them needs to be transposed
      # TODO: this can be more efficient without the for loop
      loss = 0
      for t in range(self.seq_length):
        loss += -np.log(np.dot(Y[:,t].T, p[t]))
      return loss
    
    def softmax(self, Y_pred_lin):
      """
      compute softmax activation, used in evaluating the prediction of the model
      """
      ones = np.ones(Y_pred_lin.shape[0])
      Y_pred = np.exp(Y_pred_lin) / np.dot(ones.T, np.exp(Y_pred_lin))
      return Y_pred
    
    def compute_grads(self, X, Y, p, a, h):
      
      grad_o = -(Y.T - p)
      
      grad_v = np.zeros((self.k, self.m))
      for t in range(self.seq_length):
        grad_v += np.dot(grad_o[t].reshape(self.k,1), h[t].reshape(1,self.m))
      
      grad_a = np.zeros((self.seq_length, self.m))
      grad_h = np.zeros((self.seq_length, self.m))
      
      grad_h[-1] = np.dot(grad_o[-1], self.params['v'])    
      grad_h_last = grad_h[-1]      
      diag_part = np.diag(1-np.tanh(a[-1])**2)              
      grad_a[-1] = np.dot(grad_h_last, diag_part) 

           
      for t in reversed(range(self.seq_length-1)):        
        grad_h[t] = np.dot(grad_o[t], self.params['v']) + np.dot(grad_a[t+1], self.params['w'])
        grad_h_part = grad_h[t]
        diag_part = np.diag(1-np.tanh(a[t])**2)
        grad_a[t] = np.dot(grad_h_part, diag_part) 
      
      grad_c = grad_o.sum(axis = 0).reshape(self.k, 1)
      grad_b = grad_a.sum(axis = 0).reshape(self.m, 1)
      
      grad_w = np.zeros((self.m, self.m))
      for t in range(self.seq_length):
        grad_w += np.dot(grad_a[t].T, h[t-1])
           
      grad_u = np.zeros((self.m, self.k))
      for t in range(self.seq_length):
        grad_u += np.dot(grad_a[t].reshape(self.m,1), X[:,t-1].reshape(1,self.k))
      
      self.grads['u'] = grad_u
      self.grads['v'] = grad_v
      self.grads['w'] = grad_w
      self.grads['b'] = grad_b
      self.grads['c'] = grad_c
      
      
    
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
      

    def check_gradients(self, X, Y, h_param = 1e-5):
      p, a, h = self.evaluate(X)
      self.compute_grads(X, Y, p, a, h)

      num_grads = {}
      for key in self.grads:
        print("comparing numerical and own gradient for: ", key) 
        num_grads[key] = self.num_gradient(key, X, Y, h_param)
        own_grad = self.grads[key]
        print("num grad shape: ", num_grads[key].shape)
        print("own grad shape: ", self.grads[key].shape)
        error = np.sum(self.grads[key] - num_grads[key])
        print(key, " error: ", error)


    def num_gradient(self, key, X, Y, h_param):
      self.params[key] -= h_param
      p1, _, _ = self.evaluate(X)
      l1 = self.compute_loss(p1, Y)
      self.params[key] += h_param
      p2, _, _ = self.evaluate(X)
      l2 = self.compute_loss(p2, Y)
      num_grad = (l2-l1) / (2*h_param)
      return num_grad

    
      

      
