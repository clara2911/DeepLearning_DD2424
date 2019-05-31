"""
Deep Learning in Data Science Assignment 4
Text synthesis using a vanilla RNN

Author: Clara Tump
Last updated: 30-05-2019

RNN class.
Initializes and trains the RNN.

"""

import numpy as np
import matplotlib.pyplot as plt

class RNN:

    def __init__(self):
        self.m = 20               #hidden layer length
        self.k = 80                # number of unique characters
        self.eta = 0.1             # learning rate
        self.seq_length = 5       # length of the input sequences
        self.e = 0
        self.cum_g2 = self.init_cum_gradient_squared()
        self.eps = 1e-8
        self.h_prev = np.zeros((self.m,1))

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
    
    def init_cum_gradient_squared(self):
      cum_sum = {}
      cum_sum['b'] = np.zeros((self.m,1))   # bias vector
      cum_sum['c'] = np.zeros((self.k,1))   # another bias vector
      cum_sum['u'] = np.zeros((self.m, self.k))    # weight matrix 1
      cum_sum['w'] = np.zeros((self.m, self.m))    # weight matrix 2
      cum_sum['v'] = np.zeros((self.k, self.m))    # weight matrix 3
      return cum_sum
      
        
    def train(self, data, epochs=8):
      """
      Train the RNN using backpropagation through time (bptt)
      """
      book_length = len(data.book_data)    
      smooth_loss = None
      for i in range(epochs):
        X, Y = self.get_matrices(data)
        p, a, h = self.evaluate(X)
        loss = self.compute_loss(p, Y)
        smooth_loss = self.compute_smooth_loss(loss, smooth_loss)
        self.compute_grads(X, Y, p, a, h)
        self.update_params()
        self.report_progress(smooth_loss)
        self.update_e(book_length)
        print("epoch ", i, " // loss: ", round(loss,3), " // smooth loss: ", round(smooth_loss,3))
        
    def get_matrices(self, data):
      """
      get X (input) and Y (labels) matrices from the list of characters
      """
      X_chars = data.book_data[self.e : self.e + self.seq_length]
      Y_chars = data.book_data[self.e + 1 : self.e + self.seq_length + 1]
      X = data.chars_to_onehot(X_chars)
      Y = data.chars_to_onehot(Y_chars)
      return X, Y
       
    def evaluate(self, X):
      """
      evaluates a sequence of one-hot encoded characters X and outputs a 
      probability vector at each X_t representing the predicted probs
      for the next character.
      Used as forward pass of the backpropagation through time (bptt)
      """     
      p = np.zeros((self.seq_length, self.k))
      a = np.zeros((self.seq_length, self.m))
      h = np.zeros((self.seq_length, self.m))
      
      for t in range(self.seq_length):
        xt = X[:,t].reshape((self.k, 1)) # reshape from (k,) to (k,1)
        a_curr = np.dot(self.params['w'], self.h_prev) + np.dot(self.params['u'], xt) + self.params['b']
        h_curr = np.tanh(a_curr)
        o_curr = np.dot(self.params['v'], self.h_prev) + self.params['c']
        p_curr = self.softmax(o_curr)
        
        a[t] = a_curr.reshape(self.m) #reshape from (m,1) to (m,)
        h[t] = h_curr.reshape(self.m) #reshape from (m,1) to (m,)
        p[t] = p_curr.reshape(self.k) #reshape from (k,1) to (k,)
        # TODO not 100% sure if this should be done every timestep or every
        # epoch
        self.h_prev = h_curr
        
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
    
    def compute_smooth_loss(self, loss, smooth_loss):
      """
      compute a smoothed version of the loss, since the simple loss fluctuates
      a lot
      """
      if smooth_loss == None:
        smooth_loss = loss
      else:
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
      return smooth_loss
    
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
      self.grads['b'] = 0.5*grad_b
      self.grads['c'] = 0.5*grad_c
       
    def update_params(self):
      """
      update the parameters according to AdaGrad gradient descent
      """
      for key in self.params:
        param = self.params[key]
        grad = self.grads[key]
        Gt = self.cum_g2[key] + np.square(grad)
        eps = self.eps * np.ones(Gt.shape)
        updated_param = param - self.eta / (np.sqrt(Gt + eps)) * grad
        self.params[key] = updated_param
        self.cum_g2[key] = Gt
           
    def update_e(self, book_length):
      """
      Update the counter of where we are in the book (e).
      If we are at the end of the book, we start at the beginning again.
      """
      new_e = self.e + self.seq_length
      if new_e > (book_length - self.seq_length -1):
        new_e = 0
      self.e = new_e
      
      
    def report_progress(self, smooth_loss):
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
        print("----------------------------------------------------------------")
        print("comparing numerical and own gradient for: " + str(key)) 
        print("----------------------------------------------------------------")
        num_grad = self.num_gradient(key, X, Y, h_param)
        own_grad = self.grads[key]
        print("num grad shape: ", num_grad.shape)
        print("own grad shape: ", self.grads[key].shape)
        error = np.sum(self.grads[key] - num_grad)
        
        grad_w_vec = own_grad.flatten()
        grad_w_num_vec = num_grad.flatten()
        x_w = np.arange(1, grad_w_vec.shape[0] + 1)
        plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
        plt.bar(x_w, grad_w_vec, 0.35, label='Analytical gradient', color='blue')
        plt.bar(x_w+0.35, grad_w_num_vec, 0.35, label='numerical gradient',  color='red')
        plt.legend()
        plt.title("Gradient check of: " + str(key))
        plt.show()
        rel_error = abs(grad_w_vec / grad_w_num_vec - 1)
        print("mean relative error: ", np.mean(rel_error))


    def num_gradient(self, key, X, Y, h_param):
      num_grad = np.zeros(self.grads[key].shape)
      if key == 'b' or 'c': # need to loop over 1 dim
        for i in range(self.params[key].shape[0]):
          self.params[key][i] -= h_param
          p1, _, _ = self.evaluate(X)
          l1 = self.compute_loss(p1, Y)
          self.params[key][i] += h_param
          p2, _, _ = self.evaluate(X)
          l2 = self.compute_loss(p2, Y)
          num_grad[i] = (l2-l1) / (2*h_param)
      else: # need to loop over 2 dimensions
        for i in range(self.params[key].shape[0]):
          for j in range(self.params[key].shape[1]):
            self.params[key][i,j] -= h_param
            p1, _, _ = self.evaluate(X)
            l1 = self.compute_loss(p1, Y)
            self.params[key][i,j] += h_param
            p2, _, _ = self.evaluate(X)
            l2 = self.compute_loss(p2, Y)
            num_grad[i,j] = (l2-l1) / (2*h_param)    
      return num_grad    
