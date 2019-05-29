"""
Deep Learning in Data Science Assignment 4
Text synthesis using a vanilla RNN

Author: Clara Tump
Last updated: 30-05-2019

RNN class.
Initializes and trains the RNN.


"""

class RNN:

    def __init__(self, txtfile):
        self.m = 100               #hidden layer length
        self.k = 80                # number of unique characters
        self.eta = 0.1             # learning rate
        self.seq_length = 25       # length of the input sequences
        self.b = np.zeros((m,1))   # bias vector
        self.c = np.zeros((k,1))   # ?
        self.sig = 0.01
        self.u = randn(self.m, self.k)*self.sig    # weight matrix 1
        self.w = randn(self.m, self.m)*self.sig    # weight matrix 2
        self.v = randn(self.k, self.m)*self.sig    # weight matrix 3
        


    def _read_data(self, txtfile):
      pass
