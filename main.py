"""
This is the main file of Assignment 1 for DD2424 Deep Learning
This assignment implements a one-layer neural network.

Author: Clara Tump
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from ann import ANN

def main():
  """
  Main method. 
  Load in train, validation and test set.
  Then train the neural network (ann) and test the performance. 
  """
  num_classes=10
  np.random.seed(12)

  # y = N, // X = dxN // Y=KxN
  X_train, y_train , Y_train = load_data(batch_file = "data/data_batch_1", num=500, feat_num=3072, k=num_classes)
  #X_test, y_test, Y_test = load_data(batch_file = "data/test_batch")
  #X_val, y_val , Y_val = load_data(batch_file = "data/data_batch_2", num=3)
  ann1 = ANN(X_train, Y_train)
  X_val, Y_val = X_train, Y_train
  ann1.train(X_train, Y_train, X_val, Y_val)


def load_data(batch_file = "data/data_batch_1", num=None, feat_num=None, k=10):
  """
  Load the data from the data files.
  """
  with open(batch_file, 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
    X = preprocess(data[b"data"])
    y = np.array(data[b"labels"])
    Y = one_hot(y, k)
  if num:
    return X[:num, :feat_num].T, y[:num], Y[:num].T
  else:
    return X, y, Y


def preprocess(chunk):
  """
  Preprocess the data:
  scale from pixel values to values between 0 and 1
  """
  chunk = chunk / 255.0
  return chunk


def one_hot(y, k):
  """
  Make a one-hot encoded representation (Y) of the labels vector (y)
  """
  Y = np.zeros((y.shape[0], 10))
  Y[np.arange(y.shape[0]), y] = 1
  return Y
 

if __name__ == "__main__":
  main()
