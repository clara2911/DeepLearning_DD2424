"""
This is the main file of Assignment 1 for DD2424 Deep Learning
This assignment implements a one-layer neural network.

Author: Clara Tump
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from ann_ass1 import ANN

def main():
  """
  Main method. 
  Load in train, validation and test set.
  Then train the neural network (ann) and test the performance. 
  """
  num_classes=10
  np.random.seed(18)

  # y = N, // X = dxN // Y=KxN
  X_train, y_train , Y_train = load_data(batch_file = "data/data_batch_1", k=num_classes)
  X_test, y_test, Y_test = load_data(batch_file = "data/test_batch", k=num_classes)
  X_val, y_val , Y_val = load_data(batch_file = "data/data_batch_2", k=num_classes)
  ann1 = ANN(X_train[:2,:100], Y_train[:, :100])
  ann1.check_gradients(X_train[:2,:100], Y_train[:, :100], method='finite_diff')
  # ann1.train(X_train, Y_train, X_val, Y_val, verbosity = True)
  # Y_pred_test = ann1.evaluate(X_test)
  # test_acc = ann1.compute_accuracy(Y_pred_test, Y_test)
  # print("------------------------------------------------------")
  # print("               FINAL TEST ACCURACY")
  # print(test_acc)
  # print("------------------------------------------------------")


def load_data(batch_file = "data/data_batch_1", num=None, feat_num=None, k=3):
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
    return X.T, y, Y.T


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
  Y = np.zeros((y.shape[0], k))
  Y[np.arange(y.shape[0]), y] = 1
  return Y
 

if __name__ == "__main__":
  main()
