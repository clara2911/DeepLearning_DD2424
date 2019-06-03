"""
This is the main file of Assignment 1 for DD2424 Deep Learning
This assignment implements a k-layer neural network trained with 
stochastic gradient descent and a cyclical learning rate.

Author: Clara Tump




# TODO
# keep track of moving average
# change evaluate so that in test phase the mu and var are given instead of
# computed from the batch
# check the gradients
# 
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

def main():
  """
  Main method. 
  Load in train, validation and test set.
  Then train the neural network (ann) and test the performance. 
  """
  num_classes=10
  np.random.seed(12)


  # y = N, // X = dxN // Y=KxN
  X_train, y_train , Y_train = load_data(batch_file = "data_batch_1", k=num_classes)
  X_test, y_test, Y_test = load_data(batch_file = "test_batch")
  X_val, y_val , Y_val = load_data(batch_file = "data_batch_2")
  ann1 = ANN(X_train[:2,:3], Y_train[:,:3], batch_norm_flag = True)
  #ann1 = ANN(X_train, Y_train, batch_norm_flag = True)
  # uncomment this line to check gradients. remember to init ANN with same no. data points / feats
  ann1.check_gradients(X_train[:2, :3], Y_train[:, :3], method='centered_diff')
  #ann1.train(X_train[:2,:3], Y_train[:,:3], X_val[:2,:3], Y_val[:,:3], verbosity= True)
  #ann1.train(X_train, Y_train, X_val, Y_val, verbosity= True)
  #Y_pred_test, act_h_test, _,_,_,_,_ = ann1.evaluate(X_test)
  #test_acc = ann1.compute_accuracy(Y_pred_test, Y_test)
  #print("------------------------------------------------------")
  #print("               FINAL TEST ACCURACY")
  #print(test_acc)
  #print("------------------------------------------------------")


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
    return X.T, y, Y.T


def preprocess(chunk):
    """
    Preprocess the data. X (chunk) is N*k here.
    """
    # scale from pixel values to values between 0 and 1
    chunk = chunk / 255.0
    # normalize the data to have mean 0 and std 1
    mean_vec = np.mean(chunk, axis = 0)
    std_vec = np.std(chunk, axis =0)
    chunk = chunk - mean_vec
    chunk = chunk / std_vec
    return chunk


def one_hot(y, k):
  """
  Make a one-hot encoded representation of the labels vector
  """
  Y = np.zeros((y.shape[0], 10))
  Y[np.arange(y.shape[0]), y] = 1
  return Y


if __name__ == "__main__":
  main()
