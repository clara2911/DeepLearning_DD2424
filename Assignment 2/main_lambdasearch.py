"""
This is the main file of Assignment 1 for DD2424 Deep Learning
This assignment implements a one-layer neural network.

Author: Clara Tump
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from ann_ass2 import ANN

num_classes=10
np.random.seed(12)

def main():
  """
  Main method. 
  Load in train, validation and test set.
  Then train the neural network (ann) and test the performance. 
  """
  # y = N, // X = dxN // Y=KxN
  X_train, y_train, Y_train, X_val, y_val, Y_val = get_train_val_data()
  X_test, y_test, Y_test = load_data(batch_file = "data/test_batch")
  # try 10 values kind of range 10e-5 to 10e-1
  labdas_course = [2*1e-8, 4*1e-8, 6*1e-8, 8*1e-8, 1e-7, 2*1e-7, 4*1e-7, 6*1e-7, 8*1e-7, 2*1e-4, 4*1e-4, 6*1e-4, 8*1e-4, 1e-3, 2*1e-3, 4*1e-3, 6*1e-3, 8*1e-3]
  labda_final = 0.0008
  # for labda in labdas_course:
  #   params = {
  #     "labda": labda,
  #     "epochs": 32,#32 # two cycles
  #   }
  #   ann1 = ANN(X_train, Y_train, **params)
  #   ann1.train(X_train, Y_train, X_val, Y_val, verbosity= False)
    # Y_pred_val, act_h_test = ann1.evaluate(X_val)
    # val_acc = ann1.compute_accuracy(Y_pred_val, Y_val)
    # with open('results/lambda_fine_search.txt', 'a') as course_search_results:
    #   course_search_results.write("\n Lambda: "+str(labda))
    #   course_search_results.write("\n Validation accuracy: "+str(val_acc))
    #   course_search_results.write("\n --------------------------------")
    # print(" Lambda: ", labda)
    # print(" Validation accuracy: ", val_acc)
    # print("-------------------------------------------------------------------------------------------------------")

  params = {
    "labda": labda_final,
    "epochs": 14,  # 32 # two cycles
  }
  ann1 = ANN(X_train, Y_train, **params)
  ann1.train(X_train, Y_train, X_val, Y_val, verbosity=False)
  Y_pred_test, _ = ann1.evaluate(X_test)
  test_acc = ann1.compute_accuracy(Y_pred_test, Y_test)
  print("-------------------------------------------------------------------------------------------------------")
  print("Final test accuracy: ", test_acc, " with lambda = ", labda_final)

def get_train_val_data():
  X1, y1, Y1 = load_data(batch_file="data/data_batch_1", k=num_classes)
  X2, y2, Y2 = load_data(batch_file="data/data_batch_2", k=num_classes)
  X3, y3, Y3 = load_data(batch_file="data/data_batch_3", k=num_classes)
  X4, y4, Y4 = load_data(batch_file="data/data_batch_4", k=num_classes)
  X5, y5, Y5 = load_data(batch_file="data/data_batch_5", k=num_classes)
  X_all = np.concatenate((X1, X2, X3, X4, X5), axis=1)
  y_all = np.concatenate((y1, y2, y3, y4, y5))
  Y_all = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=1)
  X_train = X_all[:, :-500]
  y_train = y_all[:-500]
  Y_train = Y_all[:, :-500]
  X_val = X_all[:, -500:]
  y_val = y_all[-500:]
  Y_val = Y_all[:, -500:]
  return X_train, y_train, Y_train, X_val, y_val, Y_val


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
  Y = np.zeros((y.shape[0], k))
  Y[np.arange(y.shape[0]), y] = 1
  return Y


if __name__ == "__main__":
  main()
