"""
This file contains the ANN class, which includes all the machine learning of a selected ANN
Authors: Clara Tump
"""

import numpy as np

class ANN:

  def __init__(self, data, targets, **kwargs):
    """
    Initialize Neural Network with data and parameters
    """
    var_defaults = {
        "learning_rate": 0.5,
        "m_weights": 0,
        "sigma_weights": 0.01,
        "lambda": 0.5

    }

    for var, default in var_defaults.items():
        setattr(self, var, kwargs.get(var, default))

    self.d = data.shape[0]
    self.n = data.shape[1]
    self.k = targets.shape[0]
    self.w = self.init_weights()
    self.b = self.init_bias()


  def init_weights(self):
    """
    Initialize weight matrix
    """
    w = np.random.normal(self.m_weights, self.sigma_weights, (self.k, self.d))
    print("w shape: ", w.shape) #should be Kxd
    return w

  def init_bias(self):
    """
    Initialize weight matrix
    """
    b = np.random.normal(self.m_weights, self.sigma_weights, self.k)
    print("b shape: ", b.shape) # should be Kx1
    return b

def evaluate_classifier(test_data):
  """
  use the classifier with current weights and bias to make a prediction of the targets (y)
  test data: dxN
  w: Kxd
  b: Kx1
  output y_pred = kxN
  """
  y_pred = np.dot(self.w, test_data) + self.b
  # must be one hot KxN
  return y_pred

def compute_cost(data, y_true):
  y_pred = evaluate_classifier(X)
  cross_entropy(y_true, y_pred) + self.lambda*regularization
  return cost

def compute_accuracy(X, y_true):
  y_pred = evaluate_classifier(X)
  # and then compare y_pred with y_true
  return accuracy
