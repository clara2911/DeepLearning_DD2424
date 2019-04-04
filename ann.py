"""
This file contains the ANN class, which includes all the machine learning of a selected ANN
Authors: Clara Tump
"""

import numpy as np
import math 
class ANN:

  def __init__(self, data, targets, **kwargs):
    """
    Initialize Neural Network with data and parameters
    """
    var_defaults = {
        "lr": 0.5, #learning rate
        "m_weights": 0, #mean of the weights
        "sigma_weights": 0.01, #variance of the weights
        "labda": 0.5, # regularization parameter
        "batch_size":2, # number of examples per minibatch
        "epochs":2 #number of epochs

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
    b = np.random.normal(self.m_weights, self.sigma_weights, (self.k,1))
    print("b shape: ", b.shape) # should be Kx1
    return b

  def train(self, X_train, Y_train):
    # divide set into mini batches in some random way
    for i in range(epochs):
      print("--- epoch ", i, " ---")
      num_batches = self.n/self.batch_size
      for j in num_batches:
        print("training on batch ", j, " out of ", num_batches)
        j_start = (j-1) * self.batch_size + 1
        j_end = j*self.batch_size
        X_batch = X_train[:,j_start:j_end]
        Y_batch = Y_train[:,j_start:j_end]
        Y_pred = self.evaluate(X_batch)
        grad_w, grad_b = self.compute_gradients(Y_pred, Y_batch)
        self.w = self.w - self.lr*grad_w
        self.b = self.b - self.lr*grad_b

  def evaluate(self,test_data):
    """
    use the classifier with current weights and bias to make a prediction of the targets (y)
    test data: dxN
    w: Kxd
    b: Kx1
    output y_pred = kxN
    """
    y_pred = self.softmax(np.dot(self.w, test_data) + self.b)
    # must be one hot KxN
    return y_pred

  def softmax(self, y_pred_lin):
    ones = np.ones(y_pred_lin.shape[0])
    y_pred = np.exp(y_pred_lin) / np.dot(ones.T, np.exp(y_pred_lin))
    return y_pred

  def compute_cost(self,data, Y_true):
    Y_pred = self.evaluate_classifier(X)
    num_exampl = data.shape[1]
    rglz = self.labda * (self.w**2).sum()
    cross_ent = self.cross_entropy(Y_true, Y_pred)
    cost = 1 / num_exampl * cross_ent + rglz
    # times reg
    return cost

  def cross_entropy(self,Y_true, Y_pred):
    return -np.log(np.dot(Y_true.T, Y_pred))

  def compute_accuracy(self,X, y_true):
    # y_pred = self.evaluate_classifier(X)
    # and then compare y_pred with y_true
    accuracy = 1
    return accuracy

  def compute_gradients(self, y_true_batch, y_pred_batch):
    grad_batch = self.compute_gradient_batch(y_true_batch, y_pred_batch)
    grad_loss_w = 1/self.batch_size * np.dot(grad_batch, X_batch.T)
    grad_loss_b = 1/self.batch_size * np.dot(grad_batch, np.ones(self.batch_size))
    grad_w = grad_loss_w + 2*self.labda*self.w
    grad_b = grad_loss_b
    return grad_w, grad_b

  def compute_gradient_batch(self, y_true_batch, y_pred_batch):
    grad_batch = - (y_true_batch - y_pred_batch)
    return grad_batch

  
