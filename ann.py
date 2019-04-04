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
        "batch_size":5, # number of examples per minibatch
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
    return w

  def init_bias(self):
    """ 
    Initialize bias vector
    """
    b = np.random.normal(self.m_weights, self.sigma_weights, (self.k,1))
    return b

  def train(self, X_train, Y_train):
    """
    train using minibatch gradient descent
    """
    # divide set into mini batches in some random way
    self.cost_history = []
    for i in range(self.epochs):
      print("--- epoch ", i, " ---")
      num_batches = int(self.n/self.batch_size)
      for j in range(num_batches-1):
        print("training on batch ", j, " out of ", num_batches)
        j_start = j * self.batch_size
        j_end = j*self.batch_size + self.batch_size
        X_batch = X_train[:,j_start:j_end]
        Y_batch = Y_train[:,j_start:j_end]
        Y_pred = self.evaluate(X_batch)
        grad_w, grad_b = self.compute_gradients(X_batch, Y_pred, Y_batch)
        self.w = self.w - self.lr*grad_w
        self.b = self.b - self.lr*grad_b
        self.compute_cost(X_batch, Y_batch)
    return self.cost_history

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
    """
    compute softmax activation, used in evaluating the prediction of the model
    """
    ones = np.ones(y_pred_lin.shape[0])
    y_pred = np.exp(y_pred_lin) / np.dot(ones.T, np.exp(y_pred_lin))
    return y_pred

  def compute_cost(self,data, Y_true):
    """
    compute the cost based on the current estimations of w and b
    """
    Y_pred = self.evaluate(data)
    num_exampl = data.shape[1]
    rglz = self.labda * (self.w**2).sum()
    cross_ent = self.cross_entropy(Y_true, Y_pred)
    cost = 1 / num_exampl * cross_ent + rglz
    self.cost_history.append(cost)
    # times reg
    return cost

  def cross_entropy(self,Y_true, Y_pred):
    """
    compute the cross entropy. Used for computing the cost.
    """
    before_log = np.sum(Y_true * Y_pred, axis=0)
    cross_ent = np.sum(-np.log(before_log))
    return cross_ent

  def compute_accuracy(self,X, y_true):
    """
    Compute the accuracy of the y_predictions of the model for a given data set
    """
    # y_pred = self.evaluate_classifier(X)
    # and then compare y_pred with y_true
    accuracy = 1
    return accuracy

  def compute_gradients(self, X_batch, y_true_batch, y_pred_batch):
    """
    compute the gradients of the loss, so the parameters can be updated in the direction of the steepest gradient. 
    """
    grad_batch = self.compute_gradient_batch(y_true_batch, y_pred_batch)
    grad_loss_w = 1/self.batch_size * np.dot(grad_batch, X_batch.T)
    grad_loss_b = 1/self.batch_size * np.dot(grad_batch, np.ones((self.batch_size,1)))
    # regularization is added to the weights but not to the bias
    grad_w = grad_loss_w + 2*self.labda*self.w
    grad_b = grad_loss_b
    return grad_w, grad_b

  def compute_gradient_batch(self, y_true_batch, y_pred_batch):
    """
    compute the gradient of a batch
    """
    grad_batch = - (y_true_batch - y_pred_batch)
    return grad_batch

  
