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
        "lr": 0.01, #learning rate
        "m_weights": 0, #mean of the weights
        "sigma_weights": 0.01, #variance of the weights
        "labda": 0, # regularization parameter
        "batch_size":5, # number of examples per minibatch
        "epochs":4 #number of epochs

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

  def train(self, X_train, Y_train, X_val, Y_val):
    """
    train using minibatch gradient descent
    """
    # divide set into mini batches in some random way
    self.cost_hist_tr = []
    self.cost_hist_val = []
    self.acc_hist_tr = []
    self.acc_hist_val = []
    for i in range(self.epochs):
      print("----------- epoch ", i, " -----------")
      num_batches = int(self.n/self.batch_size)
      for j in range(num_batches):
        print("training on batch ", j, " out of ", num_batches)
        j_start = j * self.batch_size
        j_end = j*self.batch_size + self.batch_size
        X_batch = X_train[:,j_start:j_end]
        Y_batch = Y_train[:,j_start:j_end]
        Y_pred = self.evaluate(X_batch)
        grad_w, grad_b = self.compute_gradients(X_batch, Y_pred, Y_batch)
        self.w = self.w - self.lr*grad_w
        self.b = self.b - self.lr*grad_b
      self.report_perf(X_train, Y_train, X_val, Y_val)
    self.plot_cost(cost_hist_tr, cost_hist_val, acc_hist_tr, acc_hist_val)

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
    # times reg
    return cost

  def cross_entropy(self,Y_true, Y_pred):
    """
    compute the cross entropy. Used for computing the cost.
    """
    before_log = np.sum(Y_true * Y_pred, axis=0)
    cross_ent = np.sum(-np.log(before_log))
    return cross_ent

  def compute_accuracy(self,Y_pred, Y_true):
    """
    Compute the accuracy of the y_predictions of the model for a given data set
    """
    y_pred = np.argmax(Y_pred, axis=0)
    y_true = np.argmax(Y_true, axis=0)
    # and then compare y_pred with y_true
    accuracy = 1
    # here the ys are lowercase so they are the index vectors, 
    # not the one hot encodings 
    correct = len(np.where(y_pred == y_true)[0])
    accuracy = correct/len(y_true)
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

  def report_perf(self, X_train, Y_train, X_val, Y_val):
    Y_pred_fullset = self.evaluate(X_train)
    Y_pred_val = self.evaluate(X_val)
    cost_fullset = self.compute_cost(X_train, Y_pred_fullset)
    acc_fullset = self.compute_accuracy(Y_pred_fullset, Y_train)
    cost_val = self.compute_cost(X_val, Y_pred_val)
    acc_val = self.compute_accuracy(Y_pred_val, Y_val)
    self.cost_hist_tr.append(cost_fullset)
    self.acc_hist_tr.append(acc_fullset)
    self.cost_hist_val.append(cost_val)
    self.cost_hist_val.append(acc_val)
    print("cost train: ", cost_fullset)
    print("accuracy train: ", acc_fullset)

  def plot_cost(cost_hist_tr, cost_hist_val, acc_hist_tr, acc_hist_val):
    x = range(1, len(cost_hist_tr) + 1)
    plt.plot(x, cost_hist_tr, label = "train loss")
    plt.plot(x,cost_hist_val, label = "val loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.plot(x, acc_hist_tr, label = "Train accuracy")
    plt.plot(x, acc_hist_val, label = "Val accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

  

  
