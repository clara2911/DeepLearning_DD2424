"""
This file contains the ANN class, which implements a
one-layer neural network trained with stocastich gradient descent.

Author: Clara Tump

NOTE
We're still setting rglz=0 somewhere in the cost computation
"""

import numpy as np
import matplotlib.pyplot as plt

class ANN:

  def __init__(self, data, targets, **kwargs):
    """
    Initialize Neural Network with data and parameters
    """
    var_defaults = {
        "lr": 0.1,  #learning rate
        "m_weights": 0,  #mean of the weights
        "sigma_weights": "sqrt_dims",  # variance of the weights: input a float or string "sqrt_dims" which will set it as 1/sqrt(d)
        "labda": 0,  # regularization parameter
        "batch_size": 100,  # #examples per minibatch
        "epochs": 40,  #number of epochs
        "h_size": 50  # number of nodes in the hidden layer
    }

    for var, default in var_defaults.items():
        setattr(self, var, kwargs.get(var, default))

    self.d = data.shape[0]
    self.n = data.shape[1]
    self.k = targets.shape[0]
    self.m = self.h_size
    self.w1, self.w2 = self.init_weight_mats()
    self.b1, self.b2 = self.init_biases()


  def init_weight_mats(self):
    """
    Initialize weight matrix
    """
    if self.sigma_weights == "sqrt_dims":
        sigma_weights_w1 = 1/np.sqrt(self.d)
        sigma_weights_w2 = 1/np.sqrt(self.m)
    elif type(sigma_weights == 'float'):
        sigma_weights_w1 = self.sigma_weights
        sigma_weights_w2 = self.sigma_weights
    else:
        print("ERROR: sigma_weights should either be a float or the string 'sqrt_dims'")
        exit()
    w1 = np.random.normal(self.m_weights, sigma_weights_w1, (self.m, self.d))
    w2 = np.random.normal(self.m_weights, sigma_weights_w2, (self.k, self.m))
    return w1, w2


  def init_biases(self):
    """
    Initialize bias vector for the first layer and the second layer
    """
    b1 = np.zeros((self.m, 1))
    b2 = np.zeros((self.k, 1))
    return b1, b2

  def train(self, X_train, Y_train, X_val, Y_val, verbosity=True):
    """
    train using minibatch gradient descent
    """
    self.cost_hist_tr = []
    self.cost_hist_val = []
    self.acc_hist_tr = []
    self.acc_hist_val = []
    num_batches = int(self.n/self.batch_size)

    for i in range(self.epochs):
      for j in range(num_batches):
        j_start = j * self.batch_size
        j_end = j*self.batch_size + self.batch_size
        X_batch = X_train[:, j_start:j_end]
        Y_batch = Y_train[:, j_start:j_end]
        Y_pred, act_h= self.evaluate(X_batch)
        grad_w1, grad_b1, grad_w2, grad_b2 = self.compute_gradients(X_batch, Y_batch, Y_pred, act_h)
        self.w1 = self.w1 - self.lr*grad_w1
        self.b1 = self.b1 - self.lr*grad_b1
        self.w2 = self.w2 - self.lr * grad_w2
        self.b2 = self.b2 - self.lr * grad_b2
      if verbosity:
        self.report_perf(i, X_train, Y_train, X_val, Y_val)
    self.plot_cost_and_acc()
    self.show_w()


  def evaluate(self, X):
    """
    use the classifier with current weights and bias to make a 
    prediction of the one-hot encoded targets (Y)
    test data: dxN
    w: Kxd
    b: Kx1
    output Y_pred = kxN
    """
    s1 = np.dot(self.w1, X) + self.b1
    zeros = np.zeros((self.m, self.batch_size)).T
    act_h = np.maximum(0, s1)
    s2 = np.dot(self.w2, act_h) + self.b2
    Y_pred = self.softmax(s2)
    return Y_pred, act_h

  def softmax(self, Y_pred_lin):
    """
    compute softmax activation, used in evaluating the prediction of the model
    """
    ones = np.ones(Y_pred_lin.shape[0])
    Y_pred = np.exp(Y_pred_lin) / np.dot(ones.T, np.exp(Y_pred_lin))
    return Y_pred


  def compute_cost(self, X, Y_true):
    """
    compute the cost based on the current estimations of w and b
    """
    Y_pred, act_h = self.evaluate(X)
    num_exampl = X.shape[1]
    rglz = self.labda * np.sum(self.w1**2) +  self.labda * np.sum(self.w2**2)
    cross_ent = self.cross_entropy(Y_true, Y_pred)
    cost = cross_ent / num_exampl + rglz
    return cost


  def cross_entropy(self, Y_true, Y_pred):
    """
    compute the cross entropy. Used for computing the cost.
    """
    vec = np.sum(Y_true * Y_pred, axis=0)
    cross_ent = np.sum(-np.log(vec), axis=0)
    return cross_ent


  def compute_accuracy(self,Y_pred, Y_true):
    """
    Compute the accuracy of the y_predictions of the model for a given data set
    """
    y_pred = np.array(np.argmax(Y_pred, axis=0))
    y_true = np.array(np.argmax(Y_true, axis=0))
    correct = len(np.where(y_true==y_pred)[0])
    accuracy = correct/y_true.shape[0]
    return accuracy


  def compute_gradients(self, X_batch, y_true_batch, y_pred_batch, h_act):
    """
    compute the gradients of the loss, so the parameters can be updated in the direction of the steepest gradient. 
    """

    # TODO make it for k layers instead of 2

    grad_batch = self.compute_gradient_batch(y_true_batch, y_pred_batch)

    # layer 2
    grad_w2 = 1/self.batch_size * np.dot(grad_batch, h_act.T)
    grad_b2 = 1/self.batch_size * np.sum(grad_batch, axis=1).reshape(-1,1)
    grad_batch = np.dot(self.w2.T, grad_batch)

    # layer 1
    grad_w1 = 1 / self.batch_size * np.dot(grad_batch, X_batch.T)
    grad_b1 = 1 / self.batch_size * np.sum(grad_batch, axis=1).reshape(-1, 1)

    # regularization is added to the weights but not to the bias
    grad_w1 = grad_w1 + 2*self.labda*self.w1
    grad_w2 = grad_w2 + 2 * self.labda * self.w2
    print("shape grad w1: ", grad_w1.shape)
    print("shape grad w2: ", grad_w2.shape)
    print("shape grad b1: ", grad_b1.shape)
    print("shape grad b2: ", grad_b2.shape)
    exit()
    return grad_w1, grad_b1, grad_w2, grad_b2

  # def compute_gradients(self, X_batch, y_true_batch, y_pred_batch):
  #   """
  #   compute the gradients of the loss, so the parameters can be updated in the direction of the steepest gradient.
  #   """
  #   grad_batch = self.compute_gradient_batch(y_true_batch, y_pred_batch)
  #   grad_loss_w = 1/self.batch_size * np.dot(grad_batch, X_batch.T)
  #   grad_loss_b = 1/self.batch_size * np.sum(grad_batch, axis=1).reshape(-1,1)
  #   # regularization is added to the weights but not to the bias
  #   grad_w = grad_loss_w + 2*self.labda*self.w1
  #   grad_b = grad_loss_b
  #   return grad_w, grad_b


  def compute_gradient_batch(self, y_true_batch, y_pred_batch):
    """
    compute the gradient of a batch
    """
    grad_batch = - (y_true_batch - y_pred_batch)
    return grad_batch


  def report_perf(self, epoch, X_train, Y_train, X_val, Y_val):
    """
    Compute and store the performance (cost and accuracy) of the model after every epoch, 
    so it can be used later to plot the evolution of the performance
    """
    Y_pred_train, act_h = self.evaluate(X_train)
    Y_pred_val, act_h_2 = self.evaluate(X_val)
    cost_train = self.compute_cost(X_train, Y_pred_train)
    acc_train = self.compute_accuracy(Y_pred_train, Y_train)
    cost_val = self.compute_cost(X_val, Y_pred_val)
    acc_val = self.compute_accuracy(Y_pred_val, Y_val)
    self.cost_hist_tr.append(cost_train)
    self.acc_hist_tr.append(acc_train)
    self.cost_hist_val.append(cost_val)
    self.acc_hist_val.append(acc_val)
    print("Epoch ", epoch, " // Train accuracy: ", acc_train, " // Train cost: ", cost_train)


  def plot_cost_and_acc(self):
    """
    Plot graphs for the evolution of the cost and accuracy through the epochs
    """
    x = list(range(1, len(self.cost_hist_tr) + 1))
    plt.plot(x, self.cost_hist_tr, label = "train loss")
    plt.plot(x, self.cost_hist_val, label = "val loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.plot(x, self.acc_hist_tr, label = "Train accuracy")
    plt.plot(x, self.acc_hist_val, label = "Val accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

  def show_w(self):
    # w has shape k * d
    for i in range(self.k):
      w_image = self.w1[i, :].reshape((32, 32, 3), order='F')

      w_image = ((w_image - w_image.min()) / (w_image.max() - w_image.min()))
      np.rot90(w_image, 1)
      plt.imshow(w_image)
      plt.xticks([])
      plt.yticks([])
      plt.title("Class " + str(i))
      plt.show()