"""
This file contains the ANN class, which implements a
one-layer neural network trained with stocastic gradient descent.

Author: Clara Tump

"""

import numpy as np
import matplotlib.pyplot as plt

class ANN:

  def __init__(self, data, targets, **kwargs):
    """
    Initialize Neural Network with data and parameters
    """
    var_defaults = {
        "lr": 0.01, #learning rate
        "m_weights": 0, #mean of the weights
        "sigma_weights": 0.01, #variance of the weights
        "labda": 0.1, # regularization parameter
        "batch_size":100, # #examples per minibatch
        "epochs":40, #number of epochs
        "h_param":1e-6 #parameter h for numerical grad check
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
        X_batch = X_train[:,j_start:j_end]
        Y_batch = Y_train[:,j_start:j_end]
        Y_pred = self.evaluate(X_batch)
        grad_b, grad_w = self.compute_gradients(X_batch, Y_batch, Y_pred)
        self.w = self.w - self.lr*grad_w
        self.b = self.b - self.lr*grad_b
      if verbosity:
        self.report_perf(i, X_train, Y_train, X_val, Y_val)
    self.plot_cost_and_acc()
    self.show_w()


  def check_gradients(self, X, Y, method='finite_diff'):
    """
    check the computation of the gradient against a numerical gradient computation.
    This can be done in two ways:
    a) by the fast but less accurate finite difference method
    b) by the slower but more accurate centered difference method
    The analytical (self computed) and numerical gradients of b and w are plotted for comparison
    """
    # initialize here so that it's initialized even if it doesnt satisfy the if or elif conditions
    grad_w_num = np.zeros((self.k, self.d))
    Y_pred = self.evaluate(X)
    grad_b, grad_w = self.compute_gradients(X, Y, Y_pred)
    if method == 'finite_diff':
      grad_b_num, grad_w_num = self.compute_gradient_num_fast(X, Y)
    elif method == 'centered_diff':
      grad_b_num, grad_w_num = self.compute_gradient_num_slow(X, Y)
    else:
      print(method, " IS NOT A VALID NUMERICAL GRADIENT CHECKING.")

    grad_w_vec = grad_w.flatten()
    grad_w_num_vec = grad_w_num.flatten()
    x_w = np.arange(1, grad_w_vec.shape[0] + 1)
    plt.bar(x_w, grad_w_vec, 0.35, label='Analytical gradient', color='blue')
    plt.bar(x_w+0.35, grad_w_num_vec, 0.35, label=method, color='red')
    plt.legend()
    plt.title(("Gradient check of w, batch size = " + str(X.shape[1])))
    plt.show()

    grad_b_vec = grad_b.flatten()
    grad_b_num_vec = grad_b_num.flatten()
    x_b = np.arange(1, grad_b.shape[0] + 1)
    plt.bar(x_b, grad_b_vec, 0.35, label='Analytical gradient', color='blue')
    plt.bar(x_b + 0.35, grad_b_num_vec, 0.35, label=method, color='red')
    plt.legend()
    plt.title(("Gradient check of b, batch size = " + str(X.shape[1])))
    plt.show()



  def evaluate(self, X):
    """
    use the classifier with current weights and bias to make a 
    prediction of the one-hot encoded targets (Y)
    test data: dxN
    w: Kxd
    b: Kx1
    output Y_pred = kxN
    """
    Y_pred = self.softmax(np.dot(self.w, X) + self.b)
    
    return Y_pred


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
    Y_pred = self.evaluate(X)
    num_exampl = X.shape[1]
    rglz = self.labda * np.sum(self.w ** 2)
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


  def compute_gradients(self, X_batch, y_true_batch, y_pred_batch):
    """
    compute the gradients of the loss, so the parameters can be updated in the direction of the steepest gradient. 
    """
    grad_batch = self.compute_gradient_batch(y_true_batch, y_pred_batch)
    b_grad_beforenorm = np.sum(grad_batch, axis=1).reshape(-1,1)
    grad_loss_w = 1/self.batch_size * np.dot(grad_batch, X_batch.T)
    grad_loss_b = 1/self.batch_size * b_grad_beforenorm
    # regularization is added to the weights but not to the bias
    grad_w = grad_loss_w + 2*self.labda*self.w
    grad_b = grad_loss_b
    return grad_b, grad_w


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
    Y_pred_train = self.evaluate(X_train)
    Y_pred_val = self.evaluate(X_val)
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
      w_image = self.w[i,:].reshape((32,32,3), order='F')
      w_image = ((w_image - w_image.min()) / (w_image.max() - w_image.min()))
      w_image = np.rot90(w_image, 3)
      plt.imshow(w_image)
      plt.xticks([])
      plt.yticks([])
      plt.title("Class " + str(i))
      plt.show()

  def compute_gradient_num_fast(self, X, Y_true):
    """
    Method to check the gradient calculation.
    Gradient computed numerically based on the finite difference method
    :param h_param: a parameter needed to be set for the finite difference method, usually around 1e-6
    """
    grad_w = np.zeros((self.k, self.d))
    grad_b = np.zeros((self.k,1))
    c = self.compute_cost(X, Y_true)
    for i in range(self.b.shape[0]):
      self.b[i] += self.h_param
      c2 = self.compute_cost(X, Y_true)
      grad_b[i] = (c2-c) / self.h_param
      self.b[i] -= self.h_param
    for i in range(self.w.shape[0]): #k
      for j in range(self.w.shape[1]): #d
        self.w[i,j] += self.h_param
        c2 = self.compute_cost(X, Y_true)
        grad_w[i,j] = (c2-c) / self.h_param
        self.w[i,j] -= self.h_param
    return grad_b, grad_w

  def compute_gradient_num_slow(self, X, Y_true):
    """
    Method to check the gradient calculation.
    Gradient computed numerically based on the centered difference method
    :param h_param: a parameter needed to be set for the centered difference method, usually around 1e-6
    """
    grad_w = np.zeros((self.k, self.d))
    grad_b = np.zeros((self.k, 1))
    for i in range(self.b.shape[0]):
      self.b[i] -= self.h_param
      c1 = self.compute_cost(X, Y_true)
      self.b[i] += 2*self.h_param
      c2 = self.compute_cost(X, Y_true)
      grad_b[i] = (c2 - c1) / (2*self.h_param)
      self.b[i] -= self.h_param
    for i in range(self.w.shape[0]):  # k
      for j in range(self.w.shape[1]):  # d
        self.w[i, j] -= self.h_param
        c1 = self.compute_cost(X, Y_true)
        self.w[i, j] += 2*self.h_param
        c2 = self.compute_cost(X, Y_true)
        grad_w[i, j] = (c2 - c1) / (2*self.h_param)
        self.w[i, j] -= self.h_param
    return grad_b, grad_w





