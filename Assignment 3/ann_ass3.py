"""
This file contains the ANN class, which implements a
one-layer neural network trained with stochastic gradient descent.

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
        "lr": 1e-5,  #learning rate
        "m_weights": 0,  #mean of the weights
        "sigma_weights": "sqrt_dims",  # variance of the weights: input a float or string "sqrt_dims" which will set it as 1/sqrt(d)
        "labda": 8*1e-4,  # regularization parameter
        "batch_size": 100,  # #examples per minibatch
        "epochs": 14,  #number of epochs
        "h_size": 50,  # number of nodes in the hidden layer
        "h_param": 1e-6,  # parameter h for numerical grad check
        "lr_max": 1e-1, # maximum for cyclical learning rate
        "lr_min": 1e-5, # minimum for cyclical learning rate
        "h_sizes":[20,20,50,10]
    }

    for var, default in var_defaults.items():
        setattr(self, var, kwargs.get(var, default))

    self.d = data.shape[0]
    self.n = data.shape[1]
    self.k = targets.shape[0]
    # self.m = self.hsize
    self.m = self.h_sizes
    self.num_hlayers = len(self.h_sizes)
    self.w = self.init_weight_mats()
    print("w.shape: ", len(self.w), self.w[0].shape)
    self.b = self.init_biases()
    print("b.shape: ", len(self.b), self.b[0].shape)
    self.ns = 2*int(self.n/self.batch_size)


  def init_weight_mats(self):
    """
    Initialize weight matrix
    """
    num_ws = self.num_hlayers+1
    sigma_weights_w = np.zeros(num_ws)
    if self.sigma_weights == "sqrt_dims":
      sigma_weights_w[0] = 1 / np.sqrt(self.d)
      for i in range(1, num_ws):
        sigma_weights_w[i] = 1/np.sqrt(self.m[i-1])
    elif type(sigma_weights == 'float'):
      for i in range(num_ws):
        sigma_weights_w[i] = self.sigma_weights
    else:
        print("ERROR: sigma_weights should either be a float or the string 'sqrt_dims'")
        exit()
    w = [np.zeros((2, 2))] * num_ws
    w[0] = np.random.normal(self.m_weights, sigma_weights_w[0], (self.m[0], self.d))
    for i in range(1, num_ws-1):
      w[i] = np.random.normal(self.m_weights, sigma_weights_w[i], (self.m[i], self.m[i-1]))
    w[num_ws-1] = np.random.normal(self.m_weights, sigma_weights_w[num_ws-1], (self.k, self.m[num_ws-2]))
    return w


  def init_biases(self):
    """
    Initialize bias vector for the first layer and the second layer
    """
    b = [None]*(self.num_hlayers+1)
    for i in range(self.num_hlayers):
      b[i] = np.zeros((self.m[i], 1))
    b[self.num_hlayers] = np.zeros((self.k, 1))
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
        X_batch = X_train[:, j_start:j_end]
        Y_batch = Y_train[:, j_start:j_end]
        Y_pred, act_h= self.evaluate(X_batch)
        grad_b, grad_w = self.compute_gradients(X_batch, Y_batch, Y_pred, act_h)
        for i in range(self.num_hlayers+1):
          self.w1 = self.w[i] - self.lr*grad_w[i]
          self.b1 = self.b[i] - self.lr*grad_b[i]
        self.lr = self.cyclic_lr(i*num_batches + j)
      self.report_perf(i, X_train, Y_train, X_val, Y_val, verbosity)
    self.plot_cost_and_acc()
    #self.show_w()

  def check_gradients(self, X, Y, method='finite_diff'):
    """
    check the computation of the gradient against a numerical gradient computation.
    This can be done in two ways:
    a) by the fast but less accurate finite difference method
    b) by the slower but more accurate centered difference method
    The analytical (self computed) and numerical gradients of b and w are plotted for comparison
    """
    grad_w_num = np.zeros((self.k, self.d))
    Y_pred, h_act = self.evaluate(X)
    grad_b1, grad_b2, grad_w1, grad_w2 = self.compute_gradients(X, Y, Y_pred, h_act)
    if method == 'finite_diff':
      grad_b1_num, grad_b2_num, grad_w1_num, grad_w2_num = self.compute_gradient_num_fast(X, Y)
    elif method == 'centered_diff':
      grad_b1_num, grad_b2_num, grad_w1_num, grad_w2_num = self.compute_gradient_num_slow(X, Y)
    else:
      print(method, " IS NOT A VALID NUMERICAL GRADIENT CHECKING.")

    grad_w1_vec = grad_w1.flatten()
    grad_w1_num_vec = grad_w1_num.flatten()
    x_w1 = np.arange(1, grad_w1_vec.shape[0] + 1)
    plt.bar(x_w1, grad_w1_vec, 0.35, label='Analytical gradient', color='blue')
    plt.bar(x_w1+0.35, grad_w1_num_vec, 0.35, label=method, color='red')
    plt.legend()
    plt.title(("Gradient check of w1, batch size = " + str(X.shape[1])))
    plt.show()

    grad_w2_vec = grad_w2.flatten()
    grad_w2_num_vec = grad_w2_num.flatten()
    x_w2 = np.arange(1, grad_w2_vec.shape[0] + 1)
    plt.bar(x_w2, grad_w2_vec, 0.35, label='Analytical gradient', color='blue')
    plt.bar(x_w2 + 0.35, grad_w2_num_vec, 0.35, label=method, color='red')
    plt.legend()
    plt.title(("Gradient check of w2, batch size = " + str(X.shape[1])))
    plt.show()

    grad_b1_vec = grad_b1.flatten()
    grad_b1_num_vec = grad_b1_num.flatten()
    x_b1 = np.arange(1, grad_b1.shape[0] + 1)
    plt.bar(x_b1, grad_b1_vec, 0.35, label='Analytical gradient', color='blue')
    plt.bar(x_b1 + 0.35, grad_b1_num_vec, 0.35, label=method, color='red')
    plt.legend()
    plt.title(("Gradient check of b1, batch size = " + str(X.shape[1])))
    plt.show()

    grad_b2_vec = grad_b2.flatten()
    grad_b2_num_vec = grad_b2_num.flatten()
    x_b2 = np.arange(1, grad_b2.shape[0] + 1)
    plt.bar(x_b2, grad_b2_vec, 0.35, label='Analytical gradient', color='blue')
    plt.bar(x_b2 + 0.35, grad_b2_num_vec, 0.35, label=method, color='red')
    plt.legend()
    plt.title(("Gradient check of b2, batch size = " + str(X.shape[1])))
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
    s1 = np.dot(self.w1, X) + self.b1
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
    grad_b2 = 1/self.batch_size * np.sum(grad_batch, axis=1).reshape(-1, 1)
    grad_batch = np.dot(self.w2.T, grad_batch)

    h_act_ind = np.zeros(h_act.shape)
    for i in range(h_act.shape[0]):
      for j in range(h_act.shape[1]):
        if h_act[i,j] > 0:
          h_act_ind[i, j] = 1
    grad_batch = grad_batch * h_act_ind
    grad_w1 = 1 / self.batch_size * np.dot(grad_batch, X_batch.T)
    grad_b1 = 1 / self.batch_size * np.sum(grad_batch, axis=1).reshape(-1, 1)

    grad_w1 = grad_w1 + 2 * self.labda * self.w1
    grad_w2 = grad_w2 + 2 * self.labda * self.w2
    return grad_b1, grad_b2, grad_w1, grad_w2


  def compute_gradient_batch(self, y_true_batch, y_pred_batch):
    """
    compute the gradient of a batch
    """
    grad_batch = - (y_true_batch - y_pred_batch)
    return grad_batch

  def cyclic_lr(self, t):
    """
    Update learning rate according to a cyclic learning rate scheme
    Learning rate increases linearly from 2*l*ns till (2*l+1)*ns and then decreases linearly again until 2*(l+1)*ns
    """
    l = int(t/(2*self.ns))
    if t < (2*l + 1)*self.ns:
      lr_t = self.lr_min + (t - 2*l*self.ns)/self.ns*(self.lr_max - self.lr_min)
    else:
      lr_t = self.lr_max - (t- (2*l+1)*self.ns)/self.ns*(self.lr_max - self.lr_min)
    return lr_t


  def report_perf(self, epoch, X_train, Y_train, X_val, Y_val, verbosity):
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
    if verbosity:
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
    plt.ylim(0, 3)
    plt.legend()
    plt.show()
    plt.savefig("results/loss_labda="+str(self.labda)+".png")
    plt.plot(x, self.acc_hist_tr, label = "Train accuracy")
    plt.plot(x, self.acc_hist_val, label = "Val accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0,0.8)
    plt.legend()
    plt.savefig("results/acc_labda=" + str(self.labda)+".png")
    plt.show()

  def show_w(self):
    # w has shape k * d
    for i in range(self.k):
      w_image = self.w1[i, :].reshape((32, 32, 3), order='F')
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
    grad_b1 = np.zeros((self.m, 1))
    grad_b2 = np.zeros((self.k, 1))
    grad_w1 = np.zeros((self.m, self.d))
    grad_w2 = np.zeros((self.k, self.m))

    c = self.compute_cost(X, Y_true)

    for i in range(self.b1.shape[0]):
      self.b1[i] += self.h_param
      c2 = self.compute_cost(X, Y_true)
      grad_b1[i] = (c2-c) / self.h_param
      self.b1[i] -= self.h_param

    for i in range(self.b2.shape[0]):
      self.b2[i] += self.h_param
      c2 = self.compute_cost(X, Y_true)
      grad_b2[i] = (c2-c) / self.h_param
      self.b2[i] -= self.h_param

    for i in range(self.w1.shape[0]): #k
      for j in range(self.w1.shape[1]): #d
        self.w1[i,j] += self.h_param
        c2 = self.compute_cost(X, Y_true)
        grad_w1[i,j] = (c2-c) / self.h_param
        self.w1[i,j] -= self.h_param

    for i in range(self.w2.shape[0]): #k
      for j in range(self.w2.shape[1]): #d
        self.w2[i,j] += self.h_param
        c2 = self.compute_cost(X, Y_true)
        grad_w2[i,j] = (c2-c) / self.h_param
        self.w2[i,j] -= self.h_param
    return grad_b1, grad_b2, grad_w1, grad_w2

  def compute_gradient_num_slow(self, X, Y_true):
    """
    Method to check the gradient calculation.
    Gradient computed numerically based on the centered difference method
    :param h_param: a parameter needed to be set for the centered difference method, usually around 1e-6
    """
    grad_b1 = np.zeros((self.m, 1))
    grad_b2 = np.zeros((self.k, 1))
    grad_w1 = np.zeros((self.m, self.d))
    grad_w2 = np.zeros((self.k, self.m))

    for i in range(self.b1.shape[0]):
      self.b1[i] -= self.h_param
      c1 = self.compute_cost(X, Y_true)
      self.b1[i] += 2*self.h_param
      c2 = self.compute_cost(X, Y_true)
      grad_b1[i] = (c2 - c1) / (2*self.h_param)
      self.b1[i] -= self.h_param

    for i in range(self.b2.shape[0]):
      self.b2[i] -= self.h_param
      c1 = self.compute_cost(X, Y_true)
      self.b2[i] += 2*self.h_param
      c2 = self.compute_cost(X, Y_true)
      grad_b2[i] = (c2 - c1) / (2*self.h_param)
      self.b2[i] -= self.h_param

    for i in range(self.w1.shape[0]):  # k
      for j in range(self.w1.shape[1]):  # d
        self.w1[i, j] -= self.h_param
        c1 = self.compute_cost(X, Y_true)
        self.w1[i, j] += 2*self.h_param
        c2 = self.compute_cost(X, Y_true)
        grad_w1[i, j] = (c2 - c1) / (2*self.h_param)
        self.w1[i, j] -= self.h_param

    for i in range(self.w2.shape[0]):  # k
      for j in range(self.w2.shape[1]):  # d
        self.w2[i, j] -= self.h_param
        c1 = self.compute_cost(X, Y_true)
        self.w2[i, j] += 2*self.h_param
        c2 = self.compute_cost(X, Y_true)
        grad_w2[i, j] = (c2 - c1) / (2*self.h_param)
        self.w2[i, j] -= self.h_param
    return grad_b1, grad_b2, grad_w1, grad_w2
