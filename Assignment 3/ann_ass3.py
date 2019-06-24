"""
This file contains the ANN class, which implements a
k-layer neural network trained with stochastic gradient descent 
and batch normlalization implemented from scratch.

Author: Clara Tump

"""

import numpy as np
import matplotlib.pyplot as plt

class ANN:

  def __init__(self, data, targets, batch_norm_flag = True, **kwargs):
    """
    Initialize Neural Network with data and parameters
    """
    var_defaults = {
        "lr": 1e-5,  #learning rate
        "m_weights": 0,  #mean of the weights
        "sigma_weights": "sqrt_dims",  # variance of the weights: input a float or string "sqrt_dims" which will set it as 1/sqrt(d)
        "labda": 8* 1e-3, #0, #8*1e-4,  # regularization parameter
        "batch_size": 100, # examples per minibatch
        "epochs": 30,  #number of epochs
        "h_param": 1e-6,  # parameter h for numerical grad check
        "lr_max": 1e-1, # maximum for cyclical learning rate
        "lr_min": 1e-5, # minimum for cyclical learning rate
        "h_sizes":[50, 50],
        "alpha": 0.7
    }

    for var, default in var_defaults.items():
        setattr(self, var, kwargs.get(var, default))
        
    self.batch_norm_flag = batch_norm_flag

    self.d = data.shape[0]
    self.n = data.shape[1]
    self.k = targets.shape[0]
    self.m = self.h_sizes
    self.num_hlayers = len(self.h_sizes)
    self.w = self.init_weight_mats()
    self.b = self.init_biases()
    print("n: ", self.n)
    self.ns = 5*int(self.n/self.batch_size)
    self.beta = self.init_biases()
    self.gamma = self.init_gamma()
    self.mu_avg = None
    self.var_avg = None
    


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
  
  def init_gamma(self):
    """
    Initialize gamma using He initialization
    """
    gamma = [None]*(self.num_hlayers)
    for i in range(self.num_hlayers):
      input_size = self.h_sizes[i]
      var = np.sqrt(2 / input_size)
      gamma[i] = np.random.normal(self.m_weights, var, (input_size,1))
    return gamma
  
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
        Y_pred, act_h, X_batch, s, normlz_s, mu, var = self.evaluate(X_batch, \
                                                batch_norm=self.batch_norm_flag)
        
        
        
        grad_b, grad_w, grad_beta, grad_gamma = self.compute_gradients(X_batch, \
                                   Y_batch, Y_pred, act_h, s, normlz_s, mu, var, \
                                   batch_norm=self.batch_norm_flag)
        self.update_params(grad_b, grad_w, grad_beta, grad_gamma, batch_norm = self.batch_norm_flag)
        
        self.lr = self.cyclic_lr(i*num_batches + j)
      self.report_perf(i, X_train, Y_train, X_val, Y_val, verbosity)
    self.plot_cost_and_acc()

  def check_gradients(self, X, Y, method='finite_diff'):
    """
    check the computation of the gradient against a numerical gradient computation.
    This can be done in two ways:
    a) by the fast but less accurate finite difference method
    b) by the slower but more accurate centered difference method
    The analytical (self computed) and numerical gradients of b and w are plotted for comparison
    """
    grad_w_num = np.zeros((self.k, self.d))
    Y_pred, h_act, _, s, normlz_s, mu, var = self.evaluate(X, batch_norm=self.batch_norm_flag)
    grad_b, grad_w, grad_beta, grad_gamma  = self.compute_gradients(X, Y, Y_pred, h_act, s, normlz_s, mu, var, batch_norm=self.batch_norm_flag)
    if method == 'finite_diff':
      grad_b_num, grad_w_num = self.compute_gradient_num_fast(X, Y)
    elif method == 'centered_diff':
      grad_b_num, grad_w_num= self.compute_gradient_num_slow(X, Y)
    else:
      print(method, " IS NOT A VALID NUMERICAL GRADIENT CHECKING.")

    for k in range(self.num_hlayers + 1):
        
        print("-------------------- W",k," gradients ------------------------")
        print("METHOD = ", method)
        grad_w_vec = grad_w[k].flatten()
        grad_w_num_vec = grad_w_num[k].flatten()
        x_w = np.arange(1, grad_w_vec.shape[0] + 1)
        plt.bar(x_w, grad_w_vec, 0.35, label='Analytical gradient', color='blue')
        plt.bar(x_w+0.35, grad_w_num_vec, 0.35, label=method, color='red')
        plt.legend()
        plt.title(("Gradient check of w", k, ", batch size = " + str(X.shape[1])))
        plt.show()
        rel_error = abs((grad_w_vec+self.h_param**2) / (grad_w_num_vec+self.h_param**2) - 1)
        
        print("mean relative error: ", np.mean(rel_error))


        print("--------------------- B",k," gradients ------------------------")
        grad_b_vec = grad_b[k].flatten()
        grad_b_num_vec = grad_b_num[k].flatten()
        x_b = np.arange(1, grad_b[k].shape[0] + 1)
        plt.bar(x_b, grad_b_vec, 0.35, label='Analytical gradient', color='blue')
        plt.bar(x_b + 0.35, grad_b_num_vec, 0.35, label=method, color='red')
        plt.legend()
        plt.title(("Gradient check of b", k, ", batch size = " + str(X.shape[1])))
        plt.show()
        rel_error = abs((grad_b_vec+self.h_param**2) / (grad_b_num_vec+self.h_param**2) - 1)
        
        print("mean relative error: ", np.mean(rel_error))


  def evaluate(self, X, batch_norm=True):
    """
    use the classifier with current weights and bias to make a 
    prediction of the one-hot encoded targets (Y)
    test data: dxN
    w: Kxd
    b: Kx1
    output Y_pred = kxN
    """
    
    act_h = [None] * (self.num_hlayers)
    s = [None] * (self.num_hlayers+1)
    normlz_s = [None] * (self.num_hlayers)
    final_s = [None] * (self.num_hlayers)
    mu = [None] * (self.num_hlayers)
    var = [None] * (self.num_hlayers)
    
    # first layer
    s[0] = np.dot(self.w[0], X) + self.b[0]
    if batch_norm:
        mu[0] = 1/s[0].shape[1] * np.sum(s[0], axis = 1)
        var[0] = 1/ s[0].shape[1] * np.sum(((s[0].T - mu[0]).T)**2, axis = 1)
        normlz_s[0] = self.batch_norm(s[0], mu[0], var[0])
        final_s[0] = self.gamma[0] * normlz_s[0] + self.beta[0]
        act_h[0] = np.maximum(0, final_s[0])
    else:
        act_h[0] = np.maximum(0, s[0])
    
    # hidden layers
    for i in range(1, self.num_hlayers):
      s[i] = np.dot(self.w[i], act_h[i-1]) + self.b[i] 
      if batch_norm:
        mu[i] = 1/s[i].shape[1] * np.sum(s[i], axis = 1)
        var[i] = 1/ s[i].shape[1] * np.sum(((s[i].T - mu[i]).T)**2, axis = 1)
        normlz_s[i] = self.batch_norm(s[i], mu[i], var[i])
        final_s[i] = self.gamma[i] * normlz_s[i] + self.beta[i]
        act_h[i] = np.maximum(0, final_s[i])
      else:
        act_h[i] = np.maximum(0, s[i])
        
    # last layer
    lst = self.num_hlayers
    s[lst] = np.dot(self.w[lst], act_h[lst-1]) + self.b[lst]
    Y_pred = self.softmax(s[lst])
    
    # update running averages of mean and variance (needed in test phase)
    if batch_norm:
      self.update_mu_var_avg(mu, var)
    
    return Y_pred, act_h, X, s, normlz_s, mu, var
  
  def update_mu_var_avg(self, mu, var):
    if self.mu_avg == None:
      self.mu_avg = mu
    else:
      self.mu_avg = [self.alpha * self.mu_avg[l] + (1-self.alpha) * mu[l] for l in range(len(mu))]
    if self.var_avg == None:
      self.var_avg = var
    else:
      self.var_avg = [self.alpha * self.var_avg[l] + (1-self.alpha) * var[l] for l in range(len(var))]
  
  
  
  def evaluate_test(self, X, batch_norm=True):
    """
    use the classifier with current weights and bias to make a 
    prediction of the one-hot encoded targets (Y)
    test data: dxN
    w: Kxd
    b: Kx1
    output Y_pred = kxN
    """
    
    act_h = [None] * (self.num_hlayers)
    s = [None] * (self.num_hlayers+1)
    normlz_s = [None] * (self.num_hlayers) 
    final_s = [None] * (self.num_hlayers)
    mu = [None] * (self.num_hlayers)
    var = [None] * (self.num_hlayers)
    
    # first layer
    s[0] = np.dot(self.w[0], X) + self.b[0]
    if batch_norm:
        normlz_s[0] = self.batch_norm(s[0], self.mu_avg[0], self.var_avg[0])
        final_s[0] = self.gamma[0] * normlz_s[0] + self.beta[0]
        act_h[0] = np.maximum(0, final_s[0])
    else:
        act_h[0] = np.maximum(0, s[0])
    
    # hidden layers
    for i in range(1, self.num_hlayers):
      s[i] = np.dot(self.w[i], act_h[i-1]) + self.b[i] 
      if batch_norm:
        normlz_s[i] = self.batch_norm(s[i], mu_avg[i], var_avg[i])
        final_s[i] = self.gamma[i] * normlz_s[i] + self.beta[i]
        act_h[i] = np.maximum(0, final_s[i])
      else:
        act_h[i] = np.maximum(0, s[i])
        
    # last layer
    lst = self.num_hlayers
    s[lst] = np.dot(self.w[lst], act_h[lst-1]) + self.b[lst]
    Y_pred = self.softmax(s[lst]) 
    return Y_pred, act_h, X, s, normlz_s, mu, var

  def softmax(self, Y_pred_lin):
    """
    compute softmax activation, used in evaluating the prediction of the model
    """
    ones = np.ones(Y_pred_lin.shape[0])
    Y_pred = np.exp(Y_pred_lin) / np.dot(ones.T, np.exp(Y_pred_lin))
    return Y_pred
  
  def batch_norm(self, s, mu, var):
    part1 = np.diag(np.power((var + self.h_param),(-1/2)))
    part2 = (s.T - mu).T
    normlz_s = np.dot(part1, part2)
    return normlz_s


  def compute_cost(self, X, Y_true):
    """
    compute the cost based on the current estimations of w and b
    """
    Y_pred, act_h,_,_,_,_,_ = self.evaluate(X, batch_norm = self.batch_norm_flag)
    num_exampl = X.shape[1]

    rglz = 0
    for i in range(self.num_hlayers + 1):
      rglz += self.labda * np.sum(self.w[i]**2)

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


  def compute_gradients(self, X_batch, y_true_batch, y_pred_batch, act_h, \
                        s, normlz_s, mu, var, batch_norm=True):   
    """
    compute the gradients of the loss, so the parameters can be updated in the direction of the steepest gradient. 
    """
    grad_b = [None]*(self.num_hlayers+1) # check if this shouldnt be one shorter
    grad_w = [None]*(self.num_hlayers+1)
    grad_gamma = [None]*(self.num_hlayers)
    grad_beta = [None]*(self.num_hlayers)
    
    # gradient of batch
    grad_batch = self.compute_gradient_batch(y_true_batch, y_pred_batch)
    
    # gradient of W and b for the last layer
    grad_w[-1] = (1/ self.batch_size) * np.dot(grad_batch, act_h[self.num_hlayers-1].T) \
                + 2*self.labda*self.w[self.num_hlayers]
    #grad_b[-1] = (1/ self.batch_size) * np.sum(grad_batch, axis=1).reshape(-1, 1)
    grad_b[-1] = (1/ self.batch_size) * np.dot(grad_batch, np.ones((self.batch_size,1)))
    
    #propagate the gradient to previous layers
    grad_batch = np.dot(self.w[-1].T, grad_batch)
    
    # this is grad_batch * ind(X_batch[-2] > 0)
    layers_input = act_h[self.num_hlayers-1]
    h_act_ind = np.zeros(layers_input.shape)
    for k in range(layers_input.shape[0]):
      for j in range(layers_input.shape[1]):
        if layers_input[k,j] > 0:
          h_act_ind[k, j] = 1
    grad_batch = grad_batch * h_act_ind
    
    # compute everything backwards from the second-last to the first layer 
    # (all the hidden layers, not the output one)
    for l in reversed(range(self.num_hlayers)):
      
      if batch_norm: 
        grad_gamma[l] = (1/self.batch_size) * np.dot((grad_batch * normlz_s[l]), np.ones(self.batch_size)).reshape(-1,1)
        grad_beta[l] = (1/self.batch_size) * np.dot(grad_batch, np.ones(self.batch_size)).reshape(-1,1)
        #propagate gradient through scale&shift and batch norm
        grad_batch = grad_batch * np.dot(self.gamma[l], np.ones((self.batch_size,1)).T)
        grad_batch = self.batch_norm_backpass(grad_batch, s[l], mu[l], var[l])
      
      # compute gradients for w and b for these layers
      if l == 0:
        grad_w[l] = (1/ self.batch_size) * np.dot(grad_batch, X_batch.T) + 2*self.labda*self.w[l]
      else:
        grad_w[l] = (1/ self.batch_size) * np.dot(grad_batch, act_h[l-1].T) + 2*self.labda*self.w[l]
      #grad_b[l] = (1/ self.batch_size) * np.sum(grad_batch, axis=1).reshape(-1, 1)
      grad_b[l] = (1/ self.batch_size) * np.dot(grad_batch, np.ones((self.batch_size,1)))
    
      if l > 0: 
        # propagate gradient backwards
        grad_batch = np.dot(self.w[l].T, grad_batch)
        # layers_input is the input to layer i (which is the activation of the previous layer)
        layers_input = act_h[l-1]
        h_act_ind = np.zeros(layers_input.shape)
        for k in range(layers_input.shape[0]):
          for j in range(layers_input.shape[1]):
            if layers_input[k,j] > 0:
              h_act_ind[k, j] = 1
        grad_batch = grad_batch * h_act_ind

    return grad_b, grad_w, grad_beta, grad_gamma


  def compute_gradient_batch(self, y_true_batch, y_pred_batch):
    """
    compute the gradient of a batch
    """
    grad_batch = - (y_true_batch - y_pred_batch)
    return grad_batch
  
  
  def batch_norm_backpass(self, grad_batch, s_batch, mu, var):
    """
    the backwards pass of batch normalization
    """
    sigma_1 = ((var + self.h_param)**(-0.5)).T
    sigma_2 = ((var + self.h_param)**(-1.5)).T
    bigG1 = grad_batch * np.outer(sigma_1, np.ones((self.batch_size,1)))
    bigG2 = grad_batch * np.outer(sigma_2, np.ones((self.batch_size,1)))
    bigD = s_batch - np.outer(mu, np.ones((self.batch_size,1)))
    c = np.dot((bigG2 * bigD), np.ones((self.batch_size,1)))
    grad_batch = bigG1 - 1/self.batch_size * np.dot(bigG1, np.ones((self.batch_size,1)))
    grad_batch -= 1 / self.batch_size * (bigD * np.outer(c, np.ones((self.batch_size))))
    return grad_batch
  
  def update_params(self, grad_b, grad_w, grad_beta, grad_gamma, batch_norm = True):
    """
    Update the parameters in the direction of the steepest gradient
    Beta and gamma are the scaling and shifting parameters of batch norm
    """
    for k in range(self.num_hlayers+1):
      self.w[k] = self.w[k] - self.lr*grad_w[k]
      self.b[k] = self.b[k] - self.lr*grad_b[k]
    if batch_norm: 
      for k in range(self.num_hlayers):
        self.beta[k] = self.beta[k] - self.lr*grad_beta[k]
        self.gamma[k] = self.gamma[k] - self.lr*grad_gamma[k]

  def cyclic_lr(self, t):
    """
    Update learning rate according to a cyclic learning rate scheme
    Learning rate increases linearly from 2*l*ns till (2*l+1)*ns and then
    decreases linearly again until 2*(l+1)*ns
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
    Y_pred_train, act_h, _, _, _, _, _ = self.evaluate(X_train, self.batch_norm_flag)
    Y_pred_val, act_h_2, _, _, _, _, _ = self.evaluate(X_val, self.batch_norm_flag)
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
    plt.savefig("loss_labda="+str(self.labda)+".png")
    plt.plot(x, self.acc_hist_tr, label = "Train accuracy")
    plt.plot(x, self.acc_hist_val, label = "Val accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0,0.8)
    plt.legend()
    plt.savefig("acc_labda=" + str(self.labda)+".png")
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
    :param h_param: a parameter needed to be set for
    the finite difference method, usually around 1e-6
    """
    grad_w = [None]*(self.num_hlayers+1)
    grad_b = [None]*(self.num_hlayers+1)

    for k in range(self.num_hlayers):
        grad_b[k] = np.zeros((self.m[k], 1))
    grad_b[self.num_hlayers] = np.zeros((self.k, 1))

    grad_w[0] = np.zeros((self.m[0], self.d))
    for k in range(1, self.num_hlayers):
      grad_w[k] = np.zeros((self.m[k], self.m[k - 1]))
    grad_w[self.num_hlayers] = np.zeros((self.k, self.m[self.num_hlayers - 1]))


    c = self.compute_cost(X, Y_true)

    for k in range(self.num_hlayers+1):
        for i in range(self.b[k].shape[0]):
          self.b[k][i] += self.h_param
          c2 = self.compute_cost(X, Y_true)
          grad_b[k][i] = (c2-c) / self.h_param
          self.b[k][i] -= self.h_param

        for i in range(self.w[k].shape[0]): #k
          for j in range(self.w[k].shape[1]): #d

            self.w[k][i,j] += self.h_param
            c2 = self.compute_cost(X, Y_true)
            grad_w[k][i,j] = (c2-c) / self.h_param
            self.w[k][i][j] -= self.h_param

    return grad_b, grad_w

  def compute_gradient_num_slow(self, X, Y_true):
    """
    Method to check the gradient calculation.
    Gradient computed numerically based on the centered difference method
    :param h_param: a parameter needed to be set for
    the centered difference method, usually around 1e-6
    """
    grad_w = [None] * (self.num_hlayers + 1)
    grad_b = [None] * (self.num_hlayers + 1)

    for k in range(self.num_hlayers):
        grad_b[k] = np.zeros((self.m[k], 1))
    grad_b[self.num_hlayers] = np.zeros((self.k, 1))

    grad_w[0] = np.zeros((self.m[0], self.d))
    for k in range(1,self.num_hlayers):
        grad_w[k] = np.zeros((self.m[k], self.m[k-1]))
    grad_w[self.num_hlayers] = np.zeros((self.k, self.m[self.num_hlayers-1]))

    for k in range(self.num_hlayers+1):
        for i in range(self.b[k].shape[0]):
          self.b[k][i] -= self.h_param
          c1 = self.compute_cost(X, Y_true)
          self.b[k][i] += 2*self.h_param
          c2 = self.compute_cost(X, Y_true)
          grad_b[k][i] = (c2 - c1) / (2*self.h_param)
          self.b[k][i] -= self.h_param

        for i in range(self.w[k].shape[0]):  # k
          for j in range(self.w[k].shape[1]):  # d
            self.w[k][i, j] -= self.h_param
            c1 = self.compute_cost(X, Y_true)
            self.w[k][i, j] += 2*self.h_param
            c2 = self.compute_cost(X, Y_true)
            grad_w[k][i, j] = (c2 - c1) / (2*self.h_param)
            self.w[k][i, j] -= self.h_param
    return grad_b, grad_w
