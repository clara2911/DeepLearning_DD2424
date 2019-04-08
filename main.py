import numpy as np
import pickle
print("imported pickle")
from ann import ANN
import matplotlib.pyplot as plt

def main():
  np.random.seed(12)
  # y = N, // X = dxN // Y=KxN
  X_train, y_train , Y_train = load_data(batch_file = "data/data_batch_1", num=500, feat_num=3072)
  print("loaded train data")
  #X_test, y_test, Y_test = load_data(batch_file = "data/test_batch")
  print("loaded test batch")
  #X_val, y_val , Y_val = load_data(batch_file = "data/data_batch_2", num=3)
  print("loaded all data sets")
  ann1 = ANN(X_train, Y_train)
  #print("initialized ann")
  X_val, Y_val = X_train, Y_train
  #print("Y_trains shape ", Y_train.shape)
  #print(" X TRAIN")
  #print(X_train)
  #print("Y_train")
  #print(Y_train)
  ann1.train(X_train, Y_train, X_val, Y_val)
  print("done training")

def load_data(batch_file = "data/data_batch_1", num=100, feat_num=3):
  with open(batch_file, 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
    X = preprocess(data[b"data"])
    y = np.array(data[b"labels"])
    Y = one_hot(y)
    print(X.shape)
  if num:
    return X[:num, :feat_num].T, y[:num], Y[:num].T
  else:
    return X, y, Y

def preprocess(chunk):
  """
  Preprocess the data:
  scale from pixel values to values between 0 and 1 and convert to grayscale
  """
  #r=0.3
  #g=0.59
  #b = 0.11
  # convert to grayscale
  #chunk = dataset.reshape(dataset.shape[0], 1024, 3)
  #chunk = r * chunk[:, :, 0] + g * chunk[:, :,1] + b * chunk[:, :, 2]
  # scaling the values
  chunk = chunk / 255.0
  return chunk

def one_hot(y):
  Y = np.zeros((y.shape[0], 10))
  Y[np.arange(y.shape[0]), y] = 1
  
  return Y
 

if __name__ == "__main__":
  main()
