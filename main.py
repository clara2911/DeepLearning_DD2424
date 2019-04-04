import numpy as np
print("imported numpy")
import pickle
print("imported pickle")
from ann import ANN
print("imported ann")

def main():
  # y = N, // X = dxN // Y=KxN
  X_train, y_train , Y_train= load_data(batch_file = "data/data_batch_1")
  print("loaded train data")
  print("X_train: ", X_train.shape)
  print("Y_train: ", Y_train.shape)
  X_val, y_val , Y_val = load_data(batch_file = "data/data_batch_2")
  X_test, y_test , Y_test = load_data(batch_file = "data/test_batch")
  print("loaded all data sets")
  ann1 = ANN(X_train, Y_train)
  print("initialized ann")
  #not tested yet
  ann1.train(X_train, Y_train)
  print("done training")


def load_data(batch_file = "data/data_batch_1", num=7):
  with open(batch_file, 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
    X = preprocess(data[b"data"])
    y = np.array(data[b"labels"])
    Y = one_hot(y)
    print(X.shape)
  return X[:num].T, y[:num], Y[:num].T

def preprocess(dataset):
  """
  Preprocess the data:
  scale from pixel values to values between 0 and 1 and convert to grayscale
  """
  r=0.3
  g=0.59
  b = 0.11
  # convert to grayscale
  chunk = dataset.reshape(dataset.shape[0], 1024, 3)
  chunk = r * chunk[:, :, 0] + g * chunk[:, :,1] + b * chunk[:, :, 2]
  # scaling the values
  chunk = chunk / 255.0
  print("rst shape", chunk.shape)
  return chunk

def one_hot(y):
  Y = np.zeros((y.shape[0], 10))
  Y[np.arange(y.shape[0]), y] = 1
  return Y
 

if __name__ == "__main__":
  main()
