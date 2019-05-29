"""
Deep Learning in Data Science Assignment 4
Text synthesis using a vanilla RNN

Author: Clara Tump
Last updated: 30-05-2019

main file.
Reads in the data. Runs the RNN, and reports the results.

"""

import numpy as np
#from data import Data
#from rnn import RNN

np.random.seed(138)


def main():
  data = Data("goblet_book.txt")
  book_chars = data.unique_chars
  
  rnn1 = RNN()
  x0 = data.string_to_onehot(".")
  h0 = np.zeros((rnn1.m,1))
  seq_length = 50
  
  X, Y = get_inputs(data, seq_length)
  
  onehot_seq = rnn1.generate(h0, x0, seq_length, book_chars)
  generated_text = data.onehot_to_string(onehot_seq)
  print("generated text: ", generated_text)
  
def get_inputs(data, seq_length):
  """
  get X (input) and Y (labels) matrices for training the RNN
  """
  X_chars = data.book_data[:seq_length]
  Y_chars = data.book_data[1:seq_length+1]
  X = data.chars_to_onehot(X_chars)
  Y = data.chars_to_onehot(Y_chars)
  return X, Y
  
  
if __name__ == "__main__":
  main()
