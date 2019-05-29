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
  book_ints = data.char_to_int(book_chars)
  
  rnn1 = RNN()
  x0 = data.string_to_onehot(".")
  h0 = np.random.rand(rnn1.m,1)
  n = 100
  
  generated_seq = rnn1.generate(h0, x0, n, book_chars)
  generated_text = data.onehot_to_string(generated_seq)
  print("generated text: ", generated_text)
  

if __name__ == "__main__":
  main()
