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


def main():
  data = Data("goblet_book.txt")
  book_chars = data.unique_chars
  book_ints = data.char_to_int(book_chars)
  rnn1 = RNN()
  x0 = np.random.rand(rnn1.k,1)
  h0 = np.random.rand(rnn1.m,1)
  n = 10
  generated_seq = rnn1.generate(h0, x0, n, book_chars)
  print("generated sequence: ", generated_seq)



if __name__ == "__main__":
  main()
