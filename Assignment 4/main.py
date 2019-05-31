"""
Deep Learning in Data Science Assignment 4
Text synthesis using a vanilla RNN

Author: Clara Tump
Last updated: 30-05-2019

main file.
Reads in the data. Runs the RNN, and reports the results.

"""

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
#from data import Data
#from rnn import RNN

np.random.seed(110)


def main():
  data = Data("goblet_book.txt")
  book_chars = data.unique_chars
  rnn1 = RNN()
  rnn1.train(data)
  #rnn1.check_gradients(data)
 
if __name__ == "__main__":
  main()
