"""
Deep Learning in Data Science Assignment 4
Text synthesis using a vanilla RNN

Author: Clara Tump
Last updated: 30-05-2019

main file.
Reads in the data. Runs the RNN, and reports the results.


"""


import numpy as np
from data import Data
from rnn import RNN


def main():
  data = Data("goblet_book.txt")
  book_chars = data.unique_chars
  book_ints = data.char_to_int(book_chars)
  print(book_ints)
  book_chars = data.int_to_char(book_ints)
  print(book_chars)



if __name__ == "__main__":
  main()
