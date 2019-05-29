"""
Deep Learning in Data Science Assignment 4
Text synthesis using a vanilla RNN

Author: Clara Tump
Last updated: 30-95-2019


"""


import numpy as np


def main():
  book_data = read_data()
  book_chars = get_unique_chars(book_data)
  char_to_int_dict, int_to_char_dict = create_dictionaries(book_chars)
  book_ints = char_to_int(book_chars, char_to_int_dict)
  print(book_ints)
  book_chars = int_to_char(book_ints, int_to_char_dict)
  print(book_chars)

def read_data():
  """
  reads the txt file and returns the content as a string
  """
  data_string = open('goblet_book.txt','r').read()
  return data_string

def get_unique_chars(data_string):
  """
  returns a numpy array of the unique characters in the string of text
  """
  unique_chars = list(set(data_string))
  return unique_chars

def create_dictionaries(chars):
  """
  create dictionaries which map characters to their corresponding ints and ints
  back to their corresponding characters
  """
  dictionary = dict()
  for char in chars:
      dictionary[char] = len(dictionary)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return dictionary, reverse_dictionary


def char_to_int(char_list, char_to_int_dict):
  """
  convert a list of characters to the corresponding list of integers
  """
  int_list = [char_to_int_dict[char] for char in char_list]
  return int_list

def int_to_char(int_list, int_to_char_dict):
  """
  convert a list of integers to the corresponding list of characters
  """
  char_list = [int_to_char_dict[int] for int in int_list]
  return char_list

if __name__ == "__main__":
  main()
