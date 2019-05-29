"""
Deep Learning in Data Science Assignment 4
Text synthesis using a vanilla RNN

Author: Clara Tump
Last updated: 30-05-2019

Data class.
Handles the reading of the data and the conversion between chars and ints.


"""

class Data:

    def __init__(self, txtfile):
        self.data_string = self._read_data(txtfile)
        self.unique_chars = self._get_unique_chars(self.data_string)
        self.char_to_int_dict, self.int_to_char_dict = self._create_dictionaries(self.unique_chars)


    def _read_data(self, txtfile):
      """
      reads the txt file and returns the content as a string
      """
      data_string = open(txtfile,'r').read()
      return data_string

    def _get_unique_chars(self, data_string):
      """
      returns a numpy array of the unique characters in the string of text
      """
      unique_chars = list(set(data_string))
      return unique_chars

    def _create_dictionaries(self, chars):
      """
      create dictionaries which map characters to their corresponding ints and ints
      back to their corresponding characters
      """
      dictionary = dict()
      for char in chars:
          dictionary[char] = len(dictionary)
      reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
      return dictionary, reverse_dictionary


    def char_to_int(self, char_list):
      """
      convert a list of characters to the corresponding list of integers
      """
      int_list = [self.char_to_int_dict[char] for char in char_list]
      return int_list

    def int_to_char(self, int_list):
      """
      convert a list of integers to the corresponding list of characters
      """
      char_list = [self.int_to_char_dict[int] for int in int_list]
      return char_list
