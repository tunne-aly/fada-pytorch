import torch
import numpy as np
from torch.utils.data import Dataset
import subprocess, linecache
import nltk
import re
import csv
import torch
from os.path import isfile

def get_fasttext_embeddings(embeddings_path):
  # two ways of storing embeddings:
  # a tensor where index maps to a vector
  # a dict which maps word to an index
  word_to_index = {}
  # vectors are 300 dimensional
  # amount counted with "wc -l amazon_electronics_word_embeddings.txt"
  vectors = torch.zeros((2998432, 300))
  if not isfile(embeddings_path):
      print('Please run fast text vectors first:')
      print('fastText/fasttext skipgram -input amazon_fast_text_sentences -output amazon_model -dim 300')
      print('fastText/fasttext print-word-vectors amazon_model.bin < amazon_fast_text_words > {}.txt'.format(embeddings_path))
      exit()
  with open(embeddings_path) as in_file:
      rd = csv.reader(in_file, delimiter=" ", quotechar="¤")
      i = 0
      for line in rd:
          if len(line) > 2:
              vector = np.asarray(line[1:-1], dtype=np.float32)
              vectors[i] = torch.from_numpy(vector)
              word_to_index[line[0]] = i
              i += 1
  return vectors, word_to_index

class AmazonDataset(Dataset):
    """Amazon and finnish reviews datasets."""

    def __init__(self, as_indexes=False):
        """
        Args:
            the location of reviews and embeddings is defined here
        """
        # from http://jmcauley.ucsd.edu/data/amazon/
        self.data_len = 1689188
        # the neuran netowork's input has to be of consistent size, so the sentences are cut
        self.sentence_length = 100
        self.review_file = "data/Electronics_5.json"
        if not self.review_file.endswith('.json'):
          print('Make sure that the amazon dataset is given as a .json file')
        self.embeddings, self.word_to_index = get_fasttext_embeddings("data/amazon_electronics_word_embeddings.txt")
        self.vec_size = len(self.embeddings[0])
        self.as_indexes = as_indexes

    def __len__(self):
      return self.data_len

    def indexes_to_embeddings(self, index_vector):
      return [self.embeddings[i] for i in index_vector]

    def __getitem__(self, idx):
      # indexing starts from 1 with linecache
      line = linecache.getline(self.review_file, idx + 1).strip()
      review_dict = eval(line)
      review = review_dict['reviewText']
      grade = int(review_dict['overall']) - 1
      if self.as_indexes:
        index_vector = torch.zeros((1, self.sentence_length, 1))
        for i, word in enumerate(nltk.word_tokenize(review)):
          if  i == self.sentence_length: break
          if word.lower() in self.word_to_index:
            index_vector[0][i] = self.word_to_index[word.lower()]
        assert len(index_vector) == self.sentence_length
        return (index_vector, grade)
      else:
        # tokenize and filter out punctuation
        # use word_to_index to get the right embedding
        sentence_vector = torch.zeros((1, self.sentence_length, self.vec_size))
        for i, word in enumerate(nltk.word_tokenize(review)):
          if  i == self.sentence_length: break
          if word.lower() in self.word_to_index:
            index_vector[0][i] = self.embeddings[self.word_to_index[word.lower()]]
        assert len(sentence_vector) == self.sentence_length
        return (sentence_vector, grade)
