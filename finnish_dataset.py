import torch
import numpy as np
from torch.utils.data import Dataset
import subprocess, linecache
import nltk
import re
import csv
from os.path import isfile

def get_fasttext_embeddings(embeddings_path):
  # two ways of storing embeddings:
  # a tensor where index maps to a vector
  # a dict which maps word to an index
  word_to_index = {}
  # vectors are 300 dimensional
  # amount counted with "wc -l finnish_electronics_word_embeddings.txt"
  vectors = torch.zeros((0, 300))
  if not isfile(embeddings_path):
      print('Please run fast text vectors first:')
      print('fastText/fasttext skipgram -input finnish_fast_text_sentences -output finnish_model -dim 300')
      print('fastText/fasttext print-word-vectors finnish_model.bin < finnish_fast_text_words > {}.txt'.format(embeddings_path))
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

class FinnishDataset(Dataset):
    """Finnish reviews dataset."""

    def __init__(self, as_indexes=False):
        """
        Args:
            the location of reviews and embeddings is defined here
        """
        # wc -l finnish_dataset.txt
        self.data_len = 16154
        # the neuran netowork's input has to be of consistent size, so the sentences are cut
        self.sentence_length = 100
        self.review_file = ""
        if not self.review_file.endswith('.txt'):
          print('Make sure that the finnish dataset is given as a .txt file containing tuples')
        self.embeddings, self.word_to_index = get_fasttext_embeddings("data/finnish_electronics_word_embeddings.txt")
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
        index_vector = [self.word_to_index[word.lower()] for word in nltk.word_tokenize(review) if re.search('[a-öA-Ö]', word)]
        return (index_vector, grade)
      else:
        # tokenize and filter out punctuation
        # use word_to_index to get the right embedding
        sentence = [self.embeddings[self.word_to_index[word.lower()]] for word in nltk.word_tokenize(review) if re.search('[a-öA-Ö]', word)]
        return (sentence, grade)
