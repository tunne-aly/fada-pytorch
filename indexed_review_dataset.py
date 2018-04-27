import torch
import numpy as np
from torch.utils.data import Dataset
import subprocess, linecache

class IndexedReviewDataset(Dataset):
    """Amazon and finnish reviews datasets."""

    def __init__(self, indexed_sentences_file):
        """
        Args:
            indexed_sentences_file (string): Path to the file generated with the "reviews_to_sentences_with_indexes" script. Contains the reviews changed to index vectors.
        """
        self.file = indexed_sentences_file

    def __len__(self):
      line = subprocess.check_output(['tail', '-1', self.file])
      try:
        length = int(line)
      except:
        length = 0
      return length

    def __getitem__(self, idx):
      # indexing starts from 1
      line = linecache.getline(self.file, idx + 1).strip()
      vector = eval(line)
      grade = vector[1]
      return (torch.from_numpy(np.array(vector[0])), grade)
