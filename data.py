import torchvision
import torch
import numpy as np
import gzip
import random
from gensim.models import Word2Vec, KeyedVectors
import re, csv
from os.path import isfile
from os import listdir, sep
from torch.utils.data import DataLoader
from indexed_review_dataset import IndexedReviewDataset
from amazon_dataset import AmazonDataset, get_fasttext_embeddings

''' Returns the MNIST dataloader '''
def mnist_dataloader(batch_size=256, train=True, cuda=False):
    dataset = torchvision.datasets.MNIST('./data', download=True, train=train, transform=torchvision.transforms.ToTensor())
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=cuda)

''' Returns the SVHN dataloader '''
def svhn_dataloader(batch_size=256, train=True, cuda=False):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.SVHN('./data', download=True, split=('train' if train else 'test'), transform=transform)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=cuda)

''' Reads and cleans Amazon sentences '''
def read_sentences(file_path):
  print('reading sentences from {}'.format(file_path))
  g = gzip.open(file_path, 'r')
  i = 0
  for l in g:
    dicti =  eval(l)
    grade = dicti['overall']
    line = dicti['reviewText']
    for para in line.splitlines():
      if i >= 50000: break
      if para:
          i += 1
          yield [re.sub("[^a-öA-Ö ']", '', l).strip().lower() for l in para.split()], grade

''' Reads Amazon review data'''
def get_amazon_reviews_from_zip(data_path):
    reviews = []
    for p in listdir(data_path):
        if p.endswith(".gz"):
            for sent in read_sentences("{}{}{}".format(data_path, sep, p)):
              reviews.append(sent)
    return reviews

''' Reads Amazon review data without loading memory'''
def yield_amazon_reviews_from_zip(data_path):
    for p in listdir(data_path):
        if p.endswith(".gz"):
            for sent in read_sentences("{}{}{}".format(data_path, sep, p)):
                yield sent

''' Get embedding for a specific word '''
def get_embedding(idx, sentence, embeddings, vec_size):
    return (torch.from_numpy(embeddings[sentence[idx]])
            if len(sentence) > idx and sentence[idx] in embeddings
            else torch.zeros(vec_size))

''' Create sentence matrix of a fixed length by stacking word vectors '''
def get_sentence_tensor(sentence, embeddings, block_length, vec_size):
    return torch.stack([torch.stack([get_embedding(i, sentence, embeddings, vec_size) for i in range(block_length)])])

''' Return word2vec embeddings for a set of reviews '''
def get_w2v_embeddings(reviews, embeddings_path, vec_size):
    embeddings_path = "w2v_" + embeddings_path
    if isfile(embeddings_path):
        print('Retrieving embeddings from {}'.format(embeddings_path))
        embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True, unicode_errors='ignore')
    else:
        print('Building new word2vec embeddings')
        model = Word2Vec([r[0] for r in reviews], size=300, window=10, workers=2)
        print('Done')
        embeddings = model.wv
        del model
        embeddings.save_word2vec_format(embeddings_path, binary=True)
        print("Embeddings stored to {}".format(embeddings_path))
    return embeddings

''' Samples a subset from source into memory '''
def sample_mnist_source_data(n=2000):
    dataset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=torchvision.transforms.ToTensor())
    X = torch.FloatTensor(n, 1, 28, 28)
    Y = torch.LongTensor(n)

    train_inds = torch.randperm(len(dataset))[:n]
    for i, index in enumerate(train_inds):
        x, y = dataset[index]
        X[i] = x
        Y[i] = y
    return X, Y

def get_sentence_indexes(sentence, vocabulary, block_length):
    return [vocabulary[word] for i, word in enumerate(sentence) if i < block_length]

def sample_nlp_source_data(vec_size=300, n=2000):
    # use indexes to save memory
    dataset = AmazonDataset(as_indexes=True)
    X = torch.FloatTensor(n, 1, dataset.sentence_length, dataset.vec_size)
    Y = torch.LongTensor(n)

    index_perm = torch.randperm(len(dataset))
    train_inds = index_perm[:n]
    for i, index in enumerate(train_inds):
        x, y = dataset[index]
        X[i] = x
        Y[i] = y
    return X, Y

# def sample_nlp_source_data_with_trainable_embeddings(
#     data_path,
#     vec_size=300,
#     n=2000,
#     indexed_sentences_train_path='data/amazon_indexed_sentence_vectors_train.txt'):

#     # the size of a sentence
#     # make sure sentence_length and vec_size are the same as given to fasttext
#     sentence_length = 30
#     X = torch.FloatTensor(n, sentence_length)
#     Y = torch.LongTensor(n)

#     train_dataset = IndexedReviewDataset(indexed_sentences_train_path)
#     train_inds = random.sample(range(0, len(train_dataset)), n)
#     for i, index in enumerate(train_inds):
#         x, y = train_dataset[index]
#         X[i] = x
#         Y[i] = y
#     return X, Y

# def get_source_data_for_pretrain_from_index_files(
#     embeddings_path='data/amazon_word_embeddings.txt',
#     no_classes_path='data/amazon_data_no_classes_for_whole_FADA_dataset.txt',
#     indexed_sentences_train_path='data/amazon_indexed_sentence_vectors_train.txt',
#     indexed_sentences_test_path='data/amazon_indexed_sentence_vectors_test.txt'):
#     print('Getting embeddings from {}'.format(embeddings_path))
#     print('Getting indexed source training data from {}'.format(indexed_sentences_train_path))
#     print('Getting indexed source test data from {}'.format(indexed_sentences_test_path))

#     embeddings = get_fasttext_embeddings(embeddings_path)
#     train_dataset = IndexedReviewDataset(indexed_sentences_train_path)
#     test_dataset = IndexedReviewDataset(indexed_sentences_test_path)

#     with open(no_classes_path, 'r') as info_file:
#         no_classes = eval(info_file.read())
#     print('no classes: {}'.format(no_classes))

#     train_data_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
#     test_data_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=4)
#     return train_data_loader, test_data_loader, no_classes, embeddings

def get_source_data_for_pretrain(no_classes_path = "asdf"):
    dataset = AmazonDataset()

    with open(no_classes_path, 'r') as info_file:
        no_classes = eval(info_file.read())
    print('no classes: {}'.format(no_classes))

    dataset_size = len(dataset)
    index_perm = torch.randperm(dataset_size)
    train_set_size = 0.9 * dataset_size
    train_inds = index_perm[:train_set_size]
    test_inds = index_perm[train_set_size:]

    no_classes = {}

    train_data_loader = DataLoader(dataset[train_inds], batch_size=100, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset[test_inds], batch_size=100, shuffle=True, num_workers=4)
    return train_data_loader, test_data_loader, no_classes

def get_finnish_reviews(data_path):
    ls = []
    for p in listdir(data_path):
        if p.endswith(".csv"):
            for sent in read_finnish_reviews_from_file("{}{}{}".format(data_path, sep, p)):
                ls.append(sent)
    return ls

def read_finnish_reviews_from_file(file_path):
    with open(file_path) as infile:
        reader = csv.reader(infile)
        for line in reader:
          yield [re.sub("[^a-öA-Ö ']", '', w).strip().lower() for w in line[1]], int(line[0])-1

''' Returns a subset of the target domain such that it has n_target_samples per class '''
def create_svhn_target_samples(n=1):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.SVHN('./data', download=True, split='train', transform=transform)
    X, Y = [], []
    classes = 10 * [n]

    i = 0
    while True:
        if len(X) == n*10:
            break
        x, y = dataset[i]
        if classes[y] > 0:
            X.append(x)
            Y.append(y)
            classes[y] -= 1
        i += 1

    assert(len(X) == n*10)
    return torch.stack(X), torch.from_numpy(np.array(Y))

def create_nlp_target_samples(data_path, n=1, no_classes=5, vec_size=300, embeddings_path='finnish_embeddings.txt'):
    # there are less finnish data points so we courageously load them to memory
    reviews = get_finnish_reviews(data_path)
    embeddings = get_fasttext_embeddings(embeddings_path)
    block_length = 30
    dataset = [(get_sentence_tensor(r[0], embeddings, block_length, vec_size), r[1]) for r in reviews]
    X, Y = [], []
    classes = no_classes * [n]
    i = 0
    while True:
        if len(X) == n*no_classes:
            break
        x, y = dataset[i]
        if classes[y] > 0:
            X.append(x)
            Y.append(y)
            classes[y] -= 1
        i += 1
    assert(len(X) == n*no_classes)
    test_inds = torch.randperm(len(dataset))[:100]
    test_data_loader = DataLoader([dataset[index] for index in test_inds], batch_size=100, shuffle=False)
    return torch.stack(X), torch.from_numpy(np.array(Y)), test_data_loader

'''
    Samples uniformly groups G1 and G3 from D_s x D_s and groups G2 and G4 from D_s x D_t
'''
def create_groups(X_s, y_s, X_t, y_t):
    n = X_t.shape[0]
    G1, G3 = [], []

    # TODO optimize
    # Groups G1 and G3 come from the source domain
    for i, (x1, y1) in enumerate(zip(X_s, y_s)):
        for j, (x2, y2) in enumerate(zip(X_s, y_s)):
            if y1 == y2 and i != j and len(G1) < n:
                G1.append((x1, x2))
            if y1 != y2 and i != j and len(G3) < n:
                G3.append((x1, x2))

    G2, G4 = [], []

    # Groups G2 and G4 are mixed from the source and target domains
    for i, (x1, y1) in enumerate(zip(X_s, y_s)):
        for j, (x2, y2) in enumerate(zip(X_t, y_t)):
            if y1 == y2 and i != j and len(G2) < n:
                G2.append((x1, x2))
            if y1 != y2 and i != j and len(G4) < n:
                G4.append((x1, x2))

    groups = [G1, G2, G3, G4]

    # Make sure we sampled enough samples
    for g in groups:
        assert(len(g) == n)
    return groups

'''
    Sample groups G1, G2, G3, G4. The DCD is trained with these.
    n_target_samples is the amount of labeled target data, and also the size of the groups
'''
def sample_groups(source_data_path, target_data_path, n_target_samples):
    X_s, y_s = sample_nlp_source_data()
    print('Source data samples ready')
    exit()
    #X_t, y_t, target_test_data_loader = create_nlp_target_samples(target_data_path, n_target_samples)

    print("Sampling groups")
    return create_groups(X_s, y_s, X_t, y_t), (X_s, y_s, X_t, y_t), target_test_data_loader
