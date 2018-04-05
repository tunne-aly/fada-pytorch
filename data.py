import torchvision
import torch
import numpy as np
import gzip
import re, csv
from gensim.models import Word2Vec, KeyedVectors
from os.path import isfile
from os import listdir, sep
from torch.utils.data import DataLoader

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

''' Handles Amazon review data'''
def get_amazon_reviews_from_zip(path):
  g = gzip.open(path, 'r')
  for l in g:
    dicti =  eval(l)
    yield ([re.sub("[^a-zA-Z ']", '', w).strip().lower() for w in dicti['reviewText'].split()], int(dicti['overall'])-1)

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
    if isfile(embeddings_path):
        print('Retreiving embeddings from {}'.format(embeddings_path))
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

def sample_nlp_source_data(data_path, vec_size=300, n=2000, embeddings_path='amazon_embeddings.bin'):
    reviews = get_amazon_reviews_from_zip(data_path)
    embeddings = get_w2v_embeddings(reviews, embeddings_path, vec_size)
    # TODO add one with gensim, other with fast text
    block_length = 30
    dataset = [(get_sentence_tensor(r[0], embeddings, block_length, vec_size), r[1]) for r in reviews]
    X = torch.FloatTensor(n, 1, block_length, vec_size)
    Y = torch.LongTensor(n)

    index_perm = torch.randperm(len(dataset))
    train_inds = index_perm[:n]
    test_inds = index_perm[n:n+100]
    for i, index in enumerate(train_inds):
        x, y = dataset[index]
        X[i] = x
        Y[i] = y
    test_data_loader = DataLoader([dataset[index] for index in test_inds], batch_size=100, shuffle=False)
    return X, Y, test_data_loader

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

def create_nlp_target_samples(data_path, n=1, no_classes=5, vec_size=300, embeddings_path='finnish_embeddings.bin'):
    reviews = get_finnish_reviews(data_path)
    embeddings = get_w2v_embeddings(reviews, embeddings_path, vec_size)
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

''' Sample groups G1, G2, G3, G4 '''
def sample_groups(source_data_path, target_data_path, n_target_samples):
    X_s, y_s, source_test_data_loader = sample_nlp_source_data(source_data_path)
    X_t, y_t, target_test_data_loader = create_nlp_target_samples(target_data_path, n_target_samples)

    print("Sampling groups")
    return create_groups(X_s, y_s, X_t, y_t), (X_s, y_s, X_t, y_t), source_test_data_loader,target_test_data_loader
