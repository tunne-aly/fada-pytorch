import torch
import torch.nn.functional as F
from torch import nn

'''
    Domain-Class Discriminator (see (3) in the paper)
    Takes in the concatenated latent representation of two samples from
    G1, G2, G3 or G4, and outputs a class label, one of [0, 1, 2, 3]
'''
class DCD(nn.Module):
    def __init__(self, H=100, D_in=200):
        super(DCD, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, H)
        self.out = nn.Linear(H, 4)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.softmax(self.out(out), dim=1)

''' Called h in the paper. Gives class predictions based on the latent representation '''
class Classifier(nn.Module):
    def __init__(self, D_in=100):
        super(Classifier, self).__init__()
        self.out = nn.Linear(D_in, 5)

    def forward(self, x):
        out = self.out(x)
        return out

'''
    Creates latent representation based on data. Called g in the paper.
    Like in the paper, we use g_s = g_t = g, that is, we share weights between target
    and source representations.

    Model is as specified in section 4.1. See https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
'''
class Encoder(nn.Module):
    def __init__(self, weights):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(len(weights), 300)
        self.embeddings.weight.data.copy_(weights)
        self.embeddings.weight.requires_grad = False
        self.fc1   = nn.Linear(9000, 3000)
        self.fc2   = nn.Linear(3000, 1000)
        self.fc3   = nn.Linear(1000, 100)

    def forward(self, x):
        # embeddings is the first layer
        # use convolution
        out = self.embeddings(x).view((x.size(0), -1))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
