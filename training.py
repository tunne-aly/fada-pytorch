import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from data import mnist_dataloader, svhn_dataloader
from models import Classifier, Encoder, DCD
from util import eval_on_test, into_tensor
import random
import matplotlib.pyplot as plt

def model_fn(encoder, classifier):
    return lambda x: classifier(encoder(x))

''' Pretrain the encoder and classifier as in (a) in figure 2. '''
def pretrain(source_data_loader, test_data_loader, no_classes, embeddings, epochs=20, batch_size=128, cuda=False):

    classifier = Classifier()
    encoder = Encoder(embeddings)

    if cuda:
        classifier.cuda()
        encoder.cuda()

    ''' Jointly optimize both encoder and classifier '''
    encoder_params = filter(lambda p: p.requires_grad, encoder.parameters())
    optimizer = optim.Adam(list(encoder_params) + list(classifier.parameters()))

    # Use weights to normalize imbalanced in data
    c = [1]*len(no_classes)
    weights = torch.FloatTensor(len(no_classes))
    for i, (a, b) in enumerate(zip(c, no_classes)):
        weights[i] = 0 if b == 0 else a/b

    loss_fn = nn.CrossEntropyLoss(weight=Variable(weights))

    print('Training encoder and classifier')
    for e in range(epochs):

        # pretrain with whole source data -- use groups with DCD
        for sample in source_data_loader:
            x, y = Variable(sample[0]), Variable(sample[1])
            optimizer.zero_grad()

            if cuda:
                x, y = x.cuda(), y.cuda()

            output = model_fn(encoder, classifier)(x)

            loss = loss_fn(output, y)

            loss.backward()

            optimizer.step()

        print("Epoch", e, "Loss", loss.data[0], "Accuracy", eval_on_test(test_data_loader, model_fn(encoder, classifier)))

    return encoder, classifier

''' Train the discriminator while the encoder is frozen '''
def train_discriminator(encoder, groups, n_target_samples=2, cuda=False, epochs=20):

    discriminator = DCD(D_in=200) # Takes in concatenated hidden representations
    loss_fn = nn.CrossEntropyLoss()

    # Only train DCD
    optimizer = optim.Adam(discriminator.parameters())

    # Size of group G2, the smallest one, times the amount of groups
    n_iters = 4 * n_target_samples

    if cuda:
        discriminator.cuda()

    print("Training DCD with classifier and encoder frozen")
    for e in range(epochs):

        for _ in range(n_iters):

            # Sample a pair of samples from a group
            group = random.choice([0, 1, 2, 3])

            x1, x2 = groups[group][random.randint(0, len(groups[group]) - 1)]
            x1, x2 = Variable(x1), Variable(x2)

            if cuda:
                x1, x2 = x1.cuda(), x2.cuda()

            # Optimize the DCD using sample drawn
            optimizer.zero_grad()

            # Concatenate encoded representations
            x_cat = torch.cat([encoder(x1.unsqueeze(0)), encoder(x2.unsqueeze(0))], 1)
            output = discriminator(x_cat)

            _, y_pred = torch.max(output.data, 1)

            print('pred: {} '.format(y_pred))

            # Label is the group
            y = Variable(torch.LongTensor([group]))
            print('true: {}'.format(y))
            if cuda:
                y = y.cuda()
            loss = -loss_fn(output, y)

            loss.backward()

            optimizer.step()

        print("Epoch", e, "Loss", loss.data[0])

    return discriminator

''' FADA Loss, as given by (4) in the paper. The minus sign is shifted because it seems to be wrong '''
def fada_loss(y_pred_g2, g1_true, y_pred_g4, g3_true, gamma=0.2):
    return -gamma * torch.mean(g1_true * torch.log(y_pred_g2) + g3_true * torch.log(y_pred_g4))

''' Step three of the algorithm, train everything except the DCD '''
def train(encoder, discriminator, classifier, data, test_data_loader, groups, n_target_samples=2, cuda=False, epochs=20, batch_size=256, plot_accuracy=False):

    X_s, Y_s, X_t, Y_t = data

    G1, G2, G3, G4 = groups

    ''' Two optimizers, one for DCD (which is frozen) and one for class training '''
    class_optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()))
    dcd_optimizer = optim.Adam(encoder.parameters())

    loss_fn = nn.CrossEntropyLoss()
    n_iters = 4 * n_target_samples

    if plot_accuracy:
        accuracies = []
    print('Training encoder and classifier again and freeze DCD')
    for e in range(epochs):

        # Shuffle data at each epoch
        inds = torch.randperm(X_s.shape[0])
        X_s, Y_s = X_s[inds], Y_s[inds]

        inds = torch.randperm(X_t.shape[0])
        X_t, Y_t = X_t[inds], Y_t[inds]

        g2_one, g2_two = into_tensor(G2, into_vars=True)
        g4_one, g4_two = into_tensor(G4, into_vars=True)

        inds = torch.randperm(g2_one.shape[0])
        if cuda:
            inds = inds.cuda()
        g2_one, g2_two, g4_one, g4_two = g2_one[inds], g2_two[inds], g4_one[inds], g4_two[inds]

        for _ in range(n_iters):

            class_optimizer.zero_grad()
            dcd_optimizer.zero_grad()

            # Evaluate source predictions
            inds = torch.randperm(X_s.shape[0])[:batch_size]
            x_s, y_s = Variable(X_s[inds]), Variable(Y_s[inds])
            if cuda:
                x_s, y_s = x_s.cuda(), y_s.cuda()
            y_pred_s = model_fn(encoder, classifier)(x_s)

            # Evaluate target predictions
            ind = random.randint(0, X_t.shape[0] - 1)
            x_t, y_t = Variable(X_t[ind].unsqueeze(0)), Variable(torch.LongTensor([Y_t[ind]]))
            if cuda:
                x_t, y_t = x_t.cuda(), y_t.cuda()

            y_pred_t = model_fn(encoder, classifier)(x_t)

            # Evaluate groups

            x1, x2 = encoder(g2_one), encoder(g2_two)
            y_pred_g2 = discriminator(torch.cat([x1, x2], 1))
            g1_true = 1

            x1, x2 = encoder(g4_one), encoder(g4_two)
            y_pred_g4 = discriminator(torch.cat([x1, x2], 1))
            g3_true = 3

            # Evaluate loss
            # This is the full loss given by (5) in the paper
            loss = fada_loss(y_pred_g2, g1_true, y_pred_g4, g3_true) + loss_fn(y_pred_s, y_s) + loss_fn(y_pred_t, y_t)

            loss.backward()

            class_optimizer.step()
        acc = eval_on_test(test_data_loader, model_fn(encoder, classifier))
        print("Epoch", e, "Loss", loss.data[0], "Accuracy", acc)

        if plot_accuracy:
            accuracies.append(acc)

    if plot_accuracy:
        plt.plot(range(len(accuracies)), accuracies)
        plt.title("Finnish test accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()
