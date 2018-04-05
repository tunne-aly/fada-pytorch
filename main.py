from training import pretrain, train_discriminator, train
from data import sample_groups
import torch
import sys

n_target_samples = 7
plot_accuracy = True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please provide source and target data paths')
        exit()
    cuda = torch.cuda.is_available()

    groups, data, source_test_data_loader, target_test_data_loader = sample_groups(sys.argv[1], sys.argv[2], n_target_samples)

    encoder, classifier = pretrain(data, source_test_data_loader, cuda=cuda, epochs=20)

    # discriminator = train_discriminator(encoder, groups, n_target_samples=n_target_samples, epochs=50, cuda=cuda)

    # train(encoder, discriminator, classifier, data, groups, n_target_samples=n_target_samples, cuda=cuda, epochs=150, plot_accuracy=plot_accuracy)
