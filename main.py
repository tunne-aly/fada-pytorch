from training import pretrain, train_discriminator, train
from data import sample_groups, get_source_data_for_pretrain
import torch
import sys

n_target_samples = 7
plot_accuracy = True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please provide source and target data paths')
        exit()
    cuda = torch.cuda.is_available()

    groups, data, target_test_data_loader = sample_groups(sys.argv[1], sys.argv[2], n_target_samples)

    # in pretrain the encoder is trained with the whole available source data
    source_train_data_loader, source_test_data_loader, no_source_classes = get_source_data_for_pretrain()

    # encoder, classifier = pretrain(source_train_data_loader, source_test_data_loader, no_source_classes, source_embeddings, cuda=cuda)

    # discriminator = train_discriminator(encoder, groups, n_target_samples=n_target_samples, epochs=50, cuda=cuda)

    # train(encoder, discriminator, classifier, data, target_test_data_loader, groups, n_target_samples=n_target_samples, cuda=cuda, epochs=150, plot_accuracy=plot_accuracy)
