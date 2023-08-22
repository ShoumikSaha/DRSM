import os
import magic
from secml.array import CArray
import numpy as np
import torch
import sys
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import argparse

from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware
from secml_malware.models.basee2e import End2EndModel

from secml_malware.smoothed_malconv import get_dataset, create_smoothed_malconv, modify_dataset_for_smoothed_malconv, \
    pad_ablated_input, train_model, model_predict, get_majority_voting, get_majority_voting_without_padding, modify_dataset_for_smoothed_malconv_by_ablation

from secml_malware.custom_malconv import Custom_MalConv

from secml.settings import SECML_PYTORCH_USE_CUDA
use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA
use_mps = torch.backends.mps.is_available()


inp_len = 2**21


def main(mal_path, ben_path, dir_path, ablation_idx, dataset_size, total_ablations=4, epoch=2, batch_size=16):
    X, y, file_names, lengths = get_dataset(mal_path, ben_path, inp_len, int(dataset_size / 2))
    print(len(X))
    new_X, new_y = modify_dataset_for_smoothed_malconv_by_ablation(X, np.reshape(y, (-1)), total_ablations, ablation_idx)
    print(new_X.shape, new_y.shape)

    """
    idx = np.arange(new_X.shape[0])
    np.random.shuffle(idx)
    train_size = int(new_X.shape[0] * 0.8)
    train_idx = idx[0:train_size]
    test_idx = idx[train_size:]
    print(train_idx)
    x_train = CArray(new_X[train_idx])
    x_test = CArray(new_X[test_idx])
    y_train = CArray(new_y[train_idx])
    y_test = CArray(new_y[test_idx])
    print(y_test)
    """
    """
    x_train, x_test, y_train, y_test, lengths_train, lengths_test = train_test_split(new_X, new_y, lengths,
                                                                                    test_size=0.20,
                                                                                     random_state=1, shuffle=True)  ##Change to new_padded_X depending on the model
    """

    stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_idx, test_idx in stratSplit.split(new_X, new_y):
        x_train = new_X[train_idx]
        y_train = new_y[train_idx]
        x_test = new_X[test_idx]
        y_test = new_y[test_idx]

    print(x_train.shape, x_test.shape)
    print(y_train.shape)

    net = Custom_MalConv(max_input_size=int(inp_len / total_ablations), unfreeze=True)
    net = CClassifierEnd2EndMalware(net, batch_size=batch_size)
    net._n_features = int(inp_len / total_ablations)

    model_path = dir_path + "/" + "smoothed_malconv_" + str(total_ablations) + "_" + str(ablation_idx) + ".h5"
    if os.path.exists(model_path):
        print("Loading the model from path")
        net.load_model(model_path)
    else:
        print("Created the model")
    print("Model Name: ", model_path)
    print("Ablation index ", ablation_idx, ": ")
    net._epochs = epoch
    net.batch_size = batch_size

    print("Training the model")
    out = net.fit(CArray(x_train[:, :]), CArray(y_train[:]))
    net._model = out

    preds = net.predict(CArray(x_train[:, :]))
    print('Train Accuracy: ', accuracy_score(preds.tondarray(), y_train))

    preds = net.predict(CArray(x_test[:, :]))
    #print(preds.tondarray())
    #print(y_test)
    print('Test Accuracy: ', accuracy_score(preds.tondarray(), y_test))

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    net.save_model(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the custom model")
    parser.add_argument('--mal_path', metavar='path', required=True)
    parser.add_argument('--ben_path', metavar='path', required=True)
    parser.add_argument('--dir_path', metavar='path', required=True)
    parser.add_argument('--ablation_idx', type=int, metavar='ablation_idx', required=True)
    parser.add_argument('--dataset_size', type=int, metavar='dataset_size', required=False, default=-2)
    parser.add_argument('--ablations', type=int, metavar='total_ablations', required=True)
    parser.add_argument('--epochs', type=int, metavar='epoch', required=True)
    parser.add_argument('--batch_size', type=int, metavar='batch_size', required=False, default=16)

    args = parser.parse_args()
    main(args.mal_path, args.ben_path, args.dir_path, args.ablation_idx, args.dataset_size, args.ablations, args.epochs, args.batch_size)







