import os
import magic
from secml.array import CArray
import numpy as np
import torch
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
#import pickle
import dill as pickle

# from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware
from secml_malware.models.basee2e import End2EndModel

from secml_malware.smoothed_malconv import get_dataset, create_smoothed_malconv, modify_dataset_for_smoothed_malconv, \
    pad_ablated_input, train_model, model_predict, get_majority_voting, get_majority_voting_without_padding

from secml_malware.custom_malconv import Custom_MalConv

from secml.settings import SECML_PYTORCH_USE_CUDA
use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA
use_mps = torch.backends.mps.is_available()

inp_len = 2 ** 21


def main(mal_path, ben_path, save_path, dataset_size, total_ablations=4, epoch=2, batch_size=16):
    X, y, file_names, lengths = get_dataset(mal_path, ben_path, inp_len, int(dataset_size/2))
    #print(y)
    new_X, new_y = modify_dataset_for_smoothed_malconv(X, np.reshape(y, (-1)), total_ablations)
    print(new_X.shape, new_y.shape)

    """
    new_padded_X = []
    #ablation_idx = 2
    for i in range(new_X.shape[0]):
        temp_X = pad_ablated_input(new_X[i, ablation_idx, :], ablation_idx)
        new_padded_X.append(temp_X)
    new_padded_X = np.asarray(new_padded_X)
    print(new_padded_X.shape)
    """

    x_train, x_test, y_train, y_test, lengths_train, lengths_test = train_test_split(new_X, new_y, lengths, test_size=0.20,
                                                        random_state=1)  ##Change to new_padded_X depending on the model
    print(x_train.shape, x_test.shape)
    print(y_train.shape)
    #print(lengths_test)

    nets = []

    if os.path.exists(save_path):
        print("Loading the model from path")
        with open(save_path, "rb") as file:
            nets = pickle.load(file)
    else:
        print("Creating the model")
        for i in range(total_ablations):
            # Loading the custom MalConv model using the pre-trained one
            net = Custom_MalConv(max_input_size=int(inp_len / total_ablations), unfreeze=True)
            net = CClassifierEnd2EndMalware(net, batch_size=batch_size)
            net._n_features = int(inp_len / total_ablations)
            nets.append(net)

    for ablation_idx in range(total_ablations):
        print("Ablation index ", ablation_idx, ": ")

        # Loading the custom MalConv model using the pre-trained one
        """
        net = Custom_MalConv(max_input_size=int((2 ** 20) / total_ablations), unfreeze=True)
        net = CClassifierEnd2EndMalware(net)
        net._n_features = int((2 ** 20) / total_ablations)
        #print(net)
        """
        net = nets[ablation_idx]
        net._set_device()

        """
        if use_cuda:
            net = net.cuda()
            print("Using CUDA")
        elif use_mps:
            net = net._set_device()
            print("Using MPS")
        """

        # Train the model
        # Use commented code for padded input model version
        net._epochs = epoch
        net.batch_size = batch_size
        # out = net.fit(x_train, y_train[:, ablation_idx])
        print("Training the model")
        print(net.batch_size)
        out = net.fit(x_train[:, ablation_idx, :], y_train[:, ablation_idx])
        net._model = out

        # Evaluate the model
        # preds = net.predict(CArray(x_train))
        preds = net.predict(CArray(x_train[:, ablation_idx, :]))
        #print(preds)
        print('Train Accuracy: ', accuracy_score(preds.tondarray(), y_train[:, ablation_idx]))

        # preds = net.predict(CArray(x_test))
        preds = net.predict(CArray(x_test[:, ablation_idx, :]))
        #print(preds)
        print('Test Accuracy: ', accuracy_score(preds.tondarray(), y_test[:, ablation_idx]))
        del net
        #nets.append(net)

    test_preds_detailed = model_predict(nets, total_ablations, x_test)
    votes = get_majority_voting(test_preds_detailed, y_test.shape[0])
    print("Final Accuracy: ", accuracy_score(votes, y_test[:, 0]))
    votes = get_majority_voting_without_padding(test_preds_detailed, y_test.shape[0], lengths_test, int(inp_len / total_ablations))
    print("Final Accuracy without Padding: ", accuracy_score(votes, y_test[:, 0]))

    with open(save_path, "wb+") as f:
        pickle.dump(nets, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the custom model")
    parser.add_argument('--mal_path', metavar='path', required=True)
    parser.add_argument('--ben_path', metavar='path', required=True)
    parser.add_argument('--save_path', metavar='path', required=True)
    parser.add_argument('--dataset_size', type=int, metavar='dataset_size', required=True)
    parser.add_argument('--ablations', type=int, metavar='total_ablations', required=True)
    parser.add_argument('--epochs', type=int, metavar='epoch', required=True)
    parser.add_argument('--batch_size', type=int, metavar='batch_size', required=False, default=16)

    args = parser.parse_args()
    main(args.mal_path, args.ben_path, args.save_path, args.dataset_size, args.ablations, args.epochs, args.batch_size)

