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

from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware

from secml_malware.smoothed_malconv import get_dataset, create_smoothed_malconv, modify_dataset_for_smoothed_malconv, \
    pad_ablated_input, train_model, model_predict, get_majority_voting, get_majority_voting_without_padding

from secml_malware.custom_malconv import Custom_MalConv


def main(mal_path, ben_path, model_path, dataset_size, total_ablations):
    X, y, file_names, lengths = get_dataset(mal_path, ben_path, 2 ** 20, int(dataset_size/2))
    print(y)
    new_X, new_y = modify_dataset_for_smoothed_malconv(X, np.reshape(y, (-1)), total_ablations)
    print(new_X.shape, new_y.shape)

    ##Loading the model
    with open(model_path, "rb") as file:
        nets = pickle.load(file)

    test_preds_detailed = model_predict(nets, total_ablations, new_X)
    votes = get_majority_voting_without_padding(test_preds_detailed, new_y.shape[0], lengths,
                                                int((2 ** 20) / total_ablations))
    for (file,vote) in zip(file_names,votes):
        print(file, vote)
    print("Accuracy without Padding: ", accuracy_score(votes, new_y[:, 0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the custom model")
    parser.add_argument('--mal_path', metavar='path', required=False)
    parser.add_argument('--ben_path', metavar='path', required=False)
    parser.add_argument('--model_path', metavar='path', required=True)
    parser.add_argument('--dataset_size', type=int, metavar='dataset_size', required=False, default=-2)
    parser.add_argument('--ablations', type=int, metavar='total_ablations', required=True)

    args = parser.parse_args()
    main(args.mal_path, args.ben_path, args.model_path, args.dataset_size, args.ablations)
