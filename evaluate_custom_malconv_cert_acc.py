import os
import magic
from secml.array import CArray
import numpy as np
import torch
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
# import pickle
import dill as pickle

from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware

from secml_malware.smoothed_malconv import get_dataset, create_smoothed_malconv, modify_dataset_for_smoothed_malconv, \
    pad_ablated_input, train_model, model_predict, get_majority_voting, get_majority_voting_without_padding

from secml_malware.custom_malconv import Custom_MalConv

from secml_malware.models.my_dataloader_csv import MyDataSet
from torch.utils.data import Dataset, DataLoader, random_split, default_collate

inp_len = 2 ** 21


def main(root_dir, train_path, val_path, test_path, dir_path, dataset_size, total_ablations, batch_size=16, perturb_size=20000):
    """
    net = Custom_MalConv(max_input_size=int(inp_len / total_ablations), unfreeze=True)
    net = CClassifierEnd2EndMalware(net, batch_size=batch_size)
    net._n_features = int(inp_len / total_ablations)
    """

    print("Perturbation Size: ", perturb_size)

    nets = []
    ablation_idxs = []
    for i, f in enumerate(os.listdir(dir_path)):
        if ".h5" not in f:
            continue
        ablation_idx = int(f.split('_')[-1].split('.')[0])
        model_path = os.path.join(dir_path, f)
        print(model_path)
        print("Loading the model from path")
        print(ablation_idx)
        # net.load_model(model_path)
        net_model = torch.load(model_path)
        nets.append(net_model)
        ablation_idxs.append(ablation_idx)

    test_preds_all_models = []
    train_preds_all_models = []
    val_preds_all_models = []
    for i, net in enumerate(nets):
        ablation_idx = ablation_idxs[i]
        #generator1 = torch.Generator().manual_seed(42)
        trainset = MyDataSet(root_dir, train_path, inp_len, ablation_idx, total_ablations, dataset_size)
        validset = MyDataSet(root_dir, val_path, inp_len, ablation_idx, total_ablations, dataset_size)
        testset = MyDataSet(root_dir, test_path, inp_len, ablation_idx, total_ablations, dataset_size)
        #trainset, validset, testset = random_split(dataset, [0.7, 0.15, 0.15], generator=generator1)
        train_loader = DataLoader(trainset, shuffle=False, batch_size=batch_size)
        valid_loader = DataLoader(validset, shuffle=False, batch_size=batch_size)
        test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size)

        test_preds, lengths_all = get_predicts(net, test_loader)
        test_preds_all_models.append(test_preds)

        train_preds, lengths_all_train = get_predicts(net, train_loader)
        train_preds_all_models.append(train_preds)

        val_preds, lengths_all_val = get_predicts(net, valid_loader)
        val_preds_all_models.append(val_preds)

    votes, certified_votes = get_majority_voting_without_padding(np.asarray(train_preds_all_models), len(train_preds_all_models[0]),
                                                lengths_all_train,
                                                int(inp_len / total_ablations), perturb_size)

    cert_train_acc = get_acc(certified_votes, train_loader)
    print("Train Accuracy (Certified): ", cert_train_acc)

    votes, certified_votes = get_majority_voting_without_padding(np.asarray(val_preds_all_models), len(val_preds_all_models[0]),
                                                lengths_all_val,
                                                int(inp_len / total_ablations), perturb_size)

    cert_val_acc = get_acc(certified_votes, valid_loader)
    print("Validation Accuracy (Certified): ", cert_val_acc)

    votes, certified_votes = get_majority_voting_without_padding(np.asarray(test_preds_all_models), len(test_preds_all_models[0]),
                                                lengths_all,
                                                int(inp_len / total_ablations), perturb_size)

    cert_test_acc = get_acc(certified_votes, test_loader)
    print("Test Accuracy (Certified): ", cert_test_acc)


def get_predicts(net_model, data_generator):
    net_model.eval()
    preds_all_samples = []
    lengths_all = []
    for local_batch, local_labels, local_lengths in data_generator:
        # Transfer to GPU
        #local_batch, local_labels = local_batch.to(net._device), local_labels.to(net._device)
        preds = net_model(local_batch).cpu()
        #print(preds)
        preds = preds.round().detach().numpy()
        preds_all_samples.extend(preds)
        # print(preds)
        lengths_all.extend(local_lengths.detach().numpy())
    preds_all_samples = np.asarray(preds_all_samples)
    lengths_all = np.asarray(lengths_all)
    return preds_all_samples.flatten(), lengths_all.flatten()

def get_acc(votes, data_generator):
    labels = []
    for local_batch, local_labels, local_lengths in data_generator:
        #print(local_labels)
        labels.extend(local_labels.numpy())
    labels = np.asarray(labels)
    acc = accuracy_score(votes, labels)
    return acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the custom model")
    parser.add_argument('--root_dir', metavar='path', required=True)
    parser.add_argument('--train_path', metavar='path', required=True)
    parser.add_argument('--val_path', metavar='path', required=True)
    parser.add_argument('--test_path', metavar='path', required=True)
    parser.add_argument('--dir_path', metavar='path', required=True)
    parser.add_argument('--dataset_size', type=int, metavar='dataset_size', required=False, default=-2)
    parser.add_argument('--ablations', type=int, metavar='total_ablations', required=True)
    parser.add_argument('--batch_size', type=int, metavar='batch_size', required=True)
    parser.add_argument('--perturb_size', type=int, metavar='length of perturbation', required=False, default=20000)

    args = parser.parse_args()
    main(args.root_dir, args.train_path, args.val_path, args.test_path, args.dir_path, args.dataset_size, args.ablations, args.batch_size, args.perturb_size)
