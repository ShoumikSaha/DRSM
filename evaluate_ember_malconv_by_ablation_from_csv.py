import os
import magic
from secml.array import CArray
import numpy as np
import torch
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, precision_recall_fscore_support
import argparse
# import pickle
import dill as pickle
import matplotlib.pyplot as plt

from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware

from secml_malware.smoothed_malconv import get_dataset, create_smoothed_malconv, modify_dataset_for_smoothed_malconv, \
    pad_ablated_input, train_model, model_predict, get_majority_voting, get_majority_voting_without_padding

from secml_malware.custom_malconv import Custom_MalConv

from secml_malware.models.my_dataloader_csv import MyDataSet
from torch.utils.data import Dataset, DataLoader, random_split, default_collate

inp_len = 2 ** 20


def main(root_dir, train_path, val_path, test_path, dataset_size, total_ablations=1, batch_size=16, perturb_size=20000):

    net = MalConv()
    net = CClassifierEnd2EndMalware(net, batch_size=batch_size)
    net.load_pretrained_model()


    test_preds_all_models = []
    train_preds_all_models = []
    val_preds_all_models = []

    ablation_idx = 0

    trainset = MyDataSet(root_dir, train_path, inp_len, ablation_idx, total_ablations, dataset_size)
    validset = MyDataSet(root_dir, val_path, inp_len, ablation_idx, total_ablations, dataset_size)
    testset = MyDataSet(root_dir, test_path, inp_len, ablation_idx, total_ablations, dataset_size)

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
    train_acc, y_true_train = get_acc(votes, train_loader)
    print("Train Accuracy (Standard): ", train_acc)
    fpr, tpr, threshold, roc_auc = get_roc_values(y_true_train, votes)
    plot_roc_curve(fpr, tpr, roc_auc, str(str(total_ablations) + '_ember_train'))
    print("fpr, tpr, roc_auc: ",fpr, tpr, roc_auc)
    conf_matrix = confusion_matrix(y_true_train, votes).ravel()
    print("tn, fp, fn, tp: ",conf_matrix)
    print("prec, recall, f1, sup:", precision_recall_fscore_support(y_true_train, votes, average='binary'))
    #cert_train_acc = get_acc(certified_votes, train_loader)
    #print("Train Accuracy (Certified): ", cert_train_acc)

    votes, certified_votes = get_majority_voting_without_padding(np.asarray(val_preds_all_models), len(val_preds_all_models[0]),
                                                lengths_all_val,
                                                int(inp_len / total_ablations), perturb_size)
    val_acc, y_true_val = get_acc(votes, valid_loader)
    print("Validation Accuracy (Standard): ", val_acc)
    fpr, tpr, threshold, roc_auc = get_roc_values(y_true_val, votes)
    plot_roc_curve(fpr, tpr, roc_auc, str(str(total_ablations) + '_ember_val'))
    print("fpr, tpr, roc_auc: ",fpr, tpr, roc_auc)
    conf_matrix = confusion_matrix(y_true_val, votes).ravel()
    print("tn, fp, fn, tp: ",conf_matrix)
    print("prec, recall, f1, sup:", precision_recall_fscore_support(y_true_val, votes, average='binary'))
    #cert_val_acc = get_acc(certified_votes, valid_loader)
    #print("Validation Accuracy (Certified): ", cert_val_acc)

    votes, certified_votes = get_majority_voting_without_padding(np.asarray(test_preds_all_models), len(test_preds_all_models[0]),
                                                lengths_all,
                                                int(inp_len / total_ablations), perturb_size)
    test_acc, y_true_test = get_acc(votes, test_loader)
    print("Test Accuracy (Standard): ", test_acc)
    fpr, tpr, threshold, roc_auc = get_roc_values(y_true_test, votes)
    plot_roc_curve(fpr, tpr, roc_auc, str(str(total_ablations) + '_ember_test'))
    print("fpr, tpr, roc_auc: ",fpr, tpr, roc_auc)
    conf_matrix = confusion_matrix(y_true_test, votes).ravel()
    print("tn, fp, fn, tp: ",conf_matrix)
    print("prec, recall, f1, sup:", precision_recall_fscore_support(y_true_test, votes, average='binary'))
    #cert_test_acc = get_acc(certified_votes, test_loader)
    #print("Test Accuracy (Certified): ", cert_test_acc)


def get_predicts(net_model, data_generator):
    #net_model.eval()
    preds_all_samples = []
    lengths_all = []
    for local_batch, local_labels, local_lengths in data_generator:
        # Transfer to GPU
        #local_batch, local_labels = local_batch.to(net._device), local_labels.to(net._device)
        preds = net_model.predict(local_batch).tondarray()
        #print(preds)
        #preds = preds.round().detach().numpy()
        preds = np.round(preds)
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
    return acc, labels

def get_roc_values(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(np.asarray(y_true), np.asarray(y_pred))
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, threshold, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc, total_ablations):
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('secml_malware/data/plots/roc_' + total_ablations + ".png", bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the custom model")
    parser.add_argument('--root_dir', metavar='path', required=True)
    parser.add_argument('--train_path', metavar='path', required=True)
    parser.add_argument('--val_path', metavar='path', required=True)
    parser.add_argument('--test_path', metavar='path', required=True)
    #parser.add_argument('--dir_path', metavar='path', required=True)
    parser.add_argument('--dataset_size', type=int, metavar='dataset_size', required=False, default=-2)
    parser.add_argument('--ablations', type=int, metavar='total_ablations', required=True)
    parser.add_argument('--batch_size', type=int, metavar='batch_size', required=True)
    parser.add_argument('--perturb_size', type=int, metavar='length of perturbation', required=False, default=20000)

    args = parser.parse_args()
    main(args.root_dir, args.train_path, args.val_path, args.test_path, args.dataset_size, args.ablations, args.batch_size, args.perturb_size)
