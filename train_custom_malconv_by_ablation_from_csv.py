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
from secml_malware.nonneg_malconv import NonNeg_MalConv, weightConstraint
from secml_malware.models.my_dataloader_csv import MyDataSet
from torch.utils.data import Dataset, DataLoader, random_split, default_collate

from secml.settings import SECML_PYTORCH_USE_CUDA
use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA
use_mps = torch.backends.mps.is_available()


inp_len = 2**21


def main(root_dir, train_path, val_path, test_path, dir_path, ablation_idx, dataset_size, total_ablations=4, epoch=2, batch_size=16, non_neg=False):
    if(non_neg):
        net = NonNeg_MalConv(ablation_idx=ablation_idx, max_input_size=int(inp_len / total_ablations), unfreeze=True)
        constraints = weightConstraint()
        print(net)
        net._modules['linear2'].apply(constraints)
    else:
        net = Custom_MalConv(ablation_idx=ablation_idx, max_input_size=int(inp_len / total_ablations), unfreeze=True)
    #constraints = weightConstraint()
    #if(non_neg==True): net._modules['classifier'].apply(constraints)
    net = CClassifierEnd2EndMalware(net, batch_size=batch_size)
    net._n_features = int(inp_len / total_ablations)
    net._epochs = epoch
    net.batch_size = batch_size
    net._optimizer_scheduler = torch.optim.lr_scheduler.StepLR(net._optimizer, step_size=2, gamma=0.1)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    model_path = dir_path + "/" + "smoothed_malconv_" + str(total_ablations) + "_" + str(ablation_idx) + ".h5"
    if os.path.exists(model_path):
        print("Loading the model from path")
        # net.load_model(model_path)
        net._model = torch.load(model_path)
    else:
        print("Created the model")

    print("Model Name: ", model_path)
    print("Ablation index ", ablation_idx, ": ")

    #generator1 = torch.Generator().manual_seed(42)
    trainset = MyDataSet(root_dir, train_path, inp_len, ablation_idx, total_ablations, dataset_size)
    validset = MyDataSet(root_dir, val_path, inp_len, ablation_idx, total_ablations, dataset_size)
    testset = MyDataSet(root_dir, test_path, inp_len, ablation_idx, total_ablations, dataset_size)
    #trainset, validset, testset = random_split(dataset, [0.7, 0.15, 0.15], generator=generator1)
    train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(validset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size)

    print(train_loader)
    net._model = train(net, epoch, train_loader, valid_loader, model_path)

    # net._model.eval()
    train_acc = evaluate2(net, train_loader)
    print('Train Accuracy: ', train_acc)
    test_acc = evaluate2(net, test_loader)
    print('Test Accuracy: ', test_acc)

    # net.save_model(model_path)
    # torch.save(net._model, model_path)

    # net.load_model(model_path)
    # test_acc = evaluate2(net, test_loader)
    # print('Test Accuracy: ', test_acc)

def train(net, epochs, train_loader, valid_loader, model_path):
    min_valid_loss = np.inf
    for epoch in range(epochs):
        running_loss = 0.0
        net._model.train()
        for i, data in enumerate(train_loader):
            inputs, labels, lengths = data
            # print(inputs, labels)
            inputs = inputs.to(net._device)
            labels = labels.to(net._device)
            # labels = labels.to(net._device).unsqueeze(1)
            net._optimizer.zero_grad()
            outputs = net._model(inputs)
            # print(outputs.size(), labels.size())
            # print(outputs)
            loss = net._loss(outputs, labels)
            loss.backward()
            net._optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                net.logger.info('[%d, %5d] loss: %.3f' %
                                (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        if net._optimizer_scheduler is not None:
            net._optimizer_scheduler.step()

        print('Epoch ', epoch + 1)
        print('Train Loss: ', running_loss)

        valid_loss = 0
        net._model.eval()
        for i, data in enumerate(valid_loader):
            inputs, labels, lengths = data
            # print(inputs, labels)
            inputs = inputs.to(net._device)
            labels = labels.to(net._device)
            outputs = net._model(inputs)
            # print(outputs.size(), labels.size())
            # print(outputs)
            loss = net._loss(outputs, labels)
            valid_loss += loss.item()
        print('Valid Loss: ', valid_loss)
        if valid_loss<min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(net._model, model_path)
            print("Saved the model!")


    net._trained = True
    return net._model


def evaluate(model, set):
    # create one test tensor from the set
    X_test, y_test = default_collate(set)
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc)
    return acc


def evaluate2(net, data_generator):

    total_acc = 0.0
    count = 0
    net._model.eval()
    with torch.set_grad_enabled(False):
        for local_batch, local_labels, local_lengths in data_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(net._device), local_labels.to(net._device)
            preds = net._model(local_batch)
            acc = (preds.round() == local_labels).float().mean()
            acc = float(acc)
            # print(acc)
            total_acc += acc
            count += 1
    return total_acc / count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the custom model")
    parser.add_argument('--root_dir', metavar='path', required=True)
    parser.add_argument('--train_path', metavar='path', required=True)
    parser.add_argument('--val_path', metavar='path', required=True)
    parser.add_argument('--test_path', metavar='path', required=True)
    parser.add_argument('--dir_path', metavar='path', required=True)
    parser.add_argument('--ablation_idx', type=int, metavar='ablation_idx', required=True)
    parser.add_argument('--dataset_size', type=int, metavar='dataset_size', required=False, default=-2)
    parser.add_argument('--ablations', type=int, metavar='total_ablations', required=True)
    parser.add_argument('--epochs', type=int, metavar='epoch', required=True)
    parser.add_argument('--batch_size', type=int, metavar='batch_size', required=False, default=16)
    parser.add_argument('--non_neg', type=bool, metavar='non_neg', required=False, default=False)

    args = parser.parse_args()
    main(args.root_dir, args.train_path, args.val_path, args.test_path, args.dir_path, args.ablation_idx, args.dataset_size, args.ablations, args.epochs,
         args.batch_size, args.non_neg)




