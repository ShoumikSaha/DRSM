import os
import magic
from secml.array import CArray
import numpy as np
import argparse
import pandas as pd

import torch

from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel

from secml_malware.attack.whitebox.c_header_evasion import CHeaderEvasion


def get_model(model_path):
    net = MalConv()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model()

    model = torch.load(model_path)
    net._model = model
    net.max_input_size = 2 ** 21
    net._trained = True
    net._input_shape = (1, 2 ** 21)
    net._n_features = 2 ** 21
    print(net)
    print(net.get_input_max_length())
    print(net._device)
    return net


def init_attack(attack_name, net):
    if attack_name == 'Partial_DOS':
        attack_obj = CHeaderEvasion(net, random_init=False, iterations=10, optimize_all_dos=False, threshold=0.5)

    return attack_obj


def get_samples_to_attack(net, csv_path, root_dir, no_samples=-1):
    X = []
    y = []
    file_names = []

    df = pd.read_csv(csv_path)
    choice = file_path_finder(root_dir, str(df.iloc[0, 0]))
    if choice == 'None': print("Check your file path!")

    for i in range(len(df)):
        if i == no_samples: break
        # print(df.iloc[i, 0], df.iloc[i, 1])
        file_name = str(df.iloc[i, 0])

        if choice == 'lucas':
            first_folder = file_name[0:2]
            second_folder = file_name[2:4]
            final_file_name = root_dir + "/" + first_folder + "/" + second_folder + "/" + file_name
        elif choice == 'smk':
            final_file_name = root_dir + "/" + file_name

        with open(final_file_name, "rb") as file_handle:
            code = file_handle.read()
        x = End2EndModel.bytes_to_numpy(
            code, net.get_input_max_length(), 256, False
        )

        _, confidence = net.predict(CArray(x), True)
        print(confidence)
        if (confidence[0, 1].item() < 0.5) or (confidence[0, 1].item() == 1.0):
            continue

        print(f"> Added {final_file_name} with confidence {confidence[0, 1].item()}")
        X.append(x)
        conf = confidence[1][0].item()
        y.append([1 - conf, conf])
        file_names.append(final_file_name)

    return X, y, file_names


def file_path_finder(root_dir, file_name):
    first_folder = file_name[0:2]
    second_folder = file_name[2:4]
    final_file_name_lucas = root_dir + "/" + first_folder + "/" + second_folder + "/" + file_name
    final_file_name_smk = root_dir + "/" + file_name

    if os.path.exists(final_file_name_lucas):
        choice = 'lucas'
    elif os.path.exists(final_file_name_smk):
        choice = 'smk'
    else:
        choice = 'None'

    return choice


def deploy_attack(net, attack_obj, X, y):
    adv_mal_list = []
    adv_mal_count = 0
    torch.set_default_device(net._device)
    for sample, label in zip(X, y):
        y_pred, adv_score, adv_ds, f_obj = attack_obj.run(CArray(sample), CArray(label[1]))
        print(attack_obj.confidences_)
        print(f_obj)
        if (f_obj < 0.5): adv_mal_count += 1
        adv_mal_list.append(adv_ds)
    return adv_mal_list


def save_adv_malwares(adv_mal_list, file_names, attack_name, attack_obj, net):
    save_dir = 'secml_malware/data/adv_mals/' + attack_name
    if (os.path.exists(save_dir) == False):
        os.mkdir(save_dir)

    for i, adv_ds in enumerate(adv_mal_list):
        print(i + 1)
        adv_x = adv_ds.X[0, :]
        save_path = save_dir + "/" + str.split(file_names[i], '/')[-1]
        real_adv_x = attack_obj.create_real_sample_from_adv(file_names[i], adv_x, save_path)
        # print(len(real_adv_x))
        real_x = End2EndModel.bytes_to_numpy(real_adv_x, net.get_input_max_length(), 256, False)
        _, confidence = net.predict(CArray(real_x), True)
        print(str.split(file_names[i], '/')[-1])
        print(confidence[0, 1].item())


def main(root_dir, csv_path, model_path, attack_name, dataset_size):
    net = get_model(model_path)
    attack_obj = init_attack(attack_name, net)
    X, y, file_names = get_samples_to_attack(net, csv_path, root_dir, dataset_size)
    adv_mal_list = deploy_attack(net, attack_obj, X, y)
    save_adv_malwares(adv_mal_list, file_names, attack_name, attack_obj, net)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Attack on the malconv model")
    parser.add_argument('--root_dir', metavar='path', required=True)
    parser.add_argument('--csv_path', metavar='path', required=True)
    parser.add_argument('--model_path', metavar='path', required=True)
    parser.add_argument('--attack_name', metavar='path', required=False, default='Partial_DOS')
    parser.add_argument('--dataset_size', type=int, metavar='dataset_size', required=False, default=-1)

    args = parser.parse_args()
    main(args.root_dir, args.csv_path, args.model_path, args.attack_name, args.dataset_size)
