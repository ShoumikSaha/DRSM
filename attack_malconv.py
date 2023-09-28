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
from secml_malware.attack.whitebox import CKreukEvasion
from secml_malware.attack.whitebox.c_headerfields_evasion import CHeaderFieldsEvasion
from secml_malware.attack.whitebox.c_extend_dos_evasion import CExtendDOSEvasion

from secml_malware.attack.blackbox.c_wrapper_phi import CEnd2EndWrapperPhi
from secml_malware.attack.blackbox.c_gamma_sections_evasion import CGammaSectionsEvasionProblem
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from secml_malware.attack.blackbox.c_blackbox_headerfields_problem import CBlackBoxHeaderFieldsEvasionProblem
from secml_malware.attack.blackbox.c_blackbox_header_problem import CBlackBoxHeaderEvasionProblem



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


def init_attack(attack_name, net, root_dir):
    if attack_name == 'Partial_DOS':
        attack_obj = CHeaderEvasion(net, random_init=False, iterations=10, optimize_all_dos=False, threshold=0.5)
    elif attack_name == 'Full_DOS':
        attack_obj = CHeaderEvasion(net, random_init=False, iterations=10, optimize_all_dos=True, threshold=0.5)
    elif attack_name == 'DOS_Extend':
        attack_obj = CExtendDOSEvasion(net, iterations=10, is_debug=True, threshold=0.5, chunk_hyper_parameter=256, penalty_regularizer=1e-3)
    elif attack_name == 'Kreuk_Gradient':
        attack_obj = CKreukEvasion(net, how_many_padding_bytes=10240, compute_slack=False, epsilon=1, iterations=10, is_debug=True)
    elif attack_name == 'Suciu_Gradient':
        attack_obj = CKreukEvasion(net, how_many_padding_bytes=10240, compute_slack=True, epsilon=1, iterations=10, is_debug=True)
    elif attack_name == 'HField_Evasion':
        attack_obj = CHeaderFieldsEvasion(net, iterations=20, is_debug=True, threshold=0.5)
    elif attack_name == 'HField_Evasion_blackbox':
        attack = CBlackBoxHeaderFieldsEvasionProblem(CEnd2EndWrapperPhi(net), population_size=200, penalty_regularizer=1e-6, iterations=20)
        attack_obj = CGeneticAlgorithm(attack)
    elif attack_name == 'Partial_DOS_blackbox':
        attack = CBlackBoxHeaderEvasionProblem(CEnd2EndWrapperPhi(net), population_size=200, penalty_regularizer=1e-6, iterations=20)
        attack_obj = CGeneticAlgorithm(attack)
    elif attack_name == 'Full_DOS_blackbox':
        attack = CBlackBoxHeaderEvasionProblem(CEnd2EndWrapperPhi(net), population_size=200, optimize_all_dos=True, penalty_regularizer=1e-6, iterations=20)
        attack_obj = CGeneticAlgorithm(attack)
    elif attack_name == 'Gamma':
        goodware_folder = root_dir + '/benign'  # INSERT GOODWARE IN THAT FOLDER
        section_population, what_from_who = CGammaSectionsEvasionProblem.create_section_population_from_folder(goodware_folder, how_many=200, sections_to_extract=['.data'])
        attack = CGammaSectionsEvasionProblem(section_population, CEnd2EndWrapperPhi(net), population_size=200, penalty_regularizer=1e-12,
                                              iterations=20, threshold=0.5)
        attack_obj = CGeneticAlgorithm(attack, is_debug=False)
    else:
        print(attack_name, 'does not exist!')
        return
    return attack_obj


def get_samples_to_attack(net, csv_path, root_dir, no_samples=-1):
    X = []
    y = []
    file_names = []

    df = pd.read_csv(csv_path)
    choice = file_path_finder(root_dir, str(df.iloc[0, 0]))
    if choice == 'None': print("Check your file path!")

    for i in range(len(df)):
        if len(X) >= no_samples: break
        # print(df.iloc[i, 0], df.iloc[i, 1])
        label = int(df.iloc[i, 1])
        if label == 0:  continue

        file_name = str(df.iloc[i, 0])

        if choice == 'lucas':
            first_folder = file_name[0:2]
            second_folder = file_name[2:4]
            final_file_name = root_dir + "/" + first_folder + "/" + second_folder + "/" + file_name
        elif choice == 'smk':
            final_file_name = root_dir + "/" + file_name

        if "PE32" not in magic.from_file(final_file_name):
            continue

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


def deploy_attack(net, attack_name, attack_obj, X, y, file_names):
    adv_mal_list = []
    adv_mal_count = 0
    torch.set_default_device(net._device)
    for sample, label, file_name in zip(X, y, file_names):
        print("Attacking ", file_name)
        if (attack_name=='Suciu_Gradient' and attack_obj.compute_slack):
            attack_obj.indexes_to_perturb = attack_obj.create_slack_indexes2(file_name)   ##This line for Suciu attack
        #y_pred, adv_score, adv_ds, f_obj = attack_obj.run(CArray(sample), CArray(label[1]))
        if (attack_name == 'Gamma') or (attack_name.split('_')[-1] == 'blackbox'):
            #y_pred, adv_score, adv_ds, f_obj = attack_obj.run(CArray(np.frombuffer(sample, dtype=np.uint8)), CArray(label[1]), file_name)
            y_pred, adv_score, adv_ds, f_obj = attack_obj.run(CArray(sample).astype(int),
                                                              CArray(label[1]), file_name)
        elif attack_name == 'DOS_Extend':
            #adv_ds, f_obj = attack_obj._run([CArray(np.frombuffer(sample, dtype=np.uint8)), file_name], CArray(label[1]))
            adv_ds, f_obj = attack_obj.run([CArray(sample), file_name],
                                            CArray(label[1]))
        else:
            y_pred, adv_score, adv_ds, f_obj = attack_obj.run(CArray(sample), CArray(label[1]))
        print(attack_obj.confidences_)
        print(f_obj)
        if (f_obj < 0.5): adv_mal_count += 1
        adv_mal_list.append(adv_ds)
    print("Adversarial count: ", adv_mal_count)
    return adv_mal_list


def save_adv_malwares(adv_mal_list, file_names, attack_name, attack_obj, net):
    save_dir = 'secml_malware/data/adv_mals/' + attack_name
    if (os.path.exists(save_dir) == False):
        os.mkdir(save_dir)

    for i, adv_ds in enumerate(adv_mal_list):
        print(i + 1)
        if attack_name == 'DOS_Extend':
            adv_x = adv_ds
        else:
            adv_x = adv_ds.X[0, :]

        save_path = save_dir + "/" + str.split(file_names[i], '/')[-1]

        if (attack_name == 'Gamma') or (attack_name.split('_')[-1] == 'blackbox'):
            adv_x = CArray(adv_x[:]).astype(int)
            #print(type(adv_x))
            #print(adv_x)
            read_adv_x = attack_obj.write_adv_to_file(adv_x, save_path)
            real_x = End2EndModel.bytes_to_numpy(read_adv_x, net.get_input_max_length(), 256, False)
        else:
            real_adv_x = attack_obj.create_real_sample_from_adv(file_names[i], adv_x, save_path)
            real_x = End2EndModel.bytes_to_numpy(real_adv_x, net.get_input_max_length(), 256, False)
        _, confidence = net.predict(CArray(real_x), True)
        print(str.split(file_names[i], '/')[-1])
        print(confidence[0, 1].item())


def main(root_dir, csv_path, model_path, attack_name, dataset_size):
    net = get_model(model_path)
    torch.set_default_device(net._device)
    attack_obj = init_attack(attack_name, net, root_dir)
    X, y, file_names = get_samples_to_attack(net, csv_path, root_dir, dataset_size)
    adv_mal_list = deploy_attack(net, attack_name, attack_obj, X, y, file_names)
    save_adv_malwares(adv_mal_list, file_names, attack_name, attack_obj, net)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Attack on the malconv model")
    parser.add_argument('--root_dir', metavar='path', required=True)
    parser.add_argument('--csv_path', metavar='path', required=True)
    parser.add_argument('--model_path', metavar='path', required=False, default='secml_malware/data/trained/smoothed_1/smoothed_malconv_1_0.h5')
    parser.add_argument('--attack_name', metavar='string', required=True, default='Partial_DOS')
    parser.add_argument('--dataset_size', type=int, metavar='dataset_size', required=False, default=-1)

    args = parser.parse_args()
    main(args.root_dir, args.csv_path, args.model_path, args.attack_name, args.dataset_size)
