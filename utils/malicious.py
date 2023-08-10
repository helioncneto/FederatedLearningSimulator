from utils import get_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from utils.data import FakeCICIDS2017Dataset

import torch.nn.functional as F
import numpy as np
import torch
import copy


def perform_fgsm_attack(data, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = data + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    #perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data


def gen_train_random_data(samples: int = 10000, features: int = 77, interval: Tuple[int, int] = (0, 1),
                          classes: tuple = (0, 1)) -> TensorDataset:
    train_np_x = np.array(
        [[np.random.uniform(interval[0], interval[1]) for _ in range(features)] for _ in range(samples)])
    #train_np_y = np.array([shuffle(np.array(classes)) for _ in range(samples)])
    train_np_y = np.array([np.random.choice(classes) for _ in range(samples)])

    train_tensor_x = torch.from_numpy(train_np_x)
    train_tensor_y = torch.from_numpy(train_np_y)

    #trainset = TensorDataset(train_tensor_x, train_tensor_y)
    trainset = FakeCICIDS2017Dataset(train_tensor_x, train_tensor_y)
    # dataloader = DataLoader(trainset, batch_size=batch, shuffle=False)
    return trainset


def reorder_dictionary(src_dict: dict, range_list: list) -> dict:
    num_items = len(src_dict)
    new_keys = np.random.choice(range_list, num_items, replace=False)
    new_dict = {}
    for i in src_dict.keys():
        new_dict[new_keys[i]] = copy.deepcopy(src_dict[i])
    return new_dict


def add_malicious_participants(args, directory: str, filepath: str) -> Tuple[TensorDataset, dict]:
    # ToDO: Criar um gerador de dataset malicioso para cada ataque; Obs.: o dataset_fake é um dicionario de idxs
    #       deve ser criado um dict com part: tipo_mal. o dataset_fake deve ser um dicionario de dicionario.
    #       Primeira chave o ataque, cada ataque contem um dicionário de part: idx
    participants_fake_num = int(args.num_of_clients * args.malicious_rate)
    print("   => The malicious participants are using random data")
    trainset_fake = gen_train_random_data(samples=args.num_fake_data)
    dataset_fake = get_dataset(args, trainset_fake, num_of_clients=participants_fake_num, mode=args.mode, compatible=False,
                               directory=directory, filepath=filepath)
    dataset_fake = reorder_dictionary(dataset_fake, list(range(args.num_of_clients)))
    print(dataset_fake.keys())
    return trainset_fake, dataset_fake


def get_fgsm_dataloader(ldr_train, model, batch_size, device, mal_epsilon):
    malicious_data = []
    malicious_target = []
    print("Start performing FGSM attack...")
    total_data = batch_size * len(ldr_train)
    fail_attack = 0

    for data_batch, target_batch in ldr_train:
        #print("Antigo just in case ", target_batch.size())
        for data, target in zip(data_batch, target_batch):
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            output = model(data.unsqueeze(0))
            #pred = output.max(1, keepdim=True)[1]
            '''if init_pred.item() != target.item():
                malicious_data.append(data)
                malicious_target.append(data)
                continue'''
            loss = F.cross_entropy(output, target.unsqueeze(0))
            #Por pra cima dps
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = perform_fgsm_attack(data, mal_epsilon, data_grad)
            # Todo: Tentar normalizar depois

            output = model(perturbed_data.unsqueeze(0))
            # Todo: Ver se errou pra dar append
            altered_target = output.max(1, keepdim=True)[1]
            if altered_target.item() == target.item():
                #print("ATAQUE MAL SUCEDIDO!!")
                fail_attack += 1
            malicious_data.append(perturbed_data.detach().cpu().numpy())
            malicious_target.append(altered_target.detach().cpu().item())

    train_tensor_x = torch.from_numpy(np.array(malicious_data))
    train_tensor_y = torch.from_numpy(np.array(malicious_target))
    trainset_fake = FakeCICIDS2017Dataset(train_tensor_x, train_tensor_y)
    print(f"Fail attacks: {fail_attack}/{total_data}")
    print(f"{fail_attack/total_data} % failed")
    return DataLoader(trainset_fake, batch_size=batch_size, shuffle=True)


