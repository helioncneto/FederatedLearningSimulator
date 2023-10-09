from utils import get_dataset, get_model
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from utils.data import FakeCICIDS2017Dataset
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE

import torch.nn.functional as F
import torchattacks
import numpy as np
import torch
import copy
import pandas as pd
import os


def get_random_datapoints(samples: int, features: int, classes: tuple, interval):
    print("   => The malicious participants are using random data")
    train_np_x = np.array(
        [[np.random.uniform(interval[0], interval[1]) for _ in range(features)] for _ in range(samples)])
    train_np_y = np.array([np.random.choice(classes) for _ in range(samples)])
    return train_np_x, train_np_y


def gen_train_malicious_data(args, samples: int, dirpath: str, features: int = 77, interval: Tuple[int, int] = (0, 1),
                             classes: tuple = (0, 1)) -> TensorDataset:
    filepath = dirpath + args.set + '_' + str(args.num_fake_data) + '_datapoints_' + args.malicious_type + '.csv'
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

    # Create or load the data
    if not os.path.isfile(filepath):
        train_np_x, train_np_y = get_random_datapoints(samples, features, classes, interval)
        dataset = np.concatenate((train_np_x, train_np_y.reshape(samples, 1)), axis=1)
        pd.DataFrame(dataset).to_csv(filepath, header=False, index=False)
    else:
        dataset = pd.read_csv(filepath, header=None)
        train_np_x = dataset.drop(features, axis=1)
        train_np_y = dataset[features].astype(int)
        train_np_x = train_np_x.to_numpy()
        train_np_y = train_np_y.to_numpy()

    train_tensor_x = torch.from_numpy(train_np_x)
    train_tensor_y = torch.from_numpy(train_np_y)
    trainset_fake = FakeCICIDS2017Dataset(train_tensor_x, train_tensor_y)

    return trainset_fake


def reorder_dictionary(src_dict: dict, range_list: list) -> dict:
    num_items = len(src_dict)
    new_keys = np.random.choice(range_list, num_items, replace=False)
    new_dict = {}
    for i in src_dict.keys():
        new_dict[new_keys[i]] = copy.deepcopy(src_dict[i])
    return new_dict


def add_malicious_participants(args, directory: str, filepath: str) -> Tuple[TensorDataset, dict]:
    participants_fake_num = int(args.num_of_clients * args.malicious_rate)
    trainset_fake = gen_train_malicious_data(args=args, samples=args.num_fake_data,  dirpath='data/malicious/')
    dataset_fake = get_dataset(args, trainset_fake, num_of_clients=participants_fake_num, mode=args.mode, compatible=False,
                               directory=directory, filepath=filepath)
    dataset_fake = reorder_dictionary(dataset_fake, list(range(args.num_of_clients)))
    return trainset_fake, dataset_fake


def get_fgsm_dataloader(ldr_train, model, batch_size, mal_epsilon):
    print("Start performing untargeted FGSM attack...")
    atk = torchattacks.FGSM(model, eps=mal_epsilon)
    adv_data_x = []
    adv_data_y = []

    for data_batch, target_batch in ldr_train:
        # print(data_batch)
        adv_X = atk(data_batch, target_batch)
        adv_X = adv_X.detach().cpu().numpy()
        # print(adv_X)
        #for i, x in enumerate(adv_X):
        for x in adv_X:
            y = np.random.choice((0, 1))
            adv_data_x.append(x)
            adv_data_y.append(y)

    train_tensor_x = torch.from_numpy(np.array(adv_data_x))
    train_tensor_y = torch.from_numpy(np.array(adv_data_y))
    trainset_fake = FakeCICIDS2017Dataset(train_tensor_x, train_tensor_y)
    return DataLoader(trainset_fake, batch_size=batch_size, shuffle=True)


def get_malicious_loader(malicious, ldr_train, model, batch_size, mal_epsilon):
    if malicious == 'random':
        print("Malicioso Aleatorio")
        return ldr_train
    elif not malicious:
        return ldr_train
    elif malicious == 'fgsm':
        return get_fgsm_dataloader(ldr_train, model, batch_size, mal_epsilon)
    else:
        print("Malicious type incorrect... \nFailed acting maliciously")
        return ldr_train
