
import sys
import time

from args_dir.federated import args
from libs.dataset.cicids import CICIDS2017Dataset, create_cic_ids_file

import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import random
import wandb
from build_method import build_local_update_module
from build_global_method import build_global_update_module
from sklearn.model_selection import train_test_split
import datasets as local_datasets
from utils import get_scheduler, get_optimizer, get_model, get_dataset
import copy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_visible_device)

experiment_name=args.set+"_"+args.mode+(str(args.dirichlet_alpha) if args.mode=='dirichlet' else "")+"_"+args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else '')
print(experiment_name)
wandb_log_dir = os.path.join('data1/fed/actreg/wandb', experiment_name)
if not os.path.exists('{}'.format(wandb_log_dir)):
    os.makedirs('{}'.format(wandb_log_dir))
wandb.init(entity=args.entity, project=args.project, group=args.mode+(str(args.dirichlet_alpha) if args.mode=='dirichlet' else ""),job_type=args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else '')
           , dir=wandb_log_dir)
wandb.run.name=experiment_name
wandb.run.save()
wandb.config.update(args)

random_seed = args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Build Dataset
trainset = None
testset = None

if args.set in ['CIFAR10', 'CIFAR100']:
    normalize=transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) if args.set=='CIFAR10' else transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose(
        [transforms.RandomRotation(10),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize
         ])

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize])

    if args.set=='CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=args.data, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.data, train=False,
                                                   download=True, transform=transform_test)
        # classes = ('plane', 'car', 'bird', 'cat',
        #                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.set == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=args.data, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.data, train=False,
                                                   download=True, transform=transform_test) 
        #classes= tuple(str(i) for i in range(100))

elif args.set in ['Tiny-ImageNet']:
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),  # RandomRotation
        transforms.RandomCrop(64, padding=4),
        # resize 256_comb_coteach_OpenNN_CIFAR -> random_crop 224 ==> crop 32, padding 4
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
    ])
    trainset = local_datasets.TinyImageNetDataset(
        root=os.path.join(args.data, 'tiny_imagenet'),
        split='train',
        transform=transform_train
    )
    testset = local_datasets.TinyImageNetDataset(
        root=os.path.join(args.data, 'tiny_imagenet'),
        split='test',
        transform=transform_train
    )
    #classes = tuple(str(i) for i in range(100))

elif args.set in ['CICIDS2017']:
    files_path = os.path.join('data', 'CICIDS2017')
    cic_ids_path = os.path.join(files_path, 'cicids2017.csv')
    if not os.path.exists(cic_ids_path):
        print("CICIDS2017 file not exists.")
        create_cic_ids_file()
    print('Loading CICIDS2017 file...')

    data = pd.read_csv(cic_ids_path)
    data['Flow Bytes/s'] = data['Flow Bytes/s'].astype(float)
    data[' Flow Packets/s'] = data[' Flow Packets/s'].astype(float)
    # Drop NAN and INF
    data = data.replace(np.inf, np.NaN)
    data.dropna(inplace=True)

    train, test = train_test_split(data, test_size=0.3)
    trainset = CICIDS2017Dataset(train)
    testset = CICIDS2017Dataset(test)
else:
    assert False

if trainset is not None and testset is not None:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
else:
    raise Exception("Train and Test dataset are not setup")

LocalUpdate = build_local_update_module(args)
global_update = build_global_update_module(args)
global_update(args=args, device=device, trainset=trainset, testloader=testloader, LocalUpdate=LocalUpdate)
