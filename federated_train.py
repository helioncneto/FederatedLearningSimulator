from args_dir.federated import args
from libs.dataset.dataset_loader import *

import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import random
import wandb
from build_method import build_local_update_module
from build_global_method import build_global_update_module
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def init_env():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_visible_device)

    experiment_name = args.set + "_"+args.mode+(str(args.dirichlet_alpha) if args.mode == 'dirichlet' else "") + "_" + \
                      args.method + ("_" + args.additional_experiment_name if args.additional_experiment_name != ''
                                     else '')
    print("Running the experiment: ", experiment_name)

    wandb_log_dir = os.path.join('data1/fed/actreg/wandb', experiment_name)
    if not os.path.exists('{}'.format(wandb_log_dir)):
        os.makedirs('{}'.format(wandb_log_dir))
    wandb.init(entity=args.entity, project=args.project,
               group=args.mode+(str(args.dirichlet_alpha) if args.mode=='dirichlet' else ""),
               job_type=args.method+("_"+args.additional_experiment_name if args.additional_experiment_name!='' else ''),
               dir=wandb_log_dir)
    wandb.run.name = experiment_name
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

    # Check if the data path exists. If not it will create
    if not os.path.exists(args.data):
        os.mkdir(args.data)


def main():
    """The main function of the federated learning simulator"""
    # initiate the simulator environment
    init_env()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    factory_lookup_table = {
        'CIFAR10': Cifar10DatasetFactory(),
        'CIFAR100': Cifar100DatasetFactory(),
        'Tiny-ImageNet': TinyImageNetDatasetFactory(),
        'CICIDS2017': CICIDS2017DatasetFactory()
    }

    # Build Dataset
    try:
        factory = factory_lookup_table[args.set]
        trainset, testset = factory.get_dataset()
        # trainloader = DataLoader(trainset, batch_size=args.batch_size,
        #                                          shuffle=True, num_workers=args.workers)
        testloader = DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.workers)
        LocalUpdate = build_local_update_module(args)
        global_update = build_global_update_module(args)
        global_update(args=args, device=device, trainset=trainset, testloader=testloader, LocalUpdate=LocalUpdate)
    except KeyError:
        print("The chosen dataset is not valid.\n")
        print("Valid datasets: CIFAR10, CIFAR100, Tiny-ImageNet, and CICIDS2017")


if __name__ == '__main__':
    main()
