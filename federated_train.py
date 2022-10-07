
import libs.dataset.dataset_factory as dataset_loader
from libs.methods.method_factory import LOCALUPDATE_LOOKUP_TABLE, GLOBALAGGREGATION_LOOKUP_TABLE

import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import random
import wandb
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def init_env():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_device)

    experiment_name = args.set + "_" + args.global_method  + (str(args.dirichlet_alpha) if args.mode == 'dirichlet' else "") + "_" + \
                      args.method + ("_" + args.additional_experiment_name if args.additional_experiment_name != ''
                                     else '')
    group_name = args.mode + (str(args.dirichlet_alpha) if args.mode == 'dirichlet' else "")
    job_type = args.global_method + ("_" + args.additional_experiment_name if args.additional_experiment_name != '' else '')
    print("Running the experiment: ", experiment_name)

    if args.use_wandb:
        wandb_log_dir = os.path.join('data1/fed/actreg/wandb', experiment_name)
        if not os.path.exists('{}'.format(wandb_log_dir)):
            os.makedirs('{}'.format(wandb_log_dir))
        wandb.init(entity=args.entity, project=args.project,
                   group=group_name, job_type=job_type, dir=wandb_log_dir)
        wandb.run.name = experiment_name
        wandb.run.save()
        wandb.config.update(args)

    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
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

    # Build Dataset
    dataset_factory = None
    try:
        dataset_factory = dataset_loader.DATASETS_LOOKUP_TABLE[args.set]
    except KeyError:
        print('The chosen dataset is not valid.')
        print(f'Valid datasets: {list(dataset_loader.DATASETS_LOOKUP_TABLE.keys())}')
    if dataset_factory is not None:
        global_update, local_update = None, None

        trainset, testset, valset = dataset_factory.get_dataset()
        testloader = DataLoader(testset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)
        valloader = DataLoader(valset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)
        try:
            method = args.method.casefold()
            local_update = LOCALUPDATE_LOOKUP_TABLE[method].get_local_method()
        except KeyError:
            print('The chosen method is not valid.')
            print(f'Valid methods: {list(LOCALUPDATE_LOOKUP_TABLE.keys())}')
        try:
            global_method =args.global_method.casefold()
            global_update = GLOBALAGGREGATION_LOOKUP_TABLE[global_method].get_global_method()
        except KeyError:
            print('The chosen global method is not valid.')
            print(f'Valid global methods: {list(GLOBALAGGREGATION_LOOKUP_TABLE.keys())}')
        if global_update is not None and local_update is not None:
            global_update(args=args, device=device, trainset=trainset, testloader=testloader, valloader=valloader,
                          local_update=local_update)


if __name__ == '__main__':
    from args_dir.federated import args
    main()
