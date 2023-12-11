import os
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import torchvision
import pandas as pd
import numpy as np

import datasets as local_datasets
#from args_dir.federated import args
from libs.dataset import cicids_dataset
from utils.log import setup_custom_logger, LOG_LEVEL


class IDatasetFactory(ABC):
    """Basic representation of a Dataset factory"""

    @abstractmethod
    def get_dataset(self, args) -> tuple:
        """Return the Dataset class"""


class TransformHelper:
    @staticmethod
    def get_cifar_transform(args):
        normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616)) if args.set == 'CIFAR10' else \
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        transform_train = torchvision.transforms.Compose(
            [torchvision.transforms.RandomRotation(10),
             torchvision.transforms.RandomCrop(32, padding=4),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             normalize
             ])

        transform_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                normalize])
        return transform_train, transform_test

    @staticmethod
    def get_tiny_imagenet_transform():
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(10),  # RandomRotation
            torchvision.transforms.RandomCrop(64, padding=4),
            # resize 256_comb_coteach_OpenNN_CIFAR -> random_crop 224 ==> crop 32, padding 4
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
        ])
        return transform


class Cifar10DatasetFactory(IDatasetFactory):
    """Basic representation of the CIFAR10 Dataset factory"""

    def get_dataset(self, args) -> tuple:
        """Return the Dataset class CIFAR10"""
        transform_train, transform_test = TransformHelper.get_cifar_transform()
        trainset = torchvision.datasets.CIFAR10(root=args.data, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.data, train=False,
                                               download=True, transform=transform_test)
        # classes = ('plane', 'car', 'bird', 'cat',
        #                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainset, testset


class Cifar100DatasetFactory(IDatasetFactory):
    """Basic representation of a Dataset factory"""

    def get_dataset(self, args) -> tuple:
        """Return the Dataset class CIFAR100"""
        transform_train, transform_test = TransformHelper.get_cifar_transform()
        trainset = torchvision.datasets.CIFAR100(root=args.data, train=True,
                                                 download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.data, train=False,
                                                download=True, transform=transform_test)
        return trainset, testset


class TinyImageNetDatasetFactory(IDatasetFactory):
    """Basic representation of a Dataset factory"""

    def get_dataset(self, args) -> tuple:
        """Return the Dataset class"""
        transform = TransformHelper.get_tiny_imagenet_transform()
        trainset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.data, 'tiny_imagenet'),
            split='train',
            transform=transform
        )
        testset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.data, 'tiny_imagenet'),
            split='test',
            transform=transform
        )
        return trainset, testset


class CICIDS2017DatasetFactory(IDatasetFactory):
    """Basic representation of a Dataset factory"""

    def get_dataset(self, args) -> tuple:
        """Return the CICIDS2017 Dataset class"""
        logger = setup_custom_logger('root', LOG_LEVEL[args.log_level], args.log_path)
        files_path = os.path.join(args.data, 'CICIDS2017')
        cic_ids_path = os.path.join(files_path, 'cicids2017.csv')

        if os.path.exists(files_path):
            if os.path.exists(cic_ids_path):
                if os.stat(cic_ids_path).st_size < 1474940000:
                    logger.warning("[WARN] Somthing happend while downloading CICIDS2017")
                    [os.remove(os.path.join(files_path, cur_file)) for cur_file in os.listdir(files_path)]
                    cicids_dataset.download_cicids2017(path=files_path)
                    cicids_dataset.create_cic_ids_file()
            else:
                logger.warning("CICIDS2017 file not exists.")
                cicids_dataset.download_cicids2017(path=files_path)
                cicids_dataset.create_cic_ids_file()
        else:
            os.mkdir(files_path)
            cicids_dataset.download_cicids2017(path=files_path)
            cicids_dataset.create_cic_ids_file()

        logger.info('Loading CICIDS2017 file...')
        data = pd.read_csv(cic_ids_path)
        logger.debug('Data Lenght: ' + str(len(data)))
        data['Flow Bytes/s'] = data['Flow Bytes/s'].astype(float)
        data[' Flow Packets/s'] = data[' Flow Packets/s'].astype(float)
        # Drop NAN and INF
        data = data.replace(np.inf, 1.7976931348623157e+301).replace(-np.inf, -1.7976931348623157e+301)
        data.dropna(inplace=True)

        train, test = train_test_split(data, test_size=args.test_size + args.val_size, random_state=args.seed)
        val_proportion = args.val_size / (args.test_size + args.val_size)
        val, test = train_test_split(test, test_size=val_proportion)

        trainset = cicids_dataset.CICIDS2017Dataset(train)
        testset = cicids_dataset.CICIDS2017Dataset(test)
        valset = cicids_dataset.CICIDS2017Dataset(val)

        return trainset, testset, valset


DATASETS_LOOKUP_TABLE = {
        'CIFAR10': Cifar10DatasetFactory(),
        'CIFAR100': Cifar100DatasetFactory(),
        'Tiny-ImageNet': TinyImageNetDatasetFactory(),
        'CICIDS2017': CICIDS2017DatasetFactory()
    }

NUM_CLASSES_LOOKUP_TABLE = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'Tiny-ImageNet': 200,
        'CICIDS2017': 2
}