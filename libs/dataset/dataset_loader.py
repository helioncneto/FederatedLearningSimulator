import os
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import torchvision

import datasets as local_datasets
from args_dir.federated import args
from libs.dataset.cicids import *


class IDatasetFactory(ABC):
    """Basic representation of a Dataset factory"""

    @abstractmethod
    def get_dataset(self) -> tuple:
        """Return the Dataset class"""


class TransformHelper:
    @staticmethod
    def get_cifar_transform():
        normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616)) if args.set == 'CIFAR10' else transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

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

    def get_dataset(self) -> tuple:
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

    def get_dataset(self) -> tuple:
        """Return the Dataset class CIFAR100"""
        transform_train, transform_test = TransformHelper.get_cifar_transform()
        trainset = torchvision.datasets.CIFAR100(root=args.data, train=True,
                                                 download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.data, train=False,
                                                download=True, transform=transform_test)
        return trainset, testset


class TinyImageNetDatasetFactory(IDatasetFactory):
    """Basic representation of a Dataset factory"""

    def get_dataset(self) -> tuple:
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

    def get_dataset(self) -> tuple:
        """Return the CICIDS2017 Dataset class"""
        files_path = os.path.join(args.data, 'CICIDS2017')
        cic_ids_path = os.path.join(files_path, 'cicids2017.csv')

        if os.path.exists(files_path):
            if os.path.exists(cic_ids_path):
                if os.stat(cic_ids_path).st_size < 1474940000:
                    print("[WARN] Somthing happend while downloading CICIDS2017")
                    [os.remove(os.path.join(files_path, cur_file)) for cur_file in os.listdir(files_path)]
                    download_cicids2017(path=files_path)
                    create_cic_ids_file()
            else:
                print("CICIDS2017 file not exists.")
                download_cicids2017(path=files_path)
                create_cic_ids_file()
        else:
            os.mkdir(files_path)
            download_cicids2017(path=files_path)
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

        return trainset, testset