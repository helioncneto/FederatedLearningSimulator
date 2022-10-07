import copy
import os
from collections import Counter
from functools import reduce

import numpy as np
import models
import torch.optim as optim
import torch
import torch.nn as nn

from libs.evaluation.metrics import Evaluator

__all__ = ['l2norm', 'count_label_distribution', 'check_data_distribution', 'check_data_distribution_aug',
           'feature_extractor', 'classifier', 'get_model', 'get_optimizer', 'get_scheduler', 'save', 'shuffle',
           'do_evaluation']




def l2norm(x,y):
    z= (((x-y)**2).sum())
    return z/(1+len(x))


class feature_extractor(nn.Module):
            def __init__(self,model,classifier_index=-1):
                super(feature_extractor, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(model.children())[:classifier_index]
                )
            def forward(self, x):
                x = self.features(x)
                return x


class classifier(nn.Module):
            def __init__(self,model,classifier_index=-1):
                super(classifier, self).__init__()
                self.layers = nn.Sequential(
                    # stop at conv4
                    *list(model.children())[classifier_index:]
                )
            def forward(self, x):
                x = self.layers(x)
                return x


def count_label_distribution(labels,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,label in enumerate(labels):
        data_distribution[label]+=1 
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution


def check_data_distribution(dataloader,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,(images,target) in enumerate(dataloader):
        for i in target:
            data_distribution[i]+=1 
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution


def check_data_distribution_aug(dataloader,class_num:int=10,default_dist:torch.tensor=None):
    if default_dist!=None:
        default=default_dist
    else:
        default=torch.zeros(class_num)
    data_distribution=default
    for idx,(images, _, target) in enumerate(dataloader):
        for i in target:
            data_distribution[i]+=1
    data_distribution=data_distribution/data_distribution.sum()
    return data_distribution


# TODO: Hardcoded Change it
'''def get_numclasses(dataset: str) -> int:
    if dataset in ['CIFAR10', "MNIST"]:
        num_classes = 10
    elif dataset in ["CIFAR100"]:
        num_classes = 100
    elif dataset in ["Tiny-ImageNet"]:
        num_classes = 200
    elif dataset in ["CICIDS2017"]:
        num_classes = 2
    else:
        raise Exception("The dataset specified is not available")
    return num_classes'''


def get_model(arch, num_classes, l2_norm):
    #num_classes = get_numclasses(args)
    print("=> Creating model '{}'".format(arch))
    model = models.__dict__[arch](num_classes=num_classes, l2_norm=l2_norm)
    return model


def get_optimizer(args, parameters):
    if args.set == 'CIFAR10':
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set == "MNIST":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.set == "CIFAR100":
        optimizer = optim.SGD(parameters, lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Invalid mode")
        return
    return optimizer


def save(path_tuple: tuple, metric: float) -> None:
    if len(path_tuple) == 1:
        path = path_tuple[0]
    else:
        path = os.path.join(*path_tuple)
    exists = False
    if os.path.exists(path):
        if os.stat(path).st_size > 0:
            exists = True
    file = open(path, 'a')
    if exists:
        file.write(',')
    file.write(str(metric))
    file.close()


def shuffle(arr: np.array) -> np.array:
    np.random.shuffle(arr)
    return arr


def do_evaluation(testloader, model, device: int, evaluate: bool = True, calc_entropy=False) -> dict:
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    batch_loss = []
    balance = Counter()
    print(f"DEVICE: {device}")
    with torch.no_grad():
        preds = np.array([])
        full_lables = np.array([])
        first = True
        for x, labels in testloader:
            x, labels = x.to(device), labels.to(device)
            if calc_entropy:
                for label in labels:
                    balance[label] += 1
            outputs = model(x)
            val_loss = loss_func(outputs, labels)
            batch_loss.append(val_loss.item())
            top_p, top_class = outputs.topk(1, dim=1)
            if torch.cuda.is_available():
                top_class = top_class.cpu()
                labels = labels.cpu()

            if first:
                preds = top_class.numpy()
                full_lables = copy.deepcopy(labels)
                first = False
            else:
                preds = np.concatenate((preds, top_class.numpy()))
                full_lables = np.concatenate((full_lables, labels))

        loss_avg = (sum(batch_loss) / len(batch_loss))

        #for cl, qtd in balance.items():

        #entropy = -(a * np.log2(a))
        if calc_entropy:
            p_s = [value/sum(balance.values()) for value in balance.values()]
            entropy = reduce(lambda a, b: -(a * np.log2(a) + b * np.log2(b)), p_s)

    if evaluate:
        print('calculating avg accuracy')
        evaluator = Evaluator('accuracy', 'precision', 'sensitivity', 'specificity', 'f1score')
        metrics = evaluator.run_metrics(preds, full_lables)
        metrics['loss'] = loss_avg
        model.train()
    else:
        metrics = {'loss': loss_avg}

    if calc_entropy:
        metrics['entropy'] = entropy
    return metrics


def get_scheduler(optimizer, args):
    if args.set=='CIFAR10':

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                         lr_lambda=lambda epoch: args.learning_rate_decay ** epoch,
        #                         )
    elif args.set=="MNIST":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                         lr_lambda=lambda epoch: args.learning_rate_decay ** (int(epoch/50)),
        #                         )
    elif args.set=="CIFAR100":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                         lr_lambda=lambda epoch: args.learning_rate_decay ** (int(epoch/50)),
        #                         )
    else:
        print("Invalid mode")
        return
    return scheduler