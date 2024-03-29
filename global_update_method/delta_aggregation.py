#!/usr/bin/env python
# coding: utf-8

# In[2]:

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import numpy as np
from torch import nn
import copy
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import DatasetSplit
from global_update_method.distcheck import check_data_distribution
import umap.umap_ as umap
from mpl_toolkits import mplot3d
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from torch.utils.data import DataLoader
# from utils import log_ConfusionMatrix_Umap, log_acc
# from utils import calculate_delta_cv,calculate_delta_variance, calculate_divergence_from_optimal,calculate_divergence_from_center
from utils import CenterUpdate
from utils import *


#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
from utils.helper import get_participant, get_filepath
from utils.malicious import add_malicious_participants
from global_update_method.base_aggregation import BaseGlobalUpdate


class DeltaGlobalUpdate(BaseGlobalUpdate):
    def __init__(self, args, device, trainset, testloader, local_update, experiment_name, valloader=None):
        super().__init__(args, device, trainset, testloader, local_update, experiment_name, valloader)
        self.this_tau = self.args.tau
        self.global_delta = copy.deepcopy(self.model.state_dict())
        self.local_K = {}
        for key in self.global_delta.keys():
            self.global_delta[key] = torch.zeros_like(self.global_delta[key])

    def _get_delta_args(self):
        return {'this_tau': self.this_tau, 'global_delta': self.global_delta}

    def _restart_env(self):
        super()._restart_env()
        self.local_K = []

    def _decay(self):
        super()._decay()
        self.this_tau *= self.args.server_learning_rate_decay

    def _global_aggregation(self):
        self.total_num_of_data_clients = sum(self.num_of_data_clients)
        self.FedAvg_weight = copy.deepcopy(self.local_weight[0])
        for key in self.FedAvg_weight.keys():
            for i in range(len(self.local_weight)):
                if i == 0:
                    self.FedAvg_weight[key] *= self.num_of_data_clients[i]
                else:
                    self.FedAvg_weight[key] += self.local_weight[i][key] * self.num_of_data_clients[i]
            self.FedAvg_weight[key] /= self.total_num_of_data_clients
        self.global_delta = copy.deepcopy(self.local_delta[0])

        for key in self.global_delta.keys():
            for i in range(len(self.local_delta)):
                if i == 0:
                    self.global_delta[key] *= self.num_of_data_clients[i] / self.local_K[i]
                else:
                    self.global_delta[key] += self.local_delta[i][key] * self.num_of_data_clients[i] / self.local_K[i]
            self.global_delta[key] = self.global_delta[key] / (-1 * self.total_num_of_data_clients * self.args.local_epochs * self.this_lr)
            global_lr = self.args.g_lr
            self.global_weight[key] = self.global_weight[key] - global_lr * self.global_delta[key]

    def _saving_point(self):
        create_check_point(self.experiment_name, self.model, self.epoch + 1, self.loss_train, self.malicious_list,
                           self.this_lr, self.this_alpha, self.duration, delta=self._get_delta_args())

    def _loading_point(self, checkpoint: dict):
        super()._loading_point(checkpoint)
        self.this_tau = checkpoint['delta']['this_tau']
        self.global_delta = checkpoint['delta']['global_delta']

    def _decay(self):
        super()._decay()
        self.this_tau *= self.args.server_learning_rate_decay


def GlobalUpdate(args, device, trainset, testloader, local_update):
    model = get_model(args)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    model.train()

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha
    global_delta = copy.deepcopy(model.state_dict())
    m = max(int(args.participation_rate * args.num_of_clients), 1)
    for key in global_delta.keys():
        global_delta[key] = torch.zeros_like(global_delta[key])

    # Gen fake data
    malicious_participant_dataloader_table = {}
    if args.malicious_rate > 0:
        directory, filepath = get_filepath(args, True)
        trainset_fake, dataset_fake = add_malicious_participants(args, directory, filepath)
        for participant in dataset_fake.keys():
            malicious_participant_dataloader_table[participant] = DataLoader(DatasetSplit(trainset_fake,
                                                                                          dataset_fake[
                                                                                              participant]),
                                                                             batch_size=args.batch_size,
                                                                             shuffle=True)
    else:
        trainset_fake, dataset_fake = {}, {}

    for epoch in range(args.global_epochs):
        wandb_dict={}
        num_of_data_clients=[]
        local_K=[]
        local_weight = []
        local_loss = []
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())
        if (epoch==0) or (args.participation_rate<1) :
            selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        else:
            pass 
        print(f"This is global {epoch} epoch")

        malicious_list = {}

        for user in selected_user:
            num_of_data_clients, idxs, current_trainset, malicious = get_participant(args, user, dataset,
                                                                                     dataset_fake, num_of_data_clients,
                                                                                     trainset, trainset_fake, epoch)
            local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                         batch_size=args.batch_size, dataset=current_trainset, idxs=idxs,
                                         alpha=this_alpha)
            malicious_list[user] = malicious

            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device), delta=global_delta)
            local_K.append(local_setting.K)
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))

            # Store local delta
            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - global_weight[key]
            local_delta.append(delta)            
            client_ldr_train = DataLoader(DatasetSplit(trainset, dataset[user]), batch_size=args.batch_size, shuffle=True)

        total_num_of_data_clients=sum(num_of_data_clients)
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(local_weight)):
                if i==0:
                    FedAvg_weight[key]*=num_of_data_clients[i]
                else:                       
                    FedAvg_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients
        global_delta = copy.deepcopy(local_delta[0])

        for key in global_delta.keys():
            for i in range(len(local_delta)):
                if i==0:
                    global_delta[key] *=num_of_data_clients[i]/local_K[i]
                else:
                    global_delta[key] += local_delta[i][key]*num_of_data_clients[i]/local_K[i]
            global_delta[key] = global_delta[key] / (-1 * total_num_of_data_clients * args.local_epochs * this_lr)
            global_lr = args.g_lr
            global_weight[key] = global_weight[key] - global_lr * global_delta[key]

        # Global weight update
        model.load_state_dict(FedAvg_weight)
        loss_avg = sum(local_loss) / len(local_loss)
        print(' num_of_data_clients : ',num_of_data_clients)                                   
        print(' Average loss {:.3f}'.format(loss_avg))
        loss_train.append(loss_avg)

        if epoch % args.print_freq == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %f %%' % (
                    100 * correct / float(total)))
            acc_train.append(100 * correct / float(total))

        model.train()
        wandb_dict[args.mode + "_acc"]=acc_train[-1]
        wandb_dict[args.mode + '_loss']= loss_avg
        wandb_dict['lr']=this_lr
        if args.use_wandb:
            wandb.log(wandb_dict)

        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch == True:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch == True:
            this_alpha = args.alpha / (epoch + 1)