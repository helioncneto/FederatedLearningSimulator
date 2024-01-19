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
# import umap.umap_ as umap
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
