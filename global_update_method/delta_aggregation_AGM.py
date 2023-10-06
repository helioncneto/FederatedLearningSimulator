# coding: utf-8
import os

from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
import numpy as np

from libs.evaluation.metrics import Evaluator
from utils import *
from utils.helper import save, do_evaluation, get_participant, get_filepath
from utils.malicious import add_malicious_participants
from torch.utils.data import DataLoader, TensorDataset
from global_update_method.base_aggregation import BaseGlobalUpdate


class FedAGMGlobalUpdate(BaseGlobalUpdate):
    def __init__(self, args, device, trainset, testloader, local_update, experiment_name, valloader=None):
        super().__init__(args, device, trainset, testloader, local_update, experiment_name, valloader)
        self.this_tau = self.args.tau
        self.global_delta = copy.deepcopy(self.model.state_dict())
        for key in self.global_delta.keys():
            self.global_delta[key] = torch.zeros_like(self.global_delta[key])

    def _restart_env(self):
        super()._restart_env()
        self.local_K = []

    def _decay(self):
        super()._decay()
        self.this_tau *= self.args.server_learning_rate_decay

    def _global_aggregation(self):
        self.sending_model_dict = copy.deepcopy(self.model.state_dict())
        for key in self.global_delta.keys():
            self.sending_model_dict[key] += -1 * self.args.lamb * self.global_delta[key]

        self.sending_model = copy.deepcopy(self.model)
        self.sending_model.load_state_dict(self.sending_model_dict)

        total_num_of_data_clients = sum(self.num_of_data_clients)
        FedAvg_weight = copy.deepcopy(self.local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(self.local_weight)):
                if i == 0:
                    FedAvg_weight[key] *= self.num_of_data_clients[i]
                else:
                    FedAvg_weight[key] += self.local_weight[i][key] * self.num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients
            FedAvg_weight[key] = FedAvg_weight[key] * self.this_tau + (1 - self.this_tau) * self.sending_model_dict[key]
        self.global_delta = copy.deepcopy(self.local_delta[0])

        for key in self.global_delta.keys():
            for i in range(len(self.local_delta)):
                if i == 0:
                    self.global_delta[key] *= self.num_of_data_clients[i]
                else:
                    self.global_delta[key] += self.local_delta[i][key] * self.num_of_data_clients[i]
            self.global_delta[key] = self.global_delta[key] / (-1 * total_num_of_data_clients)
