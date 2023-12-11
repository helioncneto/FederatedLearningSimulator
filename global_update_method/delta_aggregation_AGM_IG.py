# coding: utf-8
import os

from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
import numpy as np

from libs.evaluation.metrics import Evaluator
from libs.methods.ig import selection_ig, calc_ig, update_participants_score
from utils import *
from utils.helper import save, do_evaluation, get_participant, get_filepath, \
    get_participant_loader
from torch.utils.data import DataLoader, TensorDataset
from utils.malicious import add_malicious_participants
from global_update_method.base_aggregation_IG import FedSBSGlobalUpdate


class DeltaFedSBSGlobalUpdate(FedSBSGlobalUpdate):
    def __init__(self, args, device, trainset, testloader, local_update, experiment_name, valloader=None):
        super().__init__(args, device, trainset, testloader, local_update, experiment_name, valloader)
        self.this_tau = args.tau
        self.global_delta = copy.deepcopy(self.model.state_dict())
        self.m = max(int(args.participation_rate * args.num_of_clients), 1)
        for key in self.global_delta.keys():
            self.global_delta[key] = torch.zeros_like(self.global_delta[key])
        self.local_K = []
        self.sending_model = copy.deepcopy(self.model)
        self.sending_model_dict = copy.deepcopy(self.model.state_dict())

    def _get_delta_args(self):
        return {'this_tau': self.this_tau, 'global_delta': self.global_delta,
                'sending_model_dict': self.sending_model_dict}

    def _restart_env(self):
        super()._restart_env()
        self.local_K = []

    def _decay(self):
        super()._decay()
        self.this_tau *= self.args.server_learning_rate_decay

    def _local_update(self):
        for participant in self.selected_participants:
            logger.debug(f"Training participant: {participant}")
            self.num_of_data_clients, idxs, current_trainset, malicious = get_participant(self.args, participant,
                                                                                     self.dataset,
                                                                                     self.dataset_fake,
                                                                                     self.num_of_data_clients,
                                                                                     self.trainset,
                                                                                     self.trainset_fake, self.epoch)
            local_setting = self.local_update(args=self.args, lr=self.this_lr, local_epoch=self.args.local_epochs,
                                              device=self.device, batch_size=self.args.batch_size,
                                              dataset=current_trainset, idxs=idxs, alpha=self.this_alpha)
            self.malicious_list[participant] = malicious

            weight, loss = local_setting.train(copy.deepcopy(self.sending_model).to(self.device), self.epoch)
            # Novos maliciosos
            if self.args.malicious_type == 'fgsm':
                self.malicious_participant_dataloader_table[participant] = local_setting.get_dataloader()
            self.local_K.append(local_setting.K)
            self.local_weight.append(copy.deepcopy(weight))
            self.local_loss[participant] = copy.deepcopy(loss)

            # Store local delta
            delta = {}
            for key in weight.keys():
                delta[key] = self.this_tau*weight[key]+(1-self.this_tau)*self.sending_model_dict[key] - self.global_weight[key]
            self.local_delta.append(delta)

    def _global_aggregation(self):
        self.sending_model_dict = copy.deepcopy(self.model.state_dict())
        for key in self.global_delta.keys():
            self.sending_model_dict[key] += -1 * self.args.lamb * self.global_delta[key]

        self.sending_model = copy.deepcopy(self.model)
        self.sending_model.load_state_dict(self.sending_model_dict)
        self.total_num_of_data_clients = sum(self.num_of_data_clients)
        self.FedAvg_weight = copy.deepcopy(self.local_weight[0])
        for key in self.FedAvg_weight.keys():
            for i in range(len(self.local_weight)):
                if i == 0:
                    self.FedAvg_weight[key] *= self.num_of_data_clients[i]
                else:
                    self.FedAvg_weight[key] += self.local_weight[i][key] * self.num_of_data_clients[i]
            self.FedAvg_weight[key] /= self.total_num_of_data_clients
            self.FedAvg_weight[key] = self.FedAvg_weight[key] * self.this_tau + (1-self.this_tau) * self.sending_model_dict[key]
        self.global_delta = copy.deepcopy(self.local_delta[0])

        for key in self.global_delta.keys():
            for i in range(len(self.local_delta)):
                if i == 0:
                    self.global_delta[key] *= self.num_of_data_clients[i]
                else:
                    self.global_delta[key] += self.local_delta[i][key] * self.num_of_data_clients[i]
            self.global_delta[key] = self.global_delta[key] / (-1 * self.total_num_of_data_clients)

    def _saving_point(self):
        create_check_point(self.experiment_name, self.model, self.epoch + 1, self.loss_train, self.malicious_list,
                           self.this_lr, self.this_alpha, self.duration, fedsbs=self._get_fedsbs_args(),
                           delta=self._get_delta_args())

    def _loading_point(self, checkpoint: dict):
        super()._loading_point(checkpoint)
        self.this_tau = checkpoint['delta']['this_tau']
        self.global_delta = checkpoint['delta']['global_delta']
        self.sending_model_dict = checkpoint['delta']['sending_model_dict']
        self.sending_model.load_state_dict(self.sending_model_dict)
