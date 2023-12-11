# coding: utf-8

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import numpy as np
from utils import *
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
from utils.helper import save, do_evaluation, get_participant, get_filepath
from utils.malicious import add_malicious_participants
from torch.utils.data import DataLoader, TensorDataset
from global_update_method.base_aggregation import BaseGlobalUpdate


#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class ReputationGlobalUpdate(BaseGlobalUpdate):
    def __init__(self, args, device, trainset, testloader, local_update, experiment_name, valloader=None):
        super().__init__(args, device, trainset, testloader, local_update, experiment_name, valloader)
        self.all_participants = np.arange(args.num_of_clients)
        self.reputation = {}
        self.global_model_rep = {}
        self.global_metrics = []
        self.local_model = copy.deepcopy(self.model).to(self.device)

    def _get_reputation_args(self):
        return {'reputation': self.reputation, 'global_metrics': self.global_metrics}

    def _select_participants(self):
        self.selected_participants = {}

    def _local_update(self):
        for participant in self.all_participants:
            '''num_of_data_clients.append(len(dataset[participant]))
            local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                         batch_size=args.batch_size, dataset=trainset, idxs=dataset[participant],
                                         alpha=this_alpha)'''

            self.num_of_data_clients, idxs, current_trainset, malicious = get_participant(self.args, participant, self.dataset,
                                                                                     self.dataset_fake, self.num_of_data_clients,
                                                                                     self.trainset, self.trainset_fake, self.epoch)
            local_setting = self.local_update(args=self.args, lr=self.this_lr, local_epoch=self.args.local_epochs,
                                              device=self.device,
                                              batch_size=self.args.batch_size, dataset=current_trainset, idxs=idxs,
                                              alpha=self.this_alpha)
            self.malicious_list[participant] = malicious

            weight, loss = local_setting.train(net=copy.deepcopy(self.model).to(self.device))
            self.local_weight.append(copy.deepcopy(weight))
            self.local_loss[participant] = copy.deepcopy(loss)

            self.local_model = copy.deepcopy(self.model).to(self.device)
            self.local_model.load_state_dict(weight)
            self.local_model.eval()
            local_metric = do_evaluation(self.valloader, self.local_model, self.device)
            self.local_model.train()

            if participant not in self.reputation.keys():
                self.reputation[participant] = {}

            self.reputation[participant][self.epoch] = {}
            self.reputation[participant][self.epoch]["accuracy"] = local_metric['accuracy']

            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - self.global_weight[key]
            self.local_delta.append(delta)

    def _update_global_model(self):
        self.logger.debug('performing the temporary evaluation')
        self.model.eval()
        metrics_temp = do_evaluation(testloader=self.testloader, model=self.model, device=self.device)
        self.model.train()
        # accuracy = (accuracy / len(testloader)) * 100

        ### Calcula quem vai agregar
        sum_rep = [self.reputation[part][self.epoch]["accuracy"] for part in self.all_participants]
        sum_rep = sum(sum_rep)
        avg = sum_rep / len(self.all_participants)
        for participant in self.all_participants:
            if self.epoch == 1:
                self.reputation[participant][self.epoch]["score"] = (self.reputation[participant][self.epoch]["accuracy"] - avg) + (
                        self.reputation[participant][self.epoch]["accuracy"] - metrics_temp['accuracy'])
            else:
                self.reputation[participant][self.epoch]["score"] = (self.reputation[participant][self.epoch]["accuracy"] - avg) + (
                        self.reputation[participant][self.epoch]["accuracy"] - metrics_temp['accuracy']) + (
                                                                  self.reputation[participant][self.epoch]["accuracy"] -
                                                                  self.global_metrics[self.epoch - 2])

            if self.reputation[participant][self.epoch]["score"] < 0:
                self.reputation[participant][self.epoch]["selected"] = False
            else:
                self.reputation[participant][self.epoch]["selected"] = True

        self.selected_participants = []
        for part in self.all_participants:
            if self.reputation[part][self.epoch]["selected"]:
                self.selected_participants.append(part)

        if len(self.selected_participants) < self.selected_participants_num:
            missing_to_select = self.selected_participants_num - len(self.selected_participants)
            to_select = np.array(list(set(self.all_participants) - set(self.selected_participants)))
            selected_random = np.random.choice(to_select, missing_to_select, replace=False)
            self.selected_participants = np.concatenate((self.selected_participants, selected_random), axis=0).astype(int)

        self.FedAvg_weight = copy.deepcopy(self.local_weight[self.selected_participants[0]])
        for key in self.FedAvg_weight.keys():
            for i in self.selected_participants:
                if i == self.selected_participants[0]:
                    self.FedAvg_weight[key] *= self.num_of_data_clients[i]
                else:
                    self.FedAvg_weight[key] += self.local_weight[i][key] * self.num_of_data_clients[i]
            self.FedAvg_weight[key] /= self.total_num_of_data_clients
        self.model.load_state_dict(self.FedAvg_weight)

        selected_loss = [self.local_loss[selected] for selected in self.selected_participants]
        loss_avg = sum(selected_loss) / len(selected_loss)
        self.logger.debug(f' num_of_data_clients :  {[self.num_of_data_clients[selected] for selected in self.selected_participants]}')
        self.logger.debug(f' Participants IDS: {self.selected_participants}')
        self.logger.debug(' Average loss {:.3f}'.format(loss_avg))
        self.loss_train.append(loss_avg)

    def _model_validation(self):
        super()._model_validation()
        self.global_metrics.append(self.metrics['accuracy'])

    def _saving_point(self):
        create_check_point(self.experiment_name, self.model, self.epoch + 1, self.loss_train, self.malicious_list,
                           self.this_lr, self.this_alpha, self.duration, reputation=self._get_reputation_args())

    def _loading_point(self, checkpoint: dict):
        super()._loading_point(checkpoint)
        self.reputation = checkpoint['reputation']['reputation']
        self.global_metrics = checkpoint['reputation']['global_metrics']
