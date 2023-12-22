# coding: utf-8
from typing import Tuple
from libs.methods.ig import selection_ig, update_participants_score, calc_ig, update_selection_count
from utils import get_scheduler, get_optimizer, get_model, get_dataset
import numpy as np
from utils import *
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
from torch.utils.data import DataLoader, TensorDataset
from utils.helper import save, shuffle, do_evaluation, get_participant, get_participant_loader, get_filepath

from utils.malicious import add_malicious_participants
from global_update_method.base_aggregation import BaseGlobalUpdate


class FedSBSGlobalUpdate(BaseGlobalUpdate):
    def __init__(self, args, device, trainset, testloader, local_update, experiment_name, valloader=None):
        super().__init__(args, device, trainset, testloader, local_update, experiment_name, valloader)

        self.global_losses = []
        self.total_participants = self.args.num_of_clients
        self.participant_dataloader_table = {}
        self.ig = {}
        self.entropy = {}
        '''self.participants_score = {idx: self.selected_participants_num / self.total_participants for idx in
                                   range(self.total_participants)}'''
        self.participants_score = {idx: -np.inf for idx in range(self.total_participants)}
        self.not_selected_participants = list(self.participants_score.keys())
        self.ep_greedy = 1
        self.ep_greedy_decay = pow(0.01, 1 / self.args.global_epochs)
        self.participants_count = {participant: 0 for participant in list(self.participants_score.keys())}
        # self.blocked = {}
        self.eg_momentum = 0.9
        self.temperature = args.temperature
        self.cool = args.cool
        self._get_dataloader_lookup_table()

    def _get_dataloader_lookup_table(self):
        for participant in range(self.args.num_of_clients):
            participant_dataset_ldr = DataLoader(DatasetSplit(self.trainset, self.dataset[participant]),
                                                 batch_size=self.args.batch_size, shuffle=True)
            self.participant_dataloader_table[participant] = participant_dataset_ldr
        print(f"AQUII: {self.participant_dataloader_table.keys}")

    def _get_fedsbs_args(self):
        return {'ig': self.ig, 'entropy': self.entropy, 'participants_score': self.participants_score,
                'not_selected_participants': self.not_selected_participants, 'ep_greedy': self.ep_greedy,
                'participants_count': self.participants_count, 'temperature': self.temperature}

    def _restart_env(self):
        super()._restart_env()
        self.global_losses = []

    def _select_participants(self):

        if self.epoch == 1 or self.args.participation_rate >= 1:
            self.logger.debug('Selecting the participants')
            self.selected_participants = np.random.choice(range(self.args.num_of_clients),
                                                          self.selected_participants_num,
                                                          replace=False)
            self.not_selected_participants = list(set(self.not_selected_participants) - set(self.selected_participants))
            self.participants_count = update_selection_count(self.selected_participants, self.participants_count)
        elif self.args.participation_rate < 1:
            self.logger.debug(f'PARTICIPANT SCORE: {self.participants_score}', )
            self.selected_participants, self.not_selected_participants = selection_ig(self.selected_participants_num,
                                                                                      self.ep_greedy,
                                                                                      self.not_selected_participants,
                                                                                      self.participants_score,
                                                                                      self.temperature,
                                                                                      participants_count=self.participants_count)
            self.participants_count = update_selection_count(self.selected_participants, self.participants_count)
        self.logger.debug(f' Participants IDS: {self.selected_participants}')

    def _update_global_model(self):
        super()._update_global_model()
        self.model.eval()

        for participant in self.selected_participants:
            participant_dataset_loader = get_participant_loader(participant, self.malicious_list,
                                                                self.participant_dataloader_table,
                                                                self.malicious_participant_dataloader_table)
            if participant in self.entropy.keys():
                current_global_metrics = do_evaluation(testloader=participant_dataset_loader, model=self.model,
                                                       device=self.device, evaluate=False)
            else:
                current_global_metrics = do_evaluation(testloader=participant_dataset_loader, model=self.model,
                                                       device=self.device, evaluate=False, calc_entropy=True)
                self.entropy[participant] = current_global_metrics['entropy']
            self.global_losses.append(current_global_metrics['loss'])
            self.logger.debug(f'=> Participant {participant} loss: {current_global_metrics["loss"]}')

        self.global_loss = sum(self.global_losses) / len(self.global_losses)
        self.logger.debug(f'=> Mean global loss: {self.global_loss}')
        self.logger.debug("TEMPERATURE: " + str(self.temperature))

    def _model_validation(self):
        super()._model_validation()
        cur_ig = calc_ig(self.global_loss, self.local_loss, self.entropy)
        self.participants_score, self.ig = update_participants_score(self.participants_score, cur_ig, self.ig,
                                                                     self.eg_momentum)
        self.logger.debug(f'Information Gain: {self.ig}')

    def _decay(self):
        super()._decay()
        self.ep_greedy *= self.ep_greedy_decay
        self.temperature *= self.cool

    def _saving_point(self):
        create_check_point(self.experiment_name, self.model, self.epoch + 1, self.loss_train, self.malicious_list,
                           self.this_lr, self.this_alpha, self.duration, fedsbs=self._get_fedsbs_args())

    def _loading_point(self, checkpoint: dict):
        super()._loading_point(checkpoint)
        self.ig = checkpoint['fesbs']['ig']
        self.entropy = checkpoint['fesbs']['entropy']
        self.participants_score = checkpoint['fesbs']['participants_score']
        self.not_selected_participants = checkpoint['fesbs']['not_selected_participants']
        self.ep_greedy = checkpoint['fesbs']['ep_greedy']
        self.participants_count = checkpoint['fesbs']['participants_count']
        self.temperature = checkpoint['fesbs']['temperature']
