# coding: utf-8
import time
from random import random
from typing import Tuple
from libs.methods.ig import selection_ig, update_participants_score, calc_ig
from utils import get_scheduler, get_optimizer, get_model, get_dataset
import numpy as np
from utils import *
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
from torch.utils.data import DataLoader, TensorDataset
from utils.helper import save, shuffle, do_evaluation, get_participant, get_filepath
from utils.malicious import add_malicious_participants
from libs.methods.Oort import create_training_selector
from libs.config.config import load_yaml_conf
from global_update_method.base_aggregation import BaseGlobalUpdate


class OortGlobalUpdate(BaseGlobalUpdate):
    def __init__(self, args, device, trainset, testloader, local_update, experiment_name, valloader=None):
        super().__init__(args, device, trainset, testloader, local_update, experiment_name, valloader)
        self.oort_args = load_yaml_conf("libs/config/oort_config.yaml")
        self.selector = create_training_selector(self.oort_args)
        # initial_score = {'reward': 0, 'duration':0}

        # Registering participants
        [self.selector.register_client(participant, {'reward': random(), 'duration': 0}) for participant in
         range(self.args.num_of_clients)]

    def _select_participants(self):
        if self.epoch == 1 or self.args.participation_rate >= 1:
            self.logger.debug('Selecting the participants')
            self.selected_participants = np.random.choice(range(self.args.num_of_clients),
                                                          self.selected_participants_num, replace=False)
        else:
            self.selected_participants = self.selector.select_participant(self.args.num_of_clients)[:self.selected_participants_num]

    def _model_validation(self):
        super()._model_validation()
        for idx, participant in enumerate(self.selected_participants):
            self.selector.update_client_util(participant,
                                        {'reward': self.num_of_data_clients[idx] * np.sqrt(self.local_loss[participant] ** 2),
                                         'duration': self.duration[idx],
                                         'time_stamp': self.epoch,
                                         'status': True})
        self.selector.nextRound()

    def _saving_point(self):
        create_check_point(self.experiment_name, self.model, self.epoch + 1, self.loss_train, self.malicious_list,
                           self.this_lr, self.this_alpha, self.duration, selector=self.selector)

    def _loading_point(self, checkpoint: dict):
        super()._loading_point(checkpoint)
        self.selector.load_state_dict(checkpoint['oort'])
