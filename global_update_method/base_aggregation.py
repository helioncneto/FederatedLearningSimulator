# coding: utf-8

from utils import get_scheduler, get_optimizer, get_model, get_dataset, create_check_point, load_check_point
import numpy as np
import os
import sys
import time
from utils import *
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
from libs.evaluation.metrics import Evaluator
from utils.helper import save, do_evaluation, get_participant, get_filepath
from utils.malicious import add_malicious_participants
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class BaseGlobalUpdate:
    def __init__(self, args, device, trainset, testloader, local_update, experiment_name, valloader=None):
        self.args = args
        self.device = device
        self.trainset = trainset
        self.testloader = testloader
        self.local_update = local_update
        self.valloader = valloader
        self.epoch = 0
        self.loss_avg = 0
        self.total_num_of_data_clients = 1
        self.duration = []
        self.experiment_name = experiment_name

        assert os.path.isdir(self.args.eval_path)

        self.model = get_model(arch=self.args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[self.args.set],
                          l2_norm=self.args.l2_norm)
        self.model.to(self.device)
        if self.args.use_wandb:
            wandb.watch(self.model)
        self.model.train()
        self.global_weight = copy.deepcopy(self.model.state_dict())

        self.dataset = get_dataset(self.args, trainset, self.args.num_of_clients, self.args.mode)
        self.loss_train = []
        self.acc_train = []
        self.this_lr = self.args.lr
        self.this_alpha = self.args.alpha
        self.selected_participants_num = max(int(self.args.participation_rate * self.args.num_of_clients), 1)
        self.selected_participants = None
        self.FedAvg_weight = copy.deepcopy(self.model.state_dict())

        self.wandb_dict = {}
        self.num_of_data_clients = []
        self.local_weight = []
        self.local_loss = {}
        self.local_delta = []
        self.metric = {}
        self.test_metric = {}

        # Gen malicious data
        self.malicious_list = {}
        self.trainset_fake, self.dataset_fake = {}, {}
        self.malicious_participant_dataloader_table = {}
        self._set_malicious()

    def _restart_env(self):
        self.wandb_dict = {}
        self.num_of_data_clients = []
        self.local_weight = []
        self.local_loss = {}
        self.local_delta = []
        self.global_weight = copy.deepcopy(self.model.state_dict())
        self.metric = {}
        self.test_metric = {}
        self.duration = []

    def _set_malicious(self):
        if 0 < self.args.malicious_rate <= 1:
            print("=> Training with malicious participants!")
            directory, filepath = get_filepath(self.args, True)
            if self.args.malicious_type == 'random':
                self.trainset_fake, self.dataset_fake = add_malicious_participants(self.args, directory, filepath)
                for participant in self.dataset_fake.keys():
                    self.malicious_participant_dataloader_table[participant] = DataLoader(DatasetSplit(self.trainset_fake, self.dataset_fake[participant]), batch_size=self.args.batch_size, shuffle=True)
            elif self.args.malicious_type == 'fgsm':
                print("   => The malicious participants are using FGSM attack")
                mal_num = int(self.args.num_of_clients * self.args.malicious_rate)
                mal_part_ids = np.random.choice(range(self.args.num_of_clients), mal_num, replace=False)
                self.dataset_fake = {idx: [] for idx in mal_part_ids}
                self.trainset_fake = None
            elif self.args.malicious_type == 'jsma':
                print("   => The malicious participants are using JSMA attack")
            else:
                print("The malicious participant type is not defined!")
                sys.exit()
            print("Malicious participants IDS: ", list(self.dataset_fake.keys()))
            '''for participant in self.dataset_fake.keys():
                if self.args.malicious_type == 'random':
                    self.malicious_participant_dataloader_table[participant] = DataLoader(DatasetSplit(self.trainset_fake,
                                                                                                       self.dataset_fake[participant]),
                                                                                                       batch_size=self.args.batch_size, shuffle=True)'''

        elif self.args.malicious_rate > 1:
            print("The malicious rate cannot be greater than 1")
            sys.exit()
        elif self.args.malicious_rate < 0:
            print("The malicious rate cannot be negative")
            sys.exit()

    def _decay(self):
        self.this_lr *= self.args.learning_rate_decay
        if self.args.alpha_mul_epoch:
            self.this_alpha = self.args.alpha * (epoch + 1)
        elif self.args.alpha_divide_epoch:
            self.this_alpha = self.args.alpha / (epoch + 1)

    def _select_participants(self):
        if self.epoch == 1 or self.args.participation_rate < 1:
            print('Selecting the participants')
            self.selected_participants = np.random.choice(range(self.args.num_of_clients),
                                                          self.selected_participants_num,
                                                          replace=False)

    def _local_update(self):
        #for participant in tqdm(self.selected_participants, desc="Local update"):
        for participant in self.selected_participants:
            print(f"Training participant: {participant}")
            self.start_time = time.time()
            self.num_of_data_clients, idxs, current_trainset, malicious = get_participant(self.args, participant,
                                                                                          self.dataset,
                                                                                          self.dataset_fake,
                                                                                          self.num_of_data_clients,
                                                                                          self.trainset,
                                                                                          self.trainset_fake,
                                                                                          self.epoch)
            local_setting = self.local_update(args=self.args, lr=self.this_lr, local_epoch=self.args.local_epochs,
                                              device=self.device,
                                              batch_size=self.args.batch_size, dataset=current_trainset, idxs=idxs,
                                              alpha=self.this_alpha)
            self.malicious_list[participant] = malicious

            weight, loss = local_setting.train(net=copy.deepcopy(self.model).to(self.device), malicious=malicious)
            if self.args.malicious_type == 'fgsm':
                self.malicious_participant_dataloader_table[participant] = local_setting.get_dataloader()
            self.local_weight.append(copy.deepcopy(weight))
            self.local_loss[participant] = copy.deepcopy(loss)
            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - self.global_weight[key]
            self.local_delta.append(delta)
            self.end_time = time.time()
            self.duration.append(self.end_time - self.start_time)

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

    def _update_global_model(self):
        self.model.load_state_dict(self.FedAvg_weight)
        self.loss_avg = sum(self.local_loss) / len(self.local_loss)
        print(f'- Num. of data per client : {self.num_of_data_clients}')
        print(f'- Participants IDS: {self.selected_participants}')
        print(f'- Average loss {self.loss_avg}')
        print(f"Local losses: {self.local_loss}")
        self.loss_train.append(self.loss_avg)

    def _model_validation(self):
        self.model.eval()
        self.metrics = do_evaluation(testloader=self.testloader, model=self.model, device=self.device)

        print(f'Accuracy of the global model on validation set: {self.metrics["accuracy"]} %%')
        print(f'Precision of the global model on validation set: {self.metrics["precision"]} %%')
        print(f'Sensitivity of the global model on validation set: {self.metrics["sensitivity"]} %%')
        print(f'Specificity of the global model on validation set: {self.metrics["specificity"]} %%')
        print(f'F1-score of the global model on validation set: {self.metrics["f1score"]} %%')
        self.model.train()

        self.wandb_dict[self.args.mode + "_acc"] = self.metrics['accuracy']
        self.wandb_dict[self.args.mode + "_prec"] = self.metrics['precision']
        self.wandb_dict[self.args.mode + "_sens"] = self.metrics['sensitivity']
        self.wandb_dict[self.args.mode + "_spec"] = self.metrics['specificity']
        self.wandb_dict[self.args.mode + "_f1"] = self.metrics['f1score']
        self.wandb_dict[self.args.mode + '_loss'] = self.loss_avg
        self.wandb_dict['lr'] = self.this_lr
        if self.args.use_wandb:
            print('logging to wandb...')
            wandb.log(self.wandb_dict)
        save((self.args.eval_path, self.args.global_method + "_acc"), self.wandb_dict[self.args.mode + "_acc"])
        save((self.args.eval_path, self.args.global_method + "_prec"), self.wandb_dict[self.args.mode + "_prec"])
        save((self.args.eval_path, self.args.global_method + "_sens"), self.wandb_dict[self.args.mode + "_sens"])
        save((self.args.eval_path, self.args.global_method + "_spec"), self.wandb_dict[self.args.mode + "_spec"])
        save((self.args.eval_path, self.args.global_method + "_f1"), self.wandb_dict[self.args.mode + "_f1"])
        save((self.args.eval_path, self.args.global_method + "_loss"), self.wandb_dict[self.args.mode + "_loss"])

    def _model_test(self):
        if self.valloader is not None:
            self.model.eval()
            self.test_metric = do_evaluation(self.valloader, self.model, self.device)
            self.model.train()

            print(f'Final Accuracy of the global model on test set: {self.test_metric["accuracy"]} %%')
            print(f'Final Precision of the global model on test set: {self.test_metric["precision"]} %%')
            print(f'Final Sensitivity of the global model on test set: {self.test_metric["sensitivity"]} %%')
            print(f'Final Specificity of the global model on test set: {self.test_metric["specificity"]} %%')
            print(f'Final F1-score of the global model on test set: {self.test_metric["f1score"]} %%')

            save((self.args.eval_path, self.args.mode + "_test_acc"), self.test_metric['accuracy'])
            save((self.args.eval_path, self.args.mode + "_test_prec"), self.test_metric['precision'])
            save((self.args.eval_path, self.args.mode + "_test_sens"), self.test_metric['sensitivity'])
            save((self.args.eval_path, self.args.mode + "_test_spec"), self.test_metric['specificity'])
            save((self.args.eval_path, self.args.mode + "_test_f1"), self.test_metric['f1score'])

    def _saving_point(self):
        create_check_point(self.experiment_name, self.model, self.epoch+1, self.loss_train, self.malicious_list,
                           self.this_lr, self.this_alpha, self.duration)

    def _loading_point(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint['model'])
        self.epoch = checkpoint['epoch']
        self.loss_train = checkpoint['loss']
        self.malicious_list = checkpoint['malicious']
        self.this_lr = checkpoint['lr']
        self.this_alpha = checkpoint['this_alpha']
        self.duration = checkpoint['duration']

    def train(self):
        if self.args.use_checkpoint and os.path.isfile('checkpoint/' + self.experiment_name + '.pt'):
            print("=> Restarting training from the last checkpoint...")
            checkpoint = load_check_point(self.experiment_name)
            init_epoch = checkpoint['epoch']
            self._loading_point(checkpoint)
        else:
            init_epoch = 1
        for epoch in range(init_epoch, self.args.global_epochs + 1):
            self.epoch = epoch

            # Restart the environment for the next aggregation round
            self._restart_env()

            print('starting a new epoch')
            # Perform the participant selection
            self._select_participants()
            # Sample participating agents for this global round
            print(f'Aggregation Round: {self.epoch}/{self.args.global_epochs}')
            if self.selected_participants is None:
                return

            # Start the local update for each selected participant
            print('Training participants')
            self._local_update()

            # Perform the global aggregation
            self._global_aggregation()
            self._update_global_model()

            # Validade the model at each aggregation round with the validation set
            print('Performing the evaluation')
            self._model_validation()

            print('Decay LR')
            self._decay()

            print('Saving the checkpoint')
            self._saving_point()

        self._model_test()
