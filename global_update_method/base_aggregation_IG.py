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

        for participant in range(self.args.num_of_clients):
            participant_dataset_ldr = DataLoader(DatasetSplit(self.trainset, self.dataset[participant]),
                                                 batch_size=self.args.batch_size, shuffle=True)
            self.participant_dataloader_table[participant] = participant_dataset_ldr

    def _get_fedsbs_args(self):
        return {'ig': self.ig, 'entropy': self.entropy, 'participants_score': self.participants_score,
                'not_selected_participants': self.not_selected_participants, 'ep_greedy': self.ep_greedy,
                'participants_count': self.participants_count, 'temperature': self.temperature}

    def _restart_env(self):
        super()._restart_env()
        self.global_losses = []

    def _select_participants(self):

        if self.epoch == 1 or self.args.participation_rate >= 1:
            print('Selecting the participants')
            self.selected_participants = np.random.choice(range(self.args.num_of_clients),
                                                          self.selected_participants_num,
                                                          replace=False)
            self.not_selected_participants = list(set(self.not_selected_participants) - set(self.selected_participants))
            self.participants_count = update_selection_count(self.selected_participants, self.participants_count)
        elif self.args.participation_rate < 1:
            print('PARTICIPANT SCORE: ', self.participants_score)
            self.selected_participants, self.not_selected_participants = selection_ig(self.selected_participants_num,
                                                                                      self.ep_greedy,
                                                                                      self.not_selected_participants,
                                                                                      self.participants_score,
                                                                                      self.temperature,
                                                                                      participants_count=self.participants_count)
            self.participants_count = update_selection_count(self.selected_participants, self.participants_count)
        print(' Participants IDS: ', self.selected_participants)

    def _update_global_model(self):
        super()._update_global_model()
        #self.global_losses = []
        #print("TEMPERATURA: " + str(self.temperature))
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
            print(f'=> Participant {participant} loss: {current_global_metrics["loss"]}')

        self.global_loss = sum(self.global_losses) / len(self.global_losses)
        print(f'=> Mean global loss: {self.global_loss}')
        print("TEMPERATURE: " + str(self.temperature))

    def _model_validation(self):
        super()._model_validation()
        cur_ig = calc_ig(self.global_loss, self.local_loss, self.entropy)
        self.participants_score, self.ig = update_participants_score(self.participants_score, cur_ig, self.ig,
                                                                     self.eg_momentum)
        print(f'Information Gain: {self.ig}')

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


'''def GlobalUpdate(args, device, trainset, testloader, local_update, valloader=None):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    model.train()

    dataset = get_dataset(args, trainset, args.num_of_clients, args.mode)
    print("Preparing participants evaluation datasets")
    participant_dataloader_table = {}
    for participant in range(args.num_of_clients):
        participant_dataset_ldr = DataLoader(DatasetSplit(trainset, dataset[participant]),
                                             batch_size=args.batch_size, shuffle=True)
        participant_dataloader_table[participant] = participant_dataset_ldr

    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha

    total_participants = args.num_of_clients
    selected_participants_num = max(int(args.participation_rate * total_participants), 1)
    loss_func = nn.CrossEntropyLoss()
    ig = {}
    entropy = {}
    participants_score = {idx: selected_participants_num/total_participants for idx in range(total_participants)}
    not_selected_participants = list(participants_score.keys())
    #ep_greedy = args.epsilon_greedy
    ep_greedy = 1
    ep_greedy_decay = pow(0.01, 1/args.global_epochs)
    participants_count = {participant: 0 for participant in list(participants_score.keys())}
    blocked = {}

    # Gen fake data
    malicious_participant_dataloader_table = {}
    if args.malicious_rate > 0:
        directory, filepath = get_filepath(args, True)
        trainset_fake, dataset_fake = add_malicious_participants(args, directory, filepath)
        for participant in dataset_fake.keys():
            malicious_participant_dataloader_table[participant] = DataLoader(DatasetSplit(trainset_fake,
                                                                                          dataset_fake[participant]),
                                                                             batch_size=args.batch_size, shuffle=True)
    else:
        trainset_fake, dataset_fake = {}, {}

    for epoch in range(args.global_epochs):
        print('starting a new epoch')
        wandb_dict = {}
        num_of_data_clients = []
        local_weight = []
        local_loss = {}
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())
        eg_momentum = 0.9

        # Sample participating agents for this global round
        selected_participants = None

        if len(blocked) > 0:
            parts_to_ublock = []
            for part, since in blocked.items():
                if since + int(total_participants / selected_participants_num) <= epoch:
                    parts_to_ublock.append(part)
                    participants_count[part] = 0

            [blocked.pop(part) for part in parts_to_ublock]

        if epoch == 0 or args.participation_rate >= 1:
            print('Selecting the participants')
            #selected_participants = np.random.choice(range(args.num_of_clients + selected_participants_fake_num),
                                                     #selected_participants_num,
                                                     #replace=False)
            selected_participants = np.random.choice(range(args.num_of_clients),
                                                     selected_participants_num,
                                                     replace=False)
            not_selected_participants = list(set(not_selected_participants) - set(selected_participants))
        elif args.participation_rate < 1:
            selected_participants, not_selected_participants = selection_ig(selected_participants_num, ep_greedy,
                                                                            not_selected_participants,
                                                                            participants_score,
                                                                            args.temperature,
                                                                            participants_count=participants_count)
        print(' Participants IDS: ', selected_participants)
        print(f'Aggregation Round: {epoch}')
        if selected_participants is None:
            return
        print('Training participants')
        #models_val_loss = {}
        malicious_list = {}

        for participant in selected_participants:
            #if participant < args.num_of_clients:
            num_of_data_clients, idxs, current_trainset, malicious = get_participant(args, participant, dataset,
                                                                                     dataset_fake, num_of_data_clients,
                                                                                     trainset, trainset_fake, epoch)
            local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                         batch_size=args.batch_size, dataset=current_trainset, idxs=idxs,
                                         alpha=this_alpha)
            malicious_list[participant] = malicious

            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss[participant] = copy.deepcopy(loss)

            local_model = copy.deepcopy(model).to(device)
            local_model.load_state_dict(weight)
            local_model.eval()

            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - global_weight[key]
            local_delta.append(delta)

        total_num_of_data_clients = sum(num_of_data_clients)
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(local_weight)):
                if i == 0:
                    FedAvg_weight[key] *= num_of_data_clients[i]
                else:
                    FedAvg_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients
        model.load_state_dict(FedAvg_weight)

        loss_avg = sum(local_loss) / len(local_loss)
        print(' num_of_data_clients : ', num_of_data_clients)
        print(' Participants IDS: ', selected_participants)
        print(' Average loss {:.3f}'.format(loss_avg))
        print(f"Local losses: {local_loss}")
        loss_train.append(loss_avg)
        # print(models_val_loss)

        # Calculating the Global loss
        global_losses = []
        model.eval()

        for participant in selected_participants:
            participant_dataset_loader = get_participant_loader(participant, malicious_list,
                                                                participant_dataloader_table,
                                                                malicious_participant_dataloader_table)
            if participant in entropy.keys():
                current_global_metrics = do_evaluation(testloader=participant_dataset_loader, model=model,
                                                       device=device, evaluate=False)
            else:
                current_global_metrics = do_evaluation(testloader=participant_dataset_loader, model=model,
                                                       device=device, evaluate=False, calc_entropy=True)
                entropy[participant] = current_global_metrics['entropy']
            global_losses.append(current_global_metrics['loss'])
            print(f'=> Participant {participant} loss: {current_global_metrics["loss"]}')

        global_loss = sum(global_losses) / len(global_losses)
        print(f'=> Mean global loss: {global_loss}')

        global_loss = sum(global_losses) / len(global_losses)

        print('performing the evaluation')
        model.eval()
        metrics = do_evaluation(testloader=testloader, model=model, device=device)
        model.train()
        #cur_ig = calc_ig(metrics['loss'], models_val_loss, total_num_of_data_clients, num_of_data_clients)
        cur_ig = calc_ig(global_loss, local_loss, entropy)

        participants_score, ig = update_participants_score(participants_score, cur_ig, ig, eg_momentum)

        print(participants_score)


        if epoch % args.print_freq == 0:
            print('Loss of the network on the 10000 test images: %f %%' % metrics['loss'])
            print('Accuracy of the network on the 10000 test images: %f %%' % metrics['accuracy'])
            print('Precision of the network on the 10000 test images: %f %%' % metrics['precision'])
            print('Sensitivity of the network on the 10000 test images: %f %%' % metrics['sensitivity'])
            print('Specificity of the network on the 10000 test images: %f %%' % metrics['specificity'])
            print('F1-score of the network on the 10000 test images: %f %%' % metrics['f1score'])
            print(f'Information Gain: {ig}')
            #print(f'Participants loss: {models_val_loss}')


        wandb_dict[args.mode + "_acc"] = metrics['accuracy']
        wandb_dict[args.mode + "_prec"] = metrics['precision']
        wandb_dict[args.mode + "_sens"] = metrics['sensitivity']
        wandb_dict[args.mode + "_spec"] = metrics['specificity']
        wandb_dict[args.mode + "_f1"] = metrics['f1score']
        wandb_dict[args.mode + '_loss'] = loss_avg
        wandb_dict['lr'] = this_lr
        if args.use_wandb:
            print('logging to wandb...')
            wandb.log(wandb_dict)
        save((args.eval_path, args.global_method + "_acc"), wandb_dict[args.mode + "_acc"] )
        save((args.eval_path, args.global_method + "_prec"), wandb_dict[args.mode + "_prec"])
        save((args.eval_path, args.global_method + "_sens"), wandb_dict[args.mode + "_sens"])
        save((args.eval_path, args.global_method + "_spec"), wandb_dict[args.mode + "_spec"])
        save((args.eval_path, args.global_method + "_f1"), wandb_dict[args.mode + "_f1"])
        save((args.eval_path, args.global_method + "_loss"), wandb_dict[args.mode + "_loss"])
        print('Decay LR...')
        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch:
            this_alpha = args.alpha / (epoch + 1)

        if epoch != 0:
            print("Decay Epsilon...")
            ep_greedy *= ep_greedy_decay

    if valloader is not None:
        model.eval()
        test_metric = do_evaluation(valloader, model, device)
        model.train()

        print('Final Accuracy of the network on the 10000 test images: %f %%' % test_metric['accuracy'])
        print('Final Precision of the network on the 10000 test images: %f %%' % test_metric['precision'])
        print('Final Sensitivity of the network on the 10000 test images: %f %%' % test_metric['sensitivity'])
        print('Final Specificity of the network on the 10000 test images: %f %%' % test_metric['specificity'])
        print('Final F1-score of the network on the 10000 test images: %f %%' % test_metric['f1score'])
        
        save((args.eval_path, args.global_method + "_test_acc"), test_metric['accuracy'])
        save((args.eval_path, args.global_method + "_test_prec"), test_metric['precision'])
        save((args.eval_path, args.global_method + "_test_sens"), test_metric['sensitivity'])
        save((args.eval_path, args.global_method + "_test_spec"), test_metric['specificity'])
        save((args.eval_path, args.global_method + "_test_f1"), test_metric['f1score'])
        save((args.eval_path, args.global_method + "_ig"), ig)
'''