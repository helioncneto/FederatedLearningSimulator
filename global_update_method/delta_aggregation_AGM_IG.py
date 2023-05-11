# coding: utf-8
import os

from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
import numpy as np

from libs.evaluation.metrics import Evaluator
from libs.methods.ig import selection_ig, calc_ig, update_participants_score
from utils import *
from utils.helper import save, do_evaluation, add_malicious_participants, get_participant, get_filepath, \
    get_participant_loader
from torch.utils.data import DataLoader, TensorDataset


def GlobalUpdate(args, device, trainset, testloader, local_update, valloader=None):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    model.train()
    isCICIDS2017 = True if args.mode == "CICIDS2017" else False

    participant_dataloader_table = {}
    dataset = get_dataset(args, trainset, args.num_of_clients, args.mode)
    for participant in range(args.num_of_clients):
        participant_dataset_ldr = DataLoader(DatasetSplit(trainset, dataset[participant]),
                                             batch_size=args.batch_size, shuffle=True)
        participant_dataloader_table[participant] = participant_dataset_ldr
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha
    this_tau = args.tau
    global_delta = copy.deepcopy(model.state_dict())
    m = max(int(args.participation_rate * args.num_of_clients), 1)
    for key in global_delta.keys():
        global_delta[key] = torch.zeros_like(global_delta[key])

    total_participants = args.num_of_clients
    selected_participants_num = max(int(args.participation_rate * total_participants), 1)
    loss_func = nn.CrossEntropyLoss()
    ig = {}
    entropy = {}
    participants_score = {idx: selected_participants_num / total_participants for idx in range(total_participants)}
    not_selected_participants = list(participants_score.keys())
    # ep_greedy = args.epsilon_greedy
    ep_greedy = 1
    ep_greedy_decay = pow(0.01, 1 / args.global_epochs)
    participants_count = {participant: 0 for participant in list(participants_score.keys())}
    blocked = {}

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
        print('starting a new epoch')
        wandb_dict = {}
        num_of_data_clients = []
        local_K = []

        local_weight = []
        local_loss = {}
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())
        eg_momentum = 0.9
        selected_participants = None

        if len(blocked) > 0:
            parts_to_ublock = []
            for part, since in blocked.items():
                if since + int(total_participants / selected_participants_num) <= epoch:
                    parts_to_ublock.append(part)
                    participants_count[part] = 0

            [blocked.pop(part) for part in parts_to_ublock]

        # User selection
        if epoch == 0 or args.participation_rate < 1:
            #selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
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

        # AGM server model -> lookahead with global momentum
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            sending_model_dict[key] += -1 * args.lamb * global_delta[key]

        sending_model = copy.deepcopy(model)
        sending_model.load_state_dict(sending_model_dict)
        malicious_list = {}

        for participant in selected_participants:
            num_of_data_clients, idxs, current_trainset, malicious = get_participant(args, participant, dataset,
                                                                                     dataset_fake, num_of_data_clients,
                                                                                     trainset, trainset_fake, epoch)
            local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                         batch_size=args.batch_size, dataset=current_trainset, idxs=idxs,
                                         alpha=this_alpha)
            malicious_list[participant] = malicious

            weight, loss = local_setting.train(copy.deepcopy(sending_model).to(device), epoch)
            local_K.append(local_setting.K)
            local_weight.append(copy.deepcopy(weight))
            local_loss[participant] = copy.deepcopy(loss)

            # Store local delta
            delta = {}
            for key in weight.keys():
                delta[key] = this_tau*weight[key]+(1-this_tau)*sending_model_dict[key] - global_weight[key]
            local_delta.append(delta)

        total_num_of_data_clients=sum(num_of_data_clients)
        FedAvg_weight = copy.deepcopy(local_weight[0])
        for key in FedAvg_weight.keys():
            for i in range(len(local_weight)):
                if i==0:
                    FedAvg_weight[key]*=num_of_data_clients[i]
                else:                       
                    FedAvg_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients
            FedAvg_weight[key] = FedAvg_weight[key]*this_tau +(1-this_tau)*sending_model_dict[key]
        global_delta = copy.deepcopy(local_delta[0])

        for key in global_delta.keys():
            for i in range(len(local_delta)):
                if i==0:
                    global_delta[key] *= num_of_data_clients[i]
                else:
                    global_delta[key] += local_delta[i][key] * num_of_data_clients[i]
            global_delta[key] = global_delta[key] / (-1 * total_num_of_data_clients)

        model.load_state_dict(FedAvg_weight)
        loss_avg = sum(local_loss) / len(local_loss)
        print(' num_of_data_clients : ', num_of_data_clients)
        print(' Participants IDS: ', selected_participants)
        print(' Average loss {:.3f}'.format(loss_avg))
        print(f"Local losses: {local_loss}")
        loss_train.append(loss_avg)
        #loss_func = nn.NLLLoss()
        prev_model = copy.deepcopy(model)
        prev_model.load_state_dict(global_weight)
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
        cur_ig = calc_ig(global_loss, local_loss, entropy)

        participants_score, ig = update_participants_score(participants_score, cur_ig, ig, eg_momentum)

        print(participants_score)

        # accuracy = (accuracy / len(testloader)) * 100
        print('Loss of the network on the 10000 test images: %f %%' % metrics['loss'])
        print('Accuracy of the network on the 10000 test images: %f %%' % metrics['accuracy'])
        print('Precision of the network on the 10000 test images: %f %%' % metrics['precision'])
        print('Sensitivity of the network on the 10000 test images: %f %%' % metrics['sensitivity'])
        print('Specificity of the network on the 10000 test images: %f %%' % metrics['specificity'])
        print('F1-score of the network on the 10000 test images: %f %%' % metrics['f1score'])
        print(f'Information Gain: {ig}')


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
        save((args.eval_path, args.global_method + "_acc"), wandb_dict[args.mode + "_acc"])
        save((args.eval_path, args.global_method + "_prec"), wandb_dict[args.mode + "_prec"])
        save((args.eval_path, args.global_method + "_sens"), wandb_dict[args.mode + "_sens"])
        save((args.eval_path, args.global_method + "_spec"), wandb_dict[args.mode + "_spec"])
        save((args.eval_path, args.global_method + "_f1"), wandb_dict[args.mode + "_f1"])
        save((args.eval_path, args.global_method + "_loss"), wandb_dict[args.mode + "_loss"])

        this_lr *= args.learning_rate_decay
        this_tau *=args.server_learning_rate_decay
        if args.alpha_mul_epoch:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch:
            this_alpha = args.alpha / (epoch + 1)

    if valloader is not None:
        model.eval()
        #test_metric = do_evaluation(valloader, model=model, device=device, loss_func=loss_func,
        #                            prev_model=prev_model, alpha=args.alpha, mu=args.mu)
        test_metric = do_evaluation(testloader=valloader, model=model, device=device)
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
        save((args.eval_path, args.mode + "_ig"), ig)