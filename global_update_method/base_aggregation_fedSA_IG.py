#!/usr/bin/env python
import numpy as np
import torch
from torch import nn

from libs.methods.FedSA import SimulatedAnnealing
from libs.methods.ig import selection_ig, update_participants_score, calc_ig, load_from_file, save_dict_file
from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import copy
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
from utils.helper import save, do_evaluation




#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def GlobalUpdate(args, device, trainset, testloader, local_update, valloader=None):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    model.train()

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []

    selected_participants_num = max(int(args.participation_rate * args.num_of_clients), 1)
    total_participants = args.num_of_clients
    participants_score = {idx: selected_participants_num / total_participants for idx in range(total_participants)}
    not_selected_participants_list = list(participants_score.keys())
    not_selected_participants = {"not_selected": not_selected_participants_list}
    ig = {}

    save_dict_file('.tmp/ig.pkl', ig)
    save_dict_file('.tmp/participants_score.pkl', participants_score)
    save_dict_file('.tmp/not_selected_participants.pkl', not_selected_participants)

    num_of_data_clients = []

    # Sample participating agents for this global round

    sa = SimulatedAnnealing(initial_temperature=0.8, cooling=0.05, lr=(0.001, 0.1), local_update=(0, 15),
                            participants=(0, args.num_of_clients - 1, selected_participants_num), computing_time=1,
                            threshold=0.01)
    sa.run(epoch=args.global_epochs, obj=participants_train, model=model, data=dataset, trainset=trainset,
           testloader=testloader, local_update=local_update, device=device, loss_train=loss_train,
           acc_train=acc_train, args=args, selected_participants_num=selected_participants_num, ep_greedy=0.9,
           not_selected_participants=not_selected_participants, participants_score=participants_score,
           ig=ig)

    if valloader is not None:
        sa.model.eval()
        test_metric = do_evaluation(valloader, sa.model, device)
        sa.model.train()

        print('Final Accuracy of the network on the 10000 test images: %f %%' % test_metric['accuracy'])
        print('Final Precision of the network on the 10000 test images: %f %%' % test_metric['precision'])
        print('Final Sensitivity of the network on the 10000 test images: %f %%' % test_metric['sensitivity'])
        print('Final Specificity of the network on the 10000 test images: %f %%' % test_metric['specificity'])
        print('Final F1-score of the network on the 10000 test images: %f %%' % test_metric['f1score'])

        save(args.global_method + "_test_acc", test_metric['accuracy'])
        save(args.global_method + "_test_prec", test_metric['precision'])
        save(args.global_method + "_test_sens", test_metric['sensitivity'])
        save(args.global_method + "_test_spec", test_metric['specificity'])
        save(args.global_method + "_test_f1", test_metric['f1score'])


def participants_train(X, global_model, dataset, epoch, kwargs):
    lr = X[0]
    loss_func = nn.CrossEntropyLoss()
    eg_momentum = 0.9
    # print('Learning rate: {}'.format(lr))
    local_epochs = int(X[1])
    #selected_participants = [int(X[i]) for i in range(2, len(X) - 1)]

    trainset = kwargs['trainset']
    testloader = kwargs['testloader']
    local_update = kwargs['local_update']
    device = kwargs['device']
    loss_train = kwargs['loss_train']
    acc_train = kwargs['acc_train']
    args = kwargs['args']

    selected_participants_num = kwargs['selected_participants_num']
    ep_greedy = kwargs['ep_greedy']
    ep_greedy_decay = pow(0.01, 1 / args.global_epochs)
    if epoch != 1:
        ep_greedy = (ep_greedy * ep_greedy_decay) ** epoch
    not_selected_participants = load_from_file('.tmp/not_selected_participants.pkl')
    participants_score = load_from_file('.tmp/participants_score.pkl')
    ig = load_from_file('.tmp/ig.pkl')

    #not_selected_participants = kwargs['not_selected_participants']
    #participants_score = kwargs['participants_score']
    #ig = kwargs['ig']

    if epoch == 1 or args.participation_rate >= 1:
        print('Selecting the participants')
        # selected_participants = np.random.choice(range(args.num_of_clients + selected_participants_fake_num),
        # selected_participants_num,
        # replace=False)
        selected_participants = np.random.choice(range(args.num_of_clients),
                                                 selected_participants_num,
                                                 replace=False)
        not_selected_participants["not_selected"] = list(set(not_selected_participants["not_selected"]) - set(selected_participants))

    elif args.participation_rate < 1:
        selected_participants, not_selected_participants["not_selected"] = selection_ig(selected_participants_num, ep_greedy,
                                                                        not_selected_participants["not_selected"],
                                                                        participants_score)


    num_of_data_clients = []
    this_alpha = args.alpha
    local_weight = []
    local_loss = []
    local_delta = []
    global_weight = copy.deepcopy(global_model.state_dict())
    wandb_dict = {}

    print(f'Aggregation Round: {epoch}')
    models_val_loss = {}

    for participant in selected_participants:
        num_of_data_clients.append(len(dataset[participant]))
        local_setting = local_update(args=args, lr=lr, local_epoch=local_epochs, device=device,
                                     batch_size=args.batch_size, dataset=trainset, idxs=dataset[participant],
                                     alpha=this_alpha)
        weight, loss = local_setting.train(net=copy.deepcopy(global_model).to(device))
        local_weight.append(copy.deepcopy(weight))
        local_loss.append(copy.deepcopy(loss))
        local_model = copy.deepcopy(global_model).to(device)
        local_model.load_state_dict(weight)
        local_model.eval()

        batch_loss = []
        with torch.no_grad():
            for x, labels in testloader:
                x, labels = x.to(device), labels.to(device)
                outputs = local_model(x)
                local_val_loss = loss_func(outputs, labels)
                batch_loss.append(local_val_loss.item())
            models_val_loss[participant] = (sum(batch_loss) / len(batch_loss))

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
                FedAvg_weight[key] += local_weight[i][key] * num_of_data_clients[i]
        FedAvg_weight[key] /= total_num_of_data_clients
    global_model.load_state_dict(FedAvg_weight)

    loss_avg = sum(local_loss) / len(local_loss)
    print(' num_of_data_clients : ', num_of_data_clients)
    print(' Participants IDS: ', selected_participants)
    print(' Average loss {:.3f}'.format(loss_avg))
    loss_train.append(loss_avg)
    global_model.eval()
    metrics = do_evaluation(testloader, global_model, device)
    global_model.train()

    cur_ig = calc_ig(metrics['loss'], models_val_loss, total_num_of_data_clients, num_of_data_clients)

    participants_score, ig = update_participants_score(participants_score, cur_ig, ig, eg_momentum)

    save_dict_file('.tmp/ig.pkl', ig)
    save_dict_file('.tmp/participants_score.pkl', participants_score)
    save_dict_file('.tmp/not_selected_participants.pkl', not_selected_participants)

    print(participants_score)

    # accuracy = (accuracy / len(testloader)) * 100
    print('Accuracy of the network on the 10000 test images: %f %%' % metrics['accuracy'])
    print('Precision of the network on the 10000 test images: %f %%' % metrics['precision'])
    print('Sensitivity of the network on the 10000 test images: %f %%' % metrics['sensitivity'])
    print('Specificity of the network on the 10000 test images: %f %%' % metrics['specificity'])
    print('F1-score of the network on the 10000 test images: %f %%' % metrics['f1score'])
    print(f'Information Gain: {ig}')
    print(f'Participants loss: {models_val_loss}')
    # acc_train.append(accuracy)

    wandb_dict[args.mode + "_acc"] = metrics['accuracy']
    wandb_dict[args.mode + "_prec"] = metrics['precision']
    wandb_dict[args.mode + "_sens"] = metrics['sensitivity']
    wandb_dict[args.mode + "_spec"] = metrics['specificity']
    wandb_dict[args.mode + "_f1"] = metrics['f1score']
    wandb_dict[args.mode + '_loss'] = loss_avg
    wandb_dict['lr'] = lr
    if args.use_wandb:
        print('logging to wandb...')
        wandb.log(wandb_dict)
    save(args.global_method + "_acc", wandb_dict[args.mode + "_acc"])
    save(args.global_method + "_prec", wandb_dict[args.mode + "_prec"])
    save(args.global_method + "_sens", wandb_dict[args.mode + "_sens"])
    save(args.global_method + "_spec", wandb_dict[args.mode + "_spec"])
    save(args.global_method + "_f1", wandb_dict[args.mode + "_f1"])
    save(args.global_method + "_loss", wandb_dict[args.mode + "_loss"])

    if args.alpha_mul_epoch:
        this_alpha = args.alpha * (epoch + 1)
    elif args.alpha_divide_epoch:
        this_alpha = args.alpha / (epoch + 1)

    return global_model, loss_avg, metrics['accuracy']