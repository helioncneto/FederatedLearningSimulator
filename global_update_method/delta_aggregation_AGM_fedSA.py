# coding: utf-8

from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE

from libs.methods.FedSA import SimulatedAnnealing
from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import copy
import torch
import os
import numpy as np
from libs.evaluation.metrics import Evaluator
from utils.helper import save, do_evaluation


def GlobalUpdate(args, device, trainset, testloader, local_update, valloader=None):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    model.train()

    dataset = get_dataset(args, trainset, args.num_of_clients, args.mode)
    loss_train = []
    acc_train = []

    this_tau = args.tau
    global_delta = copy.deepcopy(model.state_dict())
    selected_participants_num = max(int(args.participation_rate * args.num_of_clients), 1)
    for key in global_delta.keys():
        global_delta[key] = torch.zeros_like(global_delta[key])

    sa = SimulatedAnnealing(initial_temperature=0.8, cooling=0.05, lr=(0.001, 0.1), local_update=(0, 15),
                            participants=(0, args.num_of_clients - 1, selected_participants_num), computing_time=1,
                            threshold=0.01)
    sa.run(epoch=args.global_epochs, obj=participants_train, model=model, data=dataset, trainset=trainset,
           testloader=testloader, local_update=local_update, device=device, loss_train=loss_train,
           acc_train=acc_train, args=args, global_delta=global_delta, this_tau=this_tau)

    if valloader is not None:
        sa.model.eval()
        test_metric = do_evaluation(testloader=valloader, model=sa.model, device=device)
        sa.model.train()

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



def participants_train(X, global_model, dataset, epoch, kwargs):
    wandb_dict = {}
    num_of_data_clients = []
    local_K = []

    lr = X[0]
    # print('Learning rate: {}'.format(lr))
    local_epochs = int(X[1])
    selected_participants = [int(X[i]) for i in range(2, len(X) - 1)]

    local_weight = []
    local_loss = []
    local_delta = []
    global_weight = copy.deepcopy(global_model.state_dict())

    trainset = kwargs['trainset']
    testloader = kwargs['testloader']
    local_update = kwargs['local_update']
    device = kwargs['device']
    loss_train = kwargs['loss_train']
    acc_train = kwargs['acc_train']
    args = kwargs['args']
    global_delta = kwargs['global_delta']
    this_tau = kwargs['this_tau']
    this_alpha = args.alpha

    # User selection
    '''if epoch == 0 or args.participation_rate < 1:
        selected_user = np.random.choice(range(args.num_of_clients), selected_participants_num, replace=False)
    else:
        pass'''
    print(f'Aggregation Round: {epoch}')

    # AGM server model -> lookahead with global momentum
    sending_model_dict = copy.deepcopy(global_model.state_dict())
    for key in global_delta.keys():
        sending_model_dict[key] += -1 * args.lamb * global_delta[key]

    sending_model = copy.deepcopy(global_model)
    sending_model.load_state_dict(sending_model_dict)

    for user in selected_participants:
        num_of_data_clients.append(len(dataset[user]))
        local_setting = local_update(args=args, lr=lr, local_epoch=local_epochs, device=device,
                                     batch_size=args.batch_size, dataset=trainset, idxs=dataset[user],
                                     alpha=this_alpha)
        weight, loss = local_setting.train(copy.deepcopy(sending_model).to(device), epoch)
        local_K.append(local_setting.K)
        local_weight.append(copy.deepcopy(weight))
        local_loss.append(copy.deepcopy(loss))

        # Store local delta
        delta = {}
        for key in weight.keys():
            delta[key] = this_tau * weight[key] + (1 - this_tau) * sending_model_dict[key] - global_weight[key]
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
        FedAvg_weight[key] = FedAvg_weight[key] * this_tau + (1 - this_tau) * sending_model_dict[key]
    global_delta = copy.deepcopy(local_delta[0])

    for key in global_delta.keys():
        for i in range(len(local_delta)):
            if i == 0:
                global_delta[key] *= num_of_data_clients[i]
            else:
                global_delta[key] += local_delta[i][key] * num_of_data_clients[i]
        global_delta[key] = global_delta[key] / (-1 * total_num_of_data_clients)

    global_model.load_state_dict(FedAvg_weight)
    loss_avg = sum(local_loss) / len(local_loss)
    print(' num_of_data_clients : ', num_of_data_clients)
    print(' Participants IDS: ', selected_participants)
    print(' Average loss {:.3f}'.format(loss_avg))
    loss_train.append(loss_avg)
    loss_func = torch.nn.NLLLoss()

    prev_model = copy.deepcopy(global_model)
    prev_model.load_state_dict(global_weight)
    if epoch % args.print_freq == 0:
        global_model.eval()
        metrics = do_evaluation(testloader=testloader, model=global_model, device=device, loss_func=loss_func,
                                    prev_model=prev_model, alpha=args.alpha, mu=args.mu)
        global_model.train()
        # accuracy = (accuracy / len(testloader)) * 100
        print('Accuracy of the network on the 10000 test images: %f %%' % metrics['accuracy'])
        print('Precision of the network on the 10000 test images: %f %%' % metrics['precision'])
        print('Sensitivity of the network on the 10000 test images: %f %%' % metrics['sensitivity'])
        print('Specificity of the network on the 10000 test images: %f %%' % metrics['specificity'])
        print('F1-score of the network on the 10000 test images: %f %%' % metrics['f1score'])
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
    save((args.eval_path, args.global_method + "_acc"), wandb_dict[args.mode + "_acc"])
    save((args.eval_path, args.global_method + "_prec"), wandb_dict[args.mode + "_prec"])
    save((args.eval_path, args.global_method + "_sens"), wandb_dict[args.mode + "_sens"])
    save((args.eval_path, args.global_method + "_spec"), wandb_dict[args.mode + "_spec"])
    save((args.eval_path, args.global_method + "_f1"), wandb_dict[args.mode + "_f1"])
    save((args.eval_path, args.global_method + "_loss"), wandb_dict[args.mode + "_loss"])

    #this_tau *= args.server_learning_rate_decay

    return global_model, loss_avg, metrics['accuracy']
