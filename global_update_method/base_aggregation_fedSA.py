#!/usr/bin/env python
from libs.methods.FedSA import SimulatedAnnealing
from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import copy
import torch
#from utils import *
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE

#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def GlobalUpdate(args, device, trainset, testloader, local_update):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    wandb.watch(model)
    model.train()

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []

    selected_participants_num = max(int(args.participation_rate * args.num_of_clients), 1)

    num_of_data_clients = []

    # Sample participating agents for this global round

    sa = SimulatedAnnealing(initial_temperature=0.8, cooling=0.05, lr=(0.001, 0.1), local_update=(0, 15),
                            participants=(0, args.num_of_clients - 1, selected_participants_num), computing_time=1,
                            threshold=0.01)
    sa.run(epoch=args.global_epochs, obj=participants_train, model=model, data=dataset, trainset=trainset,
           testloader=testloader, local_update=local_update, device=device, loss_train=loss_train,
           acc_train=acc_train, args=args)


def participants_train(X, global_model, dataset, epoch, kwargs):
    lr = X[0]
    # print('Learning rate: {}'.format(lr))
    local_epochs = int(X[1])
    selected_participants = [int(X[i]) for i in range(2, len(X) - 1)]
    trainset = kwargs['trainset']
    testloader = kwargs['testloader']
    local_update = kwargs['local_update']
    device = kwargs['device']
    loss_train = kwargs['loss_train']
    acc_train = kwargs['acc_train']
    args = kwargs['args']

    num_of_data_clients = []
    this_alpha = args.alpha
    local_weight = []
    local_loss = []
    local_delta = []
    global_weight = copy.deepcopy(global_model.state_dict())
    wandb_dict = {}

    print(f'Aggregation Round: {epoch}')

    for participant in selected_participants:
        num_of_data_clients.append(len(dataset[participant]))
        local_setting = local_update(args=args, lr=lr, local_epoch=local_epochs, device=device,
                                     batch_size=args.batch_size, dataset=trainset, idxs=dataset[participant],
                                     alpha=this_alpha)
        weight, loss = local_setting.train(net=copy.deepcopy(global_model).to(device))
        local_weight.append(copy.deepcopy(weight))
        local_loss.append(copy.deepcopy(loss))
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
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            x, labels = data[0].to(device), data[1].to(device)
            outputs = global_model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network: %f %%' % (
            100 * correct / float(total)))
    acc_train.append(100 * correct / float(total))

    global_model.train()
    wandb_dict[args.mode + "_acc"] = acc_train[-1]
    wandb_dict[args.mode + '_loss'] = loss_avg
    wandb_dict['lr'] = lr
    wandb.log(wandb_dict)
    return global_model, loss_avg, acc_train[-1]
