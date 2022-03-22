# coding: utf-8

from libs.dataset.dataset_loader import NUM_CLASSES_LOOKUP_TABLE

from libs.methods.FedSA import SimulatedAnnealing
from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import copy
import torch


def GlobalUpdate(args, device, trainset, testloader, local_update):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    wandb.watch(model)
    model.train()

    dataset = get_dataset(args, trainset, args.mode)
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
    print(f"This is global {epoch} epoch")

    # AGM server model -> lookahead with global momentum
    sending_model_dict = copy.deepcopy(global_model.state_dict())
    for key in global_delta.keys():
        sending_model_dict[key] += -1 * args.lamb * global_delta[key]

    sending_model = copy.deepcopy(global_model)
    sending_model.load_state_dict(sending_model_dict)

    for user in selected_participants:
        num_of_data_clients.append(len(dataset[user]))
        local_setting = local_update(args=args, lr=lr, local_epoch=args.local_epochs, device=device,
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
        correct = 0
        total = 0
        acc_test = []
        ce_loss_test = []
        reg_loss_test = []
        total_loss_test = []
        with torch.no_grad():
            for data in testloader:
                x, labels = data[0].to(device), data[1].to(device)
                outputs = global_model(x)
                ce_loss = loss_func(outputs, labels)

                ## Weight L2 loss
                reg_loss = 0
                fixed_params = {n: p for n, p in prev_model.named_parameters()}
                for n, p in global_model.named_parameters():
                    reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()

                loss = args.alpha * ce_loss + 0.5 * args.mu * reg_loss
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # print(f'Pred: {predicted} \n l=Label:{labels}')
                correct += (predicted == labels).sum().item()

                ce_loss_test.append(ce_loss.item())
                reg_loss_test.append(reg_loss.item())
                total_loss_test.append(loss.item())

        print('Accuracy of the network on the 10000 test images: %f %%' % (
                100 * correct / float(total)))
        acc_train.append(100 * correct / float(total))

        global_model.train()
        wandb_dict[args.mode + "_acc"] = acc_train[-1]
        wandb_dict[args.mode + "_total_loss"] = sum(total_loss_test) / len(total_loss_test)
        wandb_dict[args.mode + "_ce_loss"] = sum(ce_loss_test) / len(ce_loss_test)
        wandb_dict[args.mode + "_reg_loss"] = sum(reg_loss_test) / len(reg_loss_test)

    wandb_dict[args.mode + '_loss'] = loss_avg
    wandb_dict['lr'] = lr
    wandb.log(wandb_dict)

    #this_tau *= args.server_learning_rate_decay

    return global_model, loss_avg, acc_train[-1]
