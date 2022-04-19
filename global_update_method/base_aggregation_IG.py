# coding: utf-8
import copy
import random
from typing import Tuple

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import numpy as np
import os
from utils import *
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
from libs.evaluation.metrics import Evaluator
from torch.utils.data import DataLoader, TensorDataset

#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def save(path, metric):
    exists = False
    if os.path.exists(path):
        if os.stat(path).st_size > 0:
            exists = True
    file = open(path, 'a')
    if exists:
        file.write(',')
    file.write(str(metric))
    file.close()


def shuffle(arr: np.array) -> np.array:
    np.random.shuffle(arr)
    return arr


def do_evaluation(testloader, model, device):
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    # correct = 0
    # total = 0
    accuracy = 0
    batch_loss = []
    with torch.no_grad():
        preds = np.array([])
        full_lables = np.array([])
        first = True
        for x, labels in testloader:
            # print('loading data from testloader')
            x, labels = x.to(device), labels.to(device)
            # print('sending to the model..')
            outputs = model(x)
            val_loss = loss_func(outputs, labels)
            batch_loss.append(val_loss.item())
            # print('checking the classes')
            top_p, top_class = outputs.topk(1, dim=1)
            if first:
                preds = top_class.numpy()
                full_lables = copy.deepcopy(labels)
                first = False
            else:
                preds = np.concatenate((preds, top_class.numpy()))
                full_lables = np.concatenate((full_lables, labels))

            # print('evaluating the correctness')
            # equals = top_class == labels.view(*top_class.shape)
            # print('calculating accuracy')
            # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        loss_avg = (sum(batch_loss) / len(batch_loss))
    print('calculating avg accuracy')
    evaluator = Evaluator('accuracy', 'precision', 'sensitivity', 'specificity', 'f1score')
    metrics = evaluator.run_metrics(preds, full_lables)
    metrics['loss'] = loss_avg
    # acc_train.append(accuracy)

    model.train()
    return metrics


def gen_train_fake(samples: int = 10000, features: int = 77, interval: Tuple[int, int] = (0, 1),
                   classes: tuple = (0, 1)) -> TensorDataset:
    train_np_x = np.array(
        [[np.random.uniform(interval[0], interval[1]) for _ in range(features)] for _ in range(samples)])
    train_np_y = np.array([shuffle(np.array(classes)) for _ in range(samples)])

    train_tensor_x = torch.Tensor(train_np_x)
    train_tensor_y = torch.Tensor(train_np_y)

    trainset = TensorDataset(train_tensor_x, train_tensor_y)
    # dataloader = DataLoader(trainset, batch_size=batch, shuffle=False)
    return trainset


def calc_ig(parent_entropy: float, child_entropy: dict, parent_size: int, child_size: list) -> dict:
    ig = {}
    for idx, (client_id, child) in enumerate(child_entropy.items()):
        w = child_size[idx]/parent_size
        curr_ig = -np.log(parent_entropy) + (-np.log(w * child))
        ig[client_id] = curr_ig
    return ig


def GlobalUpdate(args, device, trainset, testloader, local_update, valloader=None):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    model.train()

    dataset = get_dataset(args, trainset, args.mode)
    directory = args.client_data + '/' + args.set + '/' + ('un' if args.data_unbalanced else '') + 'balanced_fake'
    filepath = directory + '/' + args.mode + (str(args.dirichlet_alpha) if args.mode == 'dirichlet' else '') + '_fake_clients' + str(
        args.num_of_clients) + '.txt'

    # Gen fake data
    selected_participants_fake_num = args.num_of_clients

    trainset_fake = gen_train_fake(samples=1500000) # 1590000
    dataset_fake = get_dataset(args, trainset_fake, args.mode, compatible=False,
                               directory=directory, filepath=filepath, participants=selected_participants_fake_num)

    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha

    total_participants = args.num_of_clients + selected_participants_fake_num
    selected_participants_num = max(int(args.participation_rate * total_participants), 1)
    #selected_participants = None
    # selected_participants_fake = np.random.choice(range(5),
                                                  #selected_participants_fake_num,
                                                  #replace=False)
    loss_func = nn.CrossEntropyLoss()
    ig = {}
    participants_score = {idx: selected_participants_num/total_participants for idx in range(total_participants)}
    ep_greedy = args.epsilon_greedy


    for epoch in range(args.global_epochs):
        print('starting a new epoch')
        wandb_dict = {}
        num_of_data_clients = []
        local_weight = []
        local_loss = []
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())

        # Sample participating agents for this global round
        selected_participants = []
        selection_helper = copy.deepcopy(participants_score)
        not_selected_participants = list(participants_score.keys())
        if epoch == 0:
            print('Selecting the participants')
            selected_participants = np.random.choice(range(args.num_of_clients + selected_participants_fake_num),
                                                     selected_participants_num,
                                                     replace=False)
            not_selected_participants = list(set(not_selected_participants) - set(selected_participants))

        elif args.participation_rate < 1:
            for _ in range(selected_participants_num):
                p = random.random()
                if p < ep_greedy:
                    print("Random selection")
                    if len(not_selected_participants) != 0:
                        selected = np.random.choice(not_selected_participants)
                    else:
                        selected = np.random.choice(list(selection_helper.keys()))
                    if selected in not_selected_participants:
                        not_selected_participants.remove(selected)
                    selection_helper.pop(selected)
                    selected_participants.append(selected)
                else:
                    # Select the best participant
                    print("Greedy selection")
                    selected = sorted(selection_helper, key=selection_helper.get, reverse=True)[0]
                    if selected in not_selected_participants:
                        not_selected_participants.remove(selected)
                    selection_helper.pop(selected)
                    selected_participants.append(selected)
            ## Selecting fake participants
            #selected_participants_fake = np.random.choice(range(args.num_of_clients),
                                                     #selected_participants_fake_num,
                                                     #replace=False)

        print(' Participants IDS: ', selected_participants)
        print(f"This is global {epoch} epoch")
        if selected_participants is None:
            return
        print('Training participants')
        models_val_loss = {}

        for participant in selected_participants:
            if participant < args.num_of_clients:
                num_of_data_clients.append(len(dataset[participant]))
                idxs = dataset[participant]
                local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                             batch_size=args.batch_size, dataset=trainset, idxs=idxs,
                                             alpha=this_alpha)
            else:
                num_of_data_clients.append(len(dataset_fake[participant - args.num_of_clients]))
                idxs = dataset_fake[participant - args.num_of_clients]
                local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                             batch_size=args.batch_size, dataset=trainset_fake,
                                             idxs=idxs,
                                             alpha=this_alpha)

            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))

            local_model = copy.deepcopy(model).to(device)
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

        '''print('Training participants fake participants')
        for participant in selected_participants_fake:
            num_of_data_clients.append(len(dataset_fake[participant]))
            local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                         batch_size=args.batch_size, dataset=trainset_fake, idxs=dataset_fake[participant],
                                         alpha=this_alpha)
            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))

            local_model = copy.deepcopy(model).to(device)
            local_model.load_state_dict(weight)
            local_model.eval()
            batch_loss = []
            with torch.no_grad():
                for x, labels in testloader:
                    x, labels = x.to(device), labels.to(device)
                    outputs = local_model(x)
                    local_val_loss = loss_func(outputs, labels)
                    batch_loss.append(local_val_loss.item())
                    models_val_loss[participant + args.num_of_clients] = sum(batch_loss) / len(batch_loss)
            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - global_weight[key]
            local_delta.append(delta)'''


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
        loss_train.append(loss_avg)
        # print(models_val_loss)


        print('performing the evaluation')
        model.eval()
        metrics = do_evaluation(testloader=testloader, model=model, device=device)
        model.train()
        cur_ig = calc_ig(metrics['loss'], models_val_loss, total_num_of_data_clients, num_of_data_clients)

        for client_id, client_ig in cur_ig.items():
            if client_id not in ig.keys():
                ig[client_id] = []
                ig[client_id].append(client_ig)
            else:
                ig[client_id].append(client_ig)
            participants_score[client_id] = sum(ig[client_id]) / len(ig[client_id])

        print(participants_score)


        if epoch % args.print_freq == 0:
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
        save(args.global_method + "_acc", wandb_dict[args.mode + "_acc"] )
        save(args.global_method + "_prec", wandb_dict[args.mode + "_prec"])
        save(args.global_method + "_sens", wandb_dict[args.mode + "_sens"])
        save(args.global_method + "_spec", wandb_dict[args.mode + "_spec"])
        save(args.global_method + "_f1", wandb_dict[args.mode + "_f1"])
        save(args.global_method + "_loss", wandb_dict[args.mode + "_loss"])
        print('Decay LR...')
        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch:
            this_alpha = args.alpha / (epoch + 1)

    if valloader is not None:
        model.eval()
        test_metric = do_evaluation(valloader, model, device)
        model.train()

        print('Final Accuracy of the network on the 10000 test images: %f %%' % test_metric['accuracy'])
        print('Final Precision of the network on the 10000 test images: %f %%' % test_metric['precision'])
        print('Final Sensitivity of the network on the 10000 test images: %f %%' % test_metric['sensitivity'])
        print('Final Specificity of the network on the 10000 test images: %f %%' % test_metric['specificity'])
        print('Final F1-score of the network on the 10000 test images: %f %%' % test_metric['f1score'])

        save(args.mode + "_test_acc", test_metric['accuracy'])
        save(args.mode + "_test_prec", test_metric['precision'])
        save(args.mode + "_test_sens", test_metric['sensitivity'])
        save(args.mode + "_test_spec", test_metric['specificity'])
        save(args.mode + "_test_f1", test_metric['f1score'])
        save(args.mode + "_ig", ig)
