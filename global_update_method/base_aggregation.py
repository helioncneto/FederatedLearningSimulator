# coding: utf-8

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import numpy as np
import os
from utils import *
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
from libs.evaluation.metrics import Evaluator
from utils.helper import save, do_evaluation, add_malicious_participants, get_participant


#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def GlobalUpdate(args, device, trainset, testloader, local_update, valloader=None):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    model.train()

    directory = args.client_data + '/' + args.set + '/' + ('un' if args.data_unbalanced else '') + 'balanced_fake'
    filepath = directory + '/' + args.mode + (
        str(args.dirichlet_alpha) if args.mode == 'dirichlet' else '') + '_fake_clients' + str(
        args.num_of_clients) + '.txt'

    dataset = get_dataset(args, trainset, args.num_of_clients, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha
    selected_participants_num = max(int(args.participation_rate * args.num_of_clients), 1)
    selected_participants = None

    # Gen fake data
    malicious_participant_dataloader_table = {}
    if args.malicious_rate > 0:
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
        local_loss = []
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())

        # Sample participating agents for this global round

        if epoch == 0 or args.participation_rate < 1:
            print('Selecting the participants')
            selected_participants = np.random.choice(range(args.num_of_clients),
                                                     selected_participants_num,
                                                     replace=False)

        print(f"This is global {epoch} epoch")
        if selected_participants is None:
            return

        print('Training participants')
        malicious_list = {}
        for participant in selected_participants:
            num_of_data_clients, idxs, current_trainset, malicious = get_participant(args, participant, dataset,
                                                                                     dataset_fake, num_of_data_clients,
                                                                                     trainset, trainset_fake, epoch)
            local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                         batch_size=args.batch_size, dataset=current_trainset, idxs=idxs,
                                         alpha=this_alpha)
            malicious_list[participant] = malicious

            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
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
                    FedAvg_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            FedAvg_weight[key] /= total_num_of_data_clients
        model.load_state_dict(FedAvg_weight)

        loss_avg = sum(local_loss) / len(local_loss)
        print(' num_of_data_clients : ', num_of_data_clients)
        print(' Participants IDS: ', selected_participants)
        print(' Average loss {:.3f}'.format(loss_avg))
        loss_train.append(loss_avg)

        if epoch % args.print_freq == 0:
            print('performing the evaluation')
            model.eval()
            metrics = do_evaluation(testloader=testloader, model=model, device=device)
            # accuracy = (accuracy / len(testloader)) * 100
            print('Accuracy of the network on the 10000 test images: %f %%' % metrics['accuracy'])
            print('Precision of the network on the 10000 test images: %f %%' % metrics['precision'])
            print('Sensitivity of the network on the 10000 test images: %f %%' % metrics['sensitivity'])
            print('Specificity of the network on the 10000 test images: %f %%' % metrics['specificity'])
            print('F1-score of the network on the 10000 test images: %f %%' % metrics['f1score'])
            model.train()

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

    if valloader is not None:
        model.eval()
        test_metric = do_evaluation(valloader, model, device)
        model.train()

        print('Final Accuracy of the network on the 10000 test images: %f %%' % test_metric['accuracy'])
        print('Final Precision of the network on the 10000 test images: %f %%' % test_metric['precision'])
        print('Final Sensitivity of the network on the 10000 test images: %f %%' % test_metric['sensitivity'])
        print('Final Specificity of the network on the 10000 test images: %f %%' % test_metric['specificity'])
        print('Final F1-score of the network on the 10000 test images: %f %%' % test_metric['f1score'])

        save((args.eval_path, args.mode + "_test_acc"), test_metric['accuracy'])
        save((args.eval_path, args.mode + "_test_prec"), test_metric['precision'])
        save((args.eval_path, args.mode + "_test_sens"), test_metric['sensitivity'])
        save((args.eval_path, args.mode + "_test_spec"), test_metric['specificity'])
        save((args.eval_path, args.mode + "_test_f1"), test_metric['f1score'])
