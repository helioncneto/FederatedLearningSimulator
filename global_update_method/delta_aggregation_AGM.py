# coding: utf-8
import os

from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
import numpy as np

from libs.evaluation.metrics import Evaluator
from utils import *
from utils.helper import save, do_evaluation


def GlobalUpdate(args, device, trainset, testloader, local_update, valloader=None):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    model.train()
    isCICIDS2017 = True if args.mode == "CICIDS2017" else False

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha
    this_tau = args.tau
    global_delta = copy.deepcopy(model.state_dict())
    m = max(int(args.participation_rate * args.num_of_clients), 1)
    for key in global_delta.keys():
        global_delta[key] = torch.zeros_like(global_delta[key])
    for epoch in range(args.global_epochs):
        wandb_dict = {}
        num_of_data_clients = []
        local_K = []

        local_weight = []
        local_loss = []
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())
        # User selection
        if epoch == 0 or args.participation_rate < 1:
            selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)

        print(f"This is global {epoch} epoch")

        # AGM server model -> lookahead with global momentum
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            sending_model_dict[key] += -1 * args.lamb * global_delta[key]

        sending_model = copy.deepcopy(model)
        sending_model.load_state_dict(sending_model_dict)

        for user in selected_user:
            num_of_data_clients.append(len(dataset[user]))
            local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                         batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha)
            weight, loss = local_setting.train(copy.deepcopy(sending_model).to(device), epoch)
            local_K.append(local_setting.K)
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))

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
        print(' Participants IDS: ', selected_user)
        print(' Average loss {:.3f}'.format(loss_avg))
        loss_train.append(loss_avg)
        #loss_func = nn.NLLLoss()
        prev_model = copy.deepcopy(model)
        prev_model.load_state_dict(global_weight)
        #if epoch % args.print_freq == 0:

        #model.eval()
        #metrics = do_evaluation(testloader=testloader, model=model, device=device,
        #                        prev_model=prev_model, alpha=args.alpha, mu=args.mu)
        metrics = do_evaluation(testloader=testloader, model=model, device=device)

        model.train()
        # accuracy = (accuracy / len(testloader)) * 100
        print('Accuracy of the network on the 10000 test images: %f %%' % metrics['accuracy'])
        print('Precision of the network on the 10000 test images: %f %%' % metrics['precision'])
        print('Sensitivity of the network on the 10000 test images: %f %%' % metrics['sensitivity'])
        print('Specificity of the network on the 10000 test images: %f %%' % metrics['specificity'])
        print('F1-score of the network on the 10000 test images: %f %%' % metrics['f1score'])


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