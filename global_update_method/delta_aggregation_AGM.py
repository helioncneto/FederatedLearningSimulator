# coding: utf-8

from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
import numpy as np
from utils import *


def GlobalUpdate(args, device, trainset, testloader, local_update):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
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
        wandb_dict={}
        num_of_data_clients=[]
        local_K=[]

        local_weight = []
        local_loss = []
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())
        # User selection
        if epoch == 0 or args.participation_rate < 1:
            selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        else:
            pass 
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
        if isCICIDS2017:
            loss_func = nn.NLLLoss()
        else:
            loss_func = nn.CrossEntropyLoss()
        prev_model = copy.deepcopy(model)
        prev_model.load_state_dict(global_weight)
        if epoch % args.print_freq == 0:
            model.eval()
            correct = 0
            total = 0
            acc_test = []
            ce_loss_test = []
            reg_loss_test = []
            total_loss_test = []
            with torch.no_grad():
                for data in testloader:
                    x, labels = data[0].to(device), data[1].to(device)
                    outputs = model(x)
                    if isCICIDS2017:
                        ce_loss = loss_func(outputs, labels.float())
                    else:
                        ce_loss = loss_func(outputs, labels)

                    ## Weight L2 loss
                    reg_loss = 0
                    fixed_params = {n: p for n, p in prev_model.named_parameters()}
                    for n, p in model.named_parameters():
                        reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()


                    loss = args.alpha * ce_loss + 0.5 * args.mu * reg_loss
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    #print(f'Pred: {predicted} \n l=Label:{labels}')
                    correct += (predicted == labels).sum().item()

                    ce_loss_test.append(ce_loss.item())
                    reg_loss_test.append(reg_loss.item())
                    total_loss_test.append(loss.item())

            print('Accuracy of the network on the 10000 test images: %f %%' % (
                    100 * correct / float(total)))
            acc_train.append(100 * correct / float(total))

            model.train()
            wandb_dict[args.mode + "_acc"]=acc_train[-1]
            wandb_dict[args.mode + "_total_loss"] = sum(total_loss_test) / len(total_loss_test)
            wandb_dict[args.mode + "_ce_loss"] = sum(ce_loss_test) / len(ce_loss_test)
            wandb_dict[args.mode + "_reg_loss"] = sum(reg_loss_test) / len(reg_loss_test)
        
        wandb_dict[args.mode + '_loss']= loss_avg
        wandb_dict['lr']=this_lr
        wandb.log(wandb_dict)

        this_lr *= args.learning_rate_decay
        this_tau *=args.server_learning_rate_decay
        if args.alpha_mul_epoch:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch:
            this_alpha = args.alpha / (epoch + 1)