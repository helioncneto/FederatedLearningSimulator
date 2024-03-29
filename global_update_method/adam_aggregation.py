#!/usr/bin/env python
# coding: utf-8

# In[2]:

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import wandb
import numpy as np
from torch import nn
import copy
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import DatasetSplit
from global_update_method.distcheck import check_data_distribution
import umap.umap_ as umap
from mpl_toolkits import mplot3d
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from torch.utils.data import DataLoader
# from utils import log_ConfusionMatrix_Umap, log_acc
from utils import *
from utils import CenterUpdate

def GlobalUpdate(args, device, trainset, testloader, local_update):
    model = get_model(args)
    model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    model.train()
    dataset = get_dataset(args, trainset, args.num_of_clients, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha

    
    
    delta_t=copy.deepcopy(model.state_dict())
    v_t=copy.deepcopy(delta_t)
    for key in delta_t.keys():
        delta_t[key]*=0
        v_t[key]=delta_t[key]+(args.tau**2)
    this_server_lr=args.g_lr

    for epoch in range(args.global_epochs):
        global_weight = copy.deepcopy(model.state_dict())
        wandb_dict={}
        num_of_data_clients=[]
        local_weight = []
        local_loss = []
        local_delta = []
        m = max(int(args.participation_rate * args.num_of_clients), 1)

        # Sample participating agents for this global round
        selected_user = np.random.choice(range(args.num_of_clients), m, replace=False)
        print(f"This is global {epoch} epoch")
        for user in selected_user:
            num_of_data_clients.append(len(dataset[user]))
            local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                         batch_size=args.batch_size, dataset=trainset, idxs=dataset[user], alpha=this_alpha)
            weight, loss = local_setting.train(net=copy.deepcopy(model).to(device))
            local_weight.append(copy.deepcopy(weight))
            local_loss.append(copy.deepcopy(loss))
            delta = {}
            for key in weight.keys():
                delta[key] = weight[key] - global_weight[key]
            local_delta.append(delta)
        total_num_of_data_clients=sum(num_of_data_clients)            
            
        client_weight = copy.deepcopy(local_weight[0])
        delta_client_mean = copy.deepcopy(model.state_dict())
        x_t = copy.deepcopy(model.state_dict())
        for key in client_weight.keys():
            for i in range(len(local_weight)):
                if i==0:
                    client_weight[key]*=num_of_data_clients[i]
                else:
                    client_weight[key] += local_weight[i][key]*num_of_data_clients[i]
            client_weight[key] /= total_num_of_data_clients
            delta_client_mean[key]=client_weight[key]-x_t[key]
            delta_t[key]=delta_t[key]*args.beta_1 + delta_client_mean[key]*(1-args.beta_1)
            v_t[key]=v_t[key]*args.beta_2+(delta_t[key]*delta_t[key])*(1-args.beta_2)
            x_t[key]+=this_server_lr*delta_t[key]/((v_t[key]**0.5)+args.tau)

        model.load_state_dict(x_t)
        loss_avg = sum(local_loss) / len(local_loss)
        print(' num_of_data_clients : ',num_of_data_clients)
        print(' Average loss {:.3f}'.format( loss_avg))
        loss_train.append(loss_avg)
        if epoch % args.print_freq == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %f %%' % (
                    100 * correct / float(total)))
            acc_train.append(100 * correct / float(total))

        model.train()
        wandb_dict[args.mode + "_acc"]=acc_train[-1]
        wandb_dict[args.mode + '_loss']= loss_avg
        wandb_dict['lr']=this_lr
        if args.use_wandb:
            wandb.log(wandb_dict)

        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch == True:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch == True:
            this_alpha = args.alpha / (epoch + 1)

    print('loss_train')
    print(loss_train)

    print('acc_train')
    print(acc_train)