# coding: utf-8

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import numpy as np
from utils import *
from libs.dataset.dataset_loader import NUM_CLASSES_LOOKUP_TABLE

#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def GlobalUpdate(args, device, trainset, testloader, local_update):
    model = get_model(arch=args.arch, num_classes=NUM_CLASSES_LOOKUP_TABLE[args.set],
                      l2_norm=args.l2_norm)
    model.to(device)
    wandb.watch(model)
    model.train()
    oneclass = True if NUM_CLASSES_LOOKUP_TABLE[args.set] <= 1 else False

    dataset = get_dataset(args, trainset, args.mode)
    loss_train = []
    acc_train = []
    this_lr = args.lr
    this_alpha = args.alpha
    selected_participants_num = max(int(args.participation_rate * args.num_of_clients), 1)
    selected_participants = None
    for epoch in range(args.global_epochs):
        wandb_dict = {}
        num_of_data_clients = []
        local_weight = []
        local_loss = []
        local_delta = []
        global_weight = copy.deepcopy(model.state_dict())

        # Sample participating agents for this global round

        if epoch == 0 or args.participation_rate < 1:
            selected_participants = np.random.choice(range(args.num_of_clients),
                                                     selected_participants_num,
                                                     replace=False)

        print(f"This is global {epoch} epoch")
        if selected_participants is None:
            return

        for participant in selected_participants:
            num_of_data_clients.append(len(dataset[participant]))
            local_setting = local_update(args=args, lr=this_lr, local_epoch=args.local_epochs, device=device,
                                         batch_size=args.batch_size, dataset=trainset, idxs=dataset[participant],
                                         alpha=this_alpha)
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
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    x, labels = data[0].to(device), data[1].to(device)
                    outputs = model(x)
                    if oneclass:
                        predicted = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in outputs]))
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %f %%' % (
                    100 * correct / float(total)))
            acc_train.append(100 * correct / float(total))

        model.train()
        wandb_dict[args.mode + "_acc"] = acc_train[-1]
        wandb_dict[args.mode + '_loss'] = loss_avg
        wandb_dict['lr'] = this_lr
        wandb.log(wandb_dict)

        this_lr *= args.learning_rate_decay
        if args.alpha_mul_epoch:
            this_alpha = args.alpha * (epoch + 1)
        elif args.alpha_divide_epoch:
            this_alpha = args.alpha / (epoch + 1)