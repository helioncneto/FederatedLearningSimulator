# coding: utf-8

from utils import get_scheduler, get_optimizer, get_model, get_dataset
import numpy as np
import os
from utils import *
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE
from libs.evaluation.metrics import Evaluator


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

def do_evaluation(testloader, model, device):
    model.eval()
    # correct = 0
    # total = 0
    accuracy = 0
    with torch.no_grad():
        preds = np.array([])
        full_lables = np.array([])
        first = True
        for x, labels in testloader:
            # print('loading data from testloader')
            x, labels = x.to(device), labels.to(device)
            # print('sending to the model..')
            outputs = model(x)
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
    print('calculating avg accuracy')
    evaluator = Evaluator('accuracy', 'precision', 'sensitivity', 'specificity', 'f1score')
    metrics = evaluator.run_metrics(preds, full_lables)
    # acc_train.append(accuracy)

    model.train()
    return metrics



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
    this_lr = args.lr
    this_alpha = args.alpha
    selected_participants_num = max(int(args.participation_rate * args.num_of_clients), 1)
    selected_participants = None
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
