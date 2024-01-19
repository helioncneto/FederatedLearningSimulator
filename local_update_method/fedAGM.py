import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.malicious import get_malicious_loader
from utils import DatasetSplit
import torch
import copy


class LocalUpdate:
    """
    Local training for FedAGM
    """
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0):
        self.lr = lr
        self.local_epoch = local_epoch
        self.device = device
        self.isCICIDS2017 = True if args.mode == "CICIDS2017" else False
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.batch_size = batch_size
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha = alpha
        self.args = args
        self.K = len(self.ldr_train)

    def train(self, net, delta=None, malicious=False):
        net.train()
        fixed_model = copy.deepcopy(net)
        for param_t in fixed_model.parameters():
            param_t.requires_grad = False
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        # Atualizar as amostras caso for malicioso
        self.ldr_train = get_malicious_loader(malicious, self.ldr_train, net, self.batch_size, self.args)

        # Train and update
        for epoch in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.ldr_train):
                x, labels = x.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = net(x)
                #if self.args.arch == "ResNet18":
                #    log_probs = net(x)
                #else:
                #    log_probs = net(x)
                ce_loss = self.loss_func(log_probs, labels)

                # print(log_probs, labels)
                # Weight L2 loss
                reg_loss = 0
                fixed_params = {n: p for n, p in fixed_model.named_parameters()}
                for n, p in net.named_parameters():
                    reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()

                loss = self.args.alpha * ce_loss + 0.5 * self.args.mu * reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def get_dataloader(self):
        return self.ldr_train