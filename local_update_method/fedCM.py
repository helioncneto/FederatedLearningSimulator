import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import DatasetSplit, FedCM_SGD
import torch


class LocalUpdate(object):
    """
    Local training for FedCM
    """
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0):
        self.lr=lr
        self.local_epoch=local_epoch
        self.device=device
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha=alpha
        self.args=args
        self.K = len(self.ldr_train)

    def train(self, net, delta=None, malicious=False):
        net.train()

        # Local update via interpolation of local gradient and downloaded global gradient
        optimizer = FedCM_SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)
        epoch_loss = []

        # train and update
        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                if self.args.arch == "ResNet18":
                    log_probs = net(images)
                else:
                    log_probs= net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step(delta=delta, lamb=self.args.lamb)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def get_dataloader(self):
        return self.ldr_train