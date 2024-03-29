import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import DatasetSplit, IL, IL_negsum
from utils.malicious import get_malicious_loader
from local_update_method.global_and_online_model import *
from libs.dataset.dataset_factory import NUM_CLASSES_LOOKUP_TABLE

from utils.data import FakeCICIDS2017Dataset
import torch


class LocalUpdate:
    """
    Base local training
    Local objective function contains only local loss function.
    """
    def __init__(self, args, lr, local_epoch, device, batch_size, dataset=None, idxs=None, alpha=0.0):
        self.lr = lr
        self.local_epoch = local_epoch
        self.device = device
        self.batch_size = batch_size
        self.oneclass = True if NUM_CLASSES_LOOKUP_TABLE[args.set] <= 1 else False
        if self.oneclass:
            self.loss_func = nn.BCELoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.alpha = alpha
        self.args = args
        self.K = len(self.ldr_train)

    def train(self, net, delta=None, malicious=False):
        model = net
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay)
        epoch_loss = []
        # Atualizar as amostras caso for malicioso
        self.ldr_train = get_malicious_loader(malicious, self.ldr_train, model, self.batch_size, self.args)

        # train and update
        for _ in range(self.local_epoch):
            batch_loss = torch.tensor([], requires_grad=False).to(self.device)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = model(images)

                if self.oneclass:
                    loss = self.loss_func(log_probs, labels.float())
                else:
                    loss = self.loss_func(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gr_clipping_max_norm)
                optimizer.step()
                #batch_loss.append(loss.item())
                batch_loss = torch.cat((batch_loss, loss.unsqueeze(0)), 0)
            #epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_loss.append((torch.sum(batch_loss) / batch_loss.size(0)).item())
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def get_dataloader(self):
        return self.ldr_train