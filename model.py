import torch.nn as nn
import os
import torch
import numpy as np
import wandb
from torch.utils.data import Dataset
from torchvision import transforms


input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 4
batch_size = 100
l_r = 0.001
# Default hyper param


class NNForMnist(nn.Module):
    def __init__(self, args):
        super(NNForMnist, self).__init__()
        self.args = args
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size, self.num_classes)
        self.weight_init()

    def forward(self, inputs):
        out1 = self.l1(inputs)
        activated_out = self.relu(out1)
        out2 = self.l2(activated_out)
        return out2

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1)

    def log_init(self):
        config = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "lr": self.args.lr,
        }
        if self.args.use_log:
            wandb.login(host='http://47.108.152.202:8080',
                        key='local-86eb7fd9098b0b6aa0e6ddd886a989e62b6075f0')
            wandb.init(project='NNForMnist', config=config)
            os.system("wandb online")
        else:
            os.system("wandb offline")


# When manually load the data from the disk
class MnistDataset(Dataset):
    def __init__(self, pils):
        super(MnistDataset, self).__init__()
        # input type: list of pictures(PIL)
        self.samples = pils['X']
        self.labels = pils['Y']

    def __getitem__(self, index):
        return (self.samples[index], self.labels[index])

    def __len__(self):
        assert len(self.samples) == len(self.labels)
        return len(self.samples)


def collate_fn_for_hf(batch):
    """
    input type:
    batch: list of dicts, dict: iamge, label
    list of dicts, len(batch) equal to batch_size.
    """
    trans = transforms.ToTensor()
    batch_size = len(batch)
    samples = trans(np.array([np.array(s['image']) for s in batch])).to(torch.float32).permute(1, 0, 2)
    labels = torch.tensor([s['label'] for s in batch]).to(torch.long)
    samples = samples.reshape(-1, 28*28)
    return samples, labels


def collate_fn_for_manual(batch):
    # batch: list of ndarray
    # size: [b, 748]
    trans = transforms.ToTensor()
    samples = trans(np.array([x[0] for x in batch])).to(torch.float32).permute(1, 0, 2)
    labels = torch.tensor([x[1] for x in batch]).to(torch.long)
    return samples, labels
