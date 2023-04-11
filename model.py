import torch.nn as nn
import os
import torch
import wandb
from torch.utils.data import Dataset


input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 4
batch_size = 100
l_r = 0.001
# Default hyper param


class NNForMnist(nn.Module):
    def _init_(self, args):
        super(NNForMnist, self).__init__()
        self.args = args
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, inputs):
        out1 = self.l1(inputs)
        activated_out = self.relu(out1)
        out2 = self.l2(activated_out)
        return out2

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
    def __init__(self, pils, labels):
        super(MnistDataset, self).__init__()
	# input type: list of pictures(PIL)
	self.samples = pils
	self.labels = labels

    def __getitem__(self, index):
	return (self.samples[index], self.labels[index])

    def __len__(self)
	assert len(self.samples) == len(self.labels)
	return len(self.samples)
	

def collate_fn_for_pils(batch):
	# input type: 
	# 	batch:train_samples, train_labels
	#	list of tuples.
    batch_size = len(batch)
    samples = [torch.tensor(s[0]) for s in batch]
    labels = [s[1] for s in batch]
    labels = torch.tensor(labels, dtype=torch.int)
    return samples, labels
    
    


