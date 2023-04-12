import pickle as pkl

from datasets import load_dataset
from torch.utils.data import DataLoader

from model import collate_fn_for_hf, collate_fn_for_manual, MnistDataset

trans_dataset = load_dataset("fashion_mnist")
loader = DataLoader(trans_dataset['train'], collate_fn=collate_fn_for_hf, batch_size=8)
"""for batch in loader:
    #print(batch)"""
print(trans_dataset['train'][0])

with open("./f_mnist/fashion_mnist_images_train.pkl", "rb") as f1:
    data_train = pkl.load(f1)

dataset_train = MnistDataset(
            data_train,
        )
train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_for_manual,
        )
for i in train_loader:
    print(i)
