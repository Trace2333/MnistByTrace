import logging
import os.path
import pickle as pkl

import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from get_args import get_args
from prepare_data import prepare
from model import MnistDataset, NNForMnist, collate_fn_for_hf, collate_fn_for_manual


def train(args, device):
    # init the model
    model = NNForMnist(args).to(device)
    # init the log func
    model.log_init()

    if args.hf_dataset:
        dataset = load_dataset("fashion_mnist")
        # Auto load and download,split as you wish.
        train_loader = DataLoader(
            dataset=dataset['train'],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_for_hf,
            # note that the fn shouldn't be instanced
        )
        test_loader = DataLoader(
            dataset=dataset['test'],
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_for_hf,
        )
    else:
        if os.path.exists('./f_mnist/fashion_mnist_images_train.pkl'):
            prepare()

        # if changed dataset, please change the file path to the data...
        with open("./f_mnist/fashion_mnist_images_train.pkl", "rb") as f1:
            data_train = pkl.load(f1)

        with open("./f_mnist/fashion_mnist_images_test.pkl", "rb") as f1:
            data_test = pkl.load(f1)

        dataset_train = MnistDataset(
            data_train,
        )
        dataset_test = MnistDataset(
            data_test
        )

        train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_for_manual,
        )
        test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_for_manual,
        )

    if args.optim_type == "SGD":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=args.lr,
        )
    else:
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr,
        )

    if args.loss_type == "CrossEntropyLoss":
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss()

    logging.info("On TRAINING Loop...")
    model.train()
    for epoch in range(args.epochs):
        t_l = tqdm(enumerate(train_loader), desc="Training")
        for t, batch in t_l:
            # tensor in batch is default config, should change device and type.
            x, y = batch
            # no device...
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            optimizer.zero_grad()
            loss = loss_func(out, y)
            train_p = torch.sum(out.argmax(-1) == y) / args.batch_size
            t_l.set_postfix(epoch=epoch, train_loss=loss.item(), train_precision=train_p)
            if args.use_log:
                wandb.log({"loss": loss, "precision": train_p})
            loss.backward()
            optimizer.step()

    logging.info("On TESTING Loop...")
    model.eval()
    for epoch in range(args.test_epochs):
        test_l = tqdm(enumerate(test_loader), desc="Testing")
        for t, batch in test_l:
            x, y = batch
            # no device...
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            test_p = torch.sum(out.argmax(-1) == y) / args.test_batch_size
            test_l.set_postfix(epoch=epoch, test_precision=test_p)
            if args.use_log:
                wandb.log({"test_precision": test_p})


if __name__ == '__main__':
    args = get_args()
    if args.device_type == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # no TPU, HPU availabl...
        device = torch.device('cpu')
    train(args=args,
          device=device)
