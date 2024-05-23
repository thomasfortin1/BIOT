import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import mup
from mup import set_base_shapes
from model import UnsupervisedPretrain

import time


def get_memmap_length(path):
    memmap = np.memmap(path, dtype=float, mode='r')
    return len(memmap)/3


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, root, shrink_factor=1):
        dats = []
        for folder in os.listdir(root):
            for file in os.listdir(os.path.join(root, folder)):
                if file[-3:] == 'dat':
                    dats.append(os.path.join(root, folder, file))
        self.dats = dats
        self.dats = self.dats[::shrink_factor]
        self.inds = np.cumsum([get_memmap_length(dat)//800 for dat in self.dats]).astype(int) - 1
        
    def __len__(self):
        return self.inds[-1]

    def __getitem__(self, index):
        i = np.searchsorted(self.inds, index)
        memmap = np.memmap(self.dats[i], dtype=float, mode = 'r', offset=3*8*800*(self.inds[i]-index), shape=(800, 3))
        samples = np.array(memmap).T
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples

class LitModel_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path, max_steps=-1, use_mup=True):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.max_steps = max_steps
        self.T = 0.2
        # self.T = 0.1 # for emb of 128 let's try this, it actually made the loss way lower somehow..
        self.val_losses = {emb_size: 0 for emb_size in args.emb_sizes}
        # self.model = UnsupervisedPretrain(emb_size=args.emb_size, heads=args.heads, depth=args.depth, n_channels=24, encoder_var=args.encoder_var, hop_length=args.hop_length, n_fft=args.n_fft, use_mup=use_mup) # 16 for PREST (resting) + 2 for SHHS (sleeping)
        self.models = {}
        for emb_size, heads in zip(args.emb_sizes, args.heads):
            self.models[emb_size] = UnsupervisedPretrain(emb_size=emb_size, heads=heads, depth=args.depth, n_channels=3, encoder_var=args.encoder_var, hop_length=args.hop_length, n_fft=args.n_fft, use_mup=use_mup)
            self.add_module(f"model_{emb_size}", self.models[emb_size])
            set_base_shapes(self.models[emb_size], "base_shapes_0.bsh")
            # send the model to the GPU now
            self.models[emb_size].to(args.device)

        
    def training_step(self, batch, batch_idx):

        # store the checkpoint every 100k steps or at the end of training
        if self.global_step % 100000 == 0 or self.global_step == self.max_steps-1:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )

        samples = batch
        contrastive_losses = []

        for emb_size in self.models:
            model = self.models[emb_size]
            masked_emb, samples_emb = model(samples)

            samples_emb = F.normalize(samples_emb, dim=1, p=2)
            masked_emb = F.normalize(masked_emb, dim=1, p=2)
            N = samples_emb.shape[0]
        
            # representation similarity matrix, NxN
            logits = torch.mm(samples_emb, masked_emb.t()) / self.T
            labels = torch.arange(N).to(logits.device)
            contrastive_losses.append(F.cross_entropy(logits, labels, reduction="mean"))

            self.log(f"train_loss_{emb_size}", contrastive_losses[-1])

        # print(self.optimizer.param_groups[0]["lr"])
        # sch = self.lr_schedulers()
        # sch.step()

        return sum(contrastive_losses)
    
    def validation_step(self, batch, batch_idx):
        samples = batch

        for emb_size in self.models:
            model = self.models[emb_size]
            masked_emb, samples_emb = model(samples)

            samples_emb = F.normalize(samples_emb, dim=1, p=2)
            masked_emb = F.normalize(masked_emb, dim=1, p=2)
            N = samples_emb.shape[0]

            # representation similarity matrix, NxN
            logits = torch.mm(samples_emb, masked_emb.t()) / self.T
            labels = torch.arange(N).to(logits.device)
            contrastive_loss = F.cross_entropy(logits, labels, reduction="mean")
            self.val_losses[emb_size] += contrastive_loss

    def on_validation_epoch_end(self):
        for emb_size in self.models:
            self.log(f"val_loss_{emb_size}", self.val_losses[emb_size])
            print(f"val_loss_{emb_size}: {self.val_losses[emb_size]}")
            self.val_losses[emb_size] = 0

    def configure_optimizers(self):
        # set optimizer
        optimizer = mup.optim.MuAdamW(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        # set learning rate scheduler
        # use a cosine annealing scheduler that reduces the learning rate by 10x over the course of training
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.max_steps, eta_min=self.args.lr/10
        # )

        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=10000, gamma=0.3
        # )

        self.optimizer = optimizer

        return [optimizer] #, [scheduler]


def prepare_dataloader(args, shuffle=True, shrink_factor=1, seed=42):
    # set random seed
    # seed = 12345
    # seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # define the (daily life) data loader
    root = args.root
    loader = UnsupervisedPretrainLoader(root, shrink_factor=shrink_factor)
    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        # num_workers=1,
        persistent_workers=True,
        drop_last=True
    )
    
    return train_loader


def pretrain(args):
    # get data loaders
    args.root = "/media/data_ssd/data/nhanes_val"
    val_loader = prepare_dataloader(args, shuffle=True)
    args.root = "/media/data_ssd/data/nhanes"
    train_loader = prepare_dataloader(args, shrink_factor=args.shrink_factor, seed=args.seed)

    # define the trainer
    # N_version = (
    #     len(os.listdir(os.path.join("log-pretrain"))) + 1
    # )
    N_version = args.name
    # define the model
    save_path = f"/media/data_ssd/results/nhanes_again/May_17/{N_version}-unsupervised/checkpoints"

    model = LitModel_supervised_pretrain(args, save_path, max_steps=args.steps, use_mup=True)

    logger = TensorBoardLogger(
        save_dir="/media/data_ssd/results/nhanes_again/",
        version=f"{N_version}/checkpoints",
        name="May_17",
    )

    trainer = pl.Trainer(
        devices=[args.device],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        enable_checkpointing=False,
        logger=logger,
        max_epochs=args.epochs,
        max_steps=args.steps,
        # profiler="simple",
        fast_dev_run=False,
    )

    # train the model
    trainer.fit(model, train_loader, val_loader)
    # trainer.validate(model, val_loader)

class Args:
    epochs = -1
    lr = 0.000388
    encoder_var = 1
    weight_decay = 1.1775e-5
    batch_size = 256
    num_workers = 32
    root = "/media/data_ssd/data/nhanes"
    # emb_size = 32
    emb_sizes = []
    depth = 4
    heads = 8
    n_fft = 80
    hop_length = n_fft//2
    device = 0
    steps = -1
    shrink_factor = 1
    seed = 42


if __name__=="__main__":
    args = Args()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--emb_size", type=int, default=100, help="size of hidden layers")
    parser.add_argument("--emb_sizes", type=int, nargs="+", default=[32, 64, 128, 256], help="size of hidden layers")
    parser.add_argument("--epochs", type=int, default=-1, help="number of epochs")
    parser.add_argument("--steps", type=int, default=-1, help="number of steps")
    parser.add_argument("--device", type=int, default=0, help="device to use")
    parser.add_argument("--lr", type=float, default=0.001132, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.00000371, help="weight decay")
    parser.add_argument("--shrink_factor", type=int, default=1, help="shrink factor for the dataset")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    a = parser.parse_args()
    if a.epochs == -1 and a.steps == -1:
        raise ValueError("Must specify either epochs or steps")
    args.emb_sizes = a.emb_sizes
    args.lr = a.lr
    args.weight_decay = a.weight_decay
    args.heads = [emb_size//8 for emb_size in args.emb_sizes]
    args.shrink_factor = a.shrink_factor
    args.epochs = a.epochs
    args.steps = a.steps
    # if a.epochs != -1:
    #     args.steps = a.epochs * 391700
    args.device = a.device
    args.seed = a.seed
    args.name = "nhanes_mup_test_" + "_".join([str(emb_size) for emb_size in args.emb_sizes]) + f"_{args.seed}"
    pretrain(args)
