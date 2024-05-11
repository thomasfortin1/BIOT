import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import signal
from scipy.signal import resample
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
import glob



def get_memmap_length(path):
    memmap = np.memmap(path, dtype=float, mode='r')
    return len(memmap)/24


def get_channel_diffs(signals):
    diffs = np.zeros((16, 2560))
    diffs[0, :] = signals[0, :] - signals[9, :]
    diffs[1, :] = signals[9, :] - signals[11, :]
    diffs[2, :] = signals[11, :] - signals[13, :]
    diffs[3, :] = signals[13, :] - signals[7, :]
    diffs[4, :] = signals[1, :] - signals[10, :]
    diffs[5, :] = signals[10, :] - signals[12, :]
    diffs[6, :] = signals[12, :] - signals[14, :]
    diffs[7, :] = signals[14, :] - signals[8, :]
    diffs[8, :] = signals[0, :] - signals[15, :]
    diffs[9, :] = signals[15, :] - signals[3, :]
    diffs[10, :] = signals[3, :] - signals[5, :]
    diffs[11, :] = signals[5, :] - signals[7, :]
    diffs[12, :] = signals[1, :] - signals[2, :]
    diffs[13, :] = signals[2, :] - signals[4, :]
    diffs[14, :] = signals[4, :] - signals[6, :]
    diffs[15, :] = signals[6, :] - signals[8, :]
    return diffs


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, root, shrink_factor=1, transform=None):
        dats = []
        dats = glob.glob(os.path.join(root, '*.dat'))
        self.dats = dats
        self.dats = self.dats[::shrink_factor]
        self.inds = np.cumsum([get_memmap_length(dat)//2560 for dat in self.dats]).astype(int) - 1
        self.transform = transform
        
    def __len__(self):
        return self.inds[-1] + 1

    def __getitem__(self, index):
        i = np.searchsorted(self.inds, index)
        memmap = np.memmap(self.dats[i], dtype=float, mode = 'r', offset=24*2560*8*(self.inds[i]-index), shape=(2560, 24))
        samples = np.array(memmap).T

        if self.transform:
            samples = get_channel_diffs(samples)

        samples = resample(samples, 2000, axis=-1)

        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # samples = torch.FloatTensor(samples).transpose(1, 0)
        samples = torch.FloatTensor(samples)
        return samples


class LitModel_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path, max_steps=-1, use_mup=False):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.max_steps = max_steps
        self.T = 0.2
        # self.T = 0.1 # for emb of 128 let's try this, it actually made the loss way lower somehow..
        self.val_loss = 0
        self.model = UnsupervisedPretrain(emb_size=args.emb_size, heads=args.heads, depth=args.depth, n_channels=16, encoder_var=args.encoder_var, hop_length=args.hop_length, n_fft=args.n_fft, use_mup=use_mup) # 16 for PREST (resting) + 2 for SHHS (sleeping)
        
    def training_step(self, batch, batch_idx):

        # store the checkpoint every 40k steps or at the end of training
        if self.global_step % 40000 == 0 or self.global_step == self.max_steps-5:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )

        contrastive_loss = 0
        samples = batch

        masked_emb, samples_emb = self.model(samples)

        samples_emb = F.normalize(samples_emb, dim=1, p=2)
        masked_emb = F.normalize(masked_emb, dim=1, p=2)
        N = samples_emb.shape[0]

        # representation similarity matrix, NxN
        logits = torch.mm(samples_emb, masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        self.log("train_loss", contrastive_loss)

        # print(self.optimizer.param_groups[0]["lr"])
        # sch = self.lr_schedulers()
        # sch.step()

        # if self.global_step % 100 == 0:
        #     print(f"train_loss: {contrastive_loss}")

        return contrastive_loss
    
    def validation_step(self, batch, batch_idx):
        samples = batch

        masked_emb, samples_emb = self.model(samples, 0)

        samples_emb = F.normalize(samples_emb, dim=1, p=2)
        masked_emb = F.normalize(masked_emb, dim=1, p=2)
        N = samples_emb.shape[0]

        # representation similarity matrix, NxN
        logits = torch.mm(samples_emb, masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss = F.cross_entropy(logits, labels, reduction="mean")
        self.val_loss += contrastive_loss

        # return contrastive_loss

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss)
        print(f"val_loss: {self.val_loss}")
        self.val_loss = 0

    def configure_optimizers(self):
        # set optimizer
        optimizer = mup.optim.MuAdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

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


def prepare_dataloader(args, shuffle=True, shrink_factor=1, seed=42, transform=None):
    # set random seed
    # seed = 12345
    # seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # define the (daily life) data loader
    root = args.root
    loader = UnsupervisedPretrainLoader(root, shrink_factor=shrink_factor, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=True
    )
    
    return train_loader


def pretrain(args):
    # get data loaders
    args.root = "/media/data_ssd/data/eeg_dat_val"
    val_loader = prepare_dataloader(args, shuffle=True, transform='16_diffs') # This is actually going to make it shuffle every epoch, not just once which is not what I want to do.
    args.root = "/media/data_ssd/data/eeg_dat"

    train_loader = prepare_dataloader(args, shrink_factor=args.shrink_factor, seed=args.seed, shuffle=True, transform='16_diffs')



    # define the trainer
    # N_version = (
    #     len(os.listdir(os.path.join("log-pretrain"))) + 1
    # )
    N_version = args.name
    # define the model
    #TODO: change the path to work with s3

    save_path = f"/media/data_ssd/results/eeg/May_10_checkpoints/{N_version}-unsupervised/checkpoints"

    model = LitModel_supervised_pretrain(args, save_path, max_steps=args.steps, use_mup=True)
    rescale_params=True
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict["state_dict"])
        print('load  from ', args.checkpoint)
        rescale_params=False

    
    set_base_shapes(model, "base_shapes_LitModel.bsh", rescale_params=rescale_params)

    logger = TensorBoardLogger(
        save_dir="/media/data_ssd/results/eeg",
        version=f"{N_version}/checkpoints",
        name="May_10",
    )

    trainer = pl.Trainer(
        # devices=[0, 1],
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
    # trainer.validate(model, val_loader)
    trainer.fit(model, train_loader, val_loader)

class Args:
    epochs = -1
    lr = 0.00140966
    encoder_var = 1
    weight_decay = 0.00013458
    batch_size = 128
    num_workers = 32
    root = "/media/data_ssd/data/eeg_dat"
    emb_size = 32
    depth = 4
    heads = emb_size//8
    n_fft = 200
    hop_length = n_fft//2
    # device = 0
    steps = -1
    shrink_factor = 1
    seed = 42
    checkpoint = None


if __name__=="__main__":

    args = Args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_size", type=int, default=100, help="size of hidden layers")
    parser.add_argument("--epochs", type=int, default=-1, help="number of epochs")
    parser.add_argument("--steps", type=int, default=-1, help="number of steps")
    # parser.add_argument("--device", type=int, default=0, help="device to use")
    parser.add_argument("--lr", type=float, default=0.00140966, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.00013458, help="weight decay")
    parser.add_argument("--shrink_factor", type=int, default=1, help="shrink factor for the dataset")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--depth", type=int, default=4, help="depth of the model")
    a = parser.parse_args()
    if a.epochs == -1 and a.steps == -1:
        raise ValueError("Must specify either epochs or steps")
    args.emb_size = a.emb_size
    args.lr = a.lr
    args.weight_decay = a.weight_decay
    args.heads = args.emb_size//8
    args.shrink_factor = a.shrink_factor
    args.epochs = a.epochs
    args.steps = a.steps
    args.depth = a.depth
    # if a.epochs != -1:
    #     args.steps = a.epochs * 391700
    # args.device = a.device
    args.seed = a.seed
    args.name = f"May_10_{args.emb_size}"
    # args.checkpoint = "/media/data_ssd/results/eeg/May_9_checkpoints/May_9_512-unsupervised/checkpoints/epoch=7_step=120000.ckpt"
    pretrain(args)
