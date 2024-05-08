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


def filter_samples(samples, cutoff):
    sos_lp = signal.butter(2, cutoff, 'lp', fs=100, output='sos')
    for i in range(samples.shape[1]): 
        filtered = signal.sosfilt(sos_lp, samples[:,i])
        samples[:,i] = filtered
    return samples


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, kaggle_root, long_term_root, cutoff):
        self.kaggle_root = kaggle_root
        self.long_term_room = long_term_root
        self.cutoff = cutoff
        self.sample_paths = []
        if kaggle_root:
            patients = [patient.split(',')[0] for patient in os.listdir(kaggle_root)]
            self.sample_paths.extend([os.path.join(kaggle_root, patient, sample) for patient in patients for sample in os.listdir(os.path.join(kaggle_root, patient))])
        if long_term_root:
            patients = [patient.split(',')[0] for patient in os.listdir(long_term_root)]
            self.sample_paths.extend([os.path.join(long_term_root, patient, sample) for patient in patients for sample in os.listdir(os.path.join(long_term_root, patient))])

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        path = self.sample_paths[index]
        samples = np.load(path)
        samples = filter_samples(samples, self.cutoff)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # generate samples and targets and mask_indices
        samples = torch.FloatTensor(samples).transpose(1, 0)
        flag = 0 if self.kaggle_root in path else 1
        return samples, flag

class LitModel_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.T = 0.2
        self.val_loss = 0
        self.model = UnsupervisedPretrain(emb_size=args.emb_size, heads=args.heads, depth=args.depth, n_channels=6, encoder_var=args.encoder_var) # 16 for PREST (resting) + 2 for SHHS (sleeping)
        
    def training_step(self, batch, batch_idx):

        # store the checkpoint every 5000 steps
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )

        contrastive_loss = 0
        kaggle_samples, long_term_samples = batch

        if len(long_term_samples) > 1:
            long_term_masked_emb, long_term_samples_emb = self.model(long_term_samples, 3)

            long_term_samples_emb = F.normalize(long_term_samples_emb, dim=1, p=2)
            long_term_masked_emb = F.normalize(long_term_masked_emb, dim=1, p=2)
            N = long_term_samples_emb.shape[0]

            # representation similarity matrix, NxN
            logits = torch.mm(long_term_samples_emb, long_term_masked_emb.t()) / self.T
            labels = torch.arange(N).to(logits.device)
            contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        if len(kaggle_samples) > 1:
            kaggle_masked_emb, kaggle_samples_emb = self.model(kaggle_samples, 0)

            kaggle_samples_emb = F.normalize(kaggle_samples_emb, dim=1, p=2)
            kaggle_masked_emb = F.normalize(kaggle_masked_emb, dim=1, p=2)
            N = kaggle_samples_emb.shape[0]

            # representation similarity matrix, NxN
            logits = torch.mm(kaggle_samples_emb, kaggle_masked_emb.t()) / self.T
            labels = torch.arange(N).to(logits.device)
            contrastive_loss += F.cross_entropy(logits, labels, reduction="mean")

        self.log("train_loss", contrastive_loss)
        return contrastive_loss

    def validation_step(self, batch, batch_idx):
        kaggle_samples, _ = batch

        kaggle_masked_emb, kaggle_samples_emb = self.model(kaggle_samples, 0)

        kaggle_samples_emb = F.normalize(kaggle_samples_emb, dim=1, p=2)
        kaggle_masked_emb = F.normalize(kaggle_masked_emb, dim=1, p=2)
        N = kaggle_samples_emb.shape[0]

        # representation similarity matrix, NxN
        logits = torch.mm(kaggle_samples_emb, kaggle_masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss = F.cross_entropy(logits, labels, reduction="mean")
        self.val_loss += contrastive_loss
        return contrastive_loss

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss)
        self.val_loss = 0
        

    def configure_optimizers(self):
        # set optimizer
        optimizer = mup.optim.MuAdam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        # set learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.3
        )

        return [optimizer], [scheduler]

def collate_fn_unsupervised_pretrain(batch):
    kaggle_samples, long_term_samples = [], []
    for sample, flag in batch:
        if flag == 0:
            kaggle_samples.append(sample)
        else:
            long_term_samples.append(sample)

    kaggle_samples = torch.stack(kaggle_samples, 0) if len(kaggle_samples) > 0 else []
    long_term_samples = torch.stack(long_term_samples, 0) if len(long_term_samples) > 0 else []

    return kaggle_samples, long_term_samples

def prepare_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # define the (daily life) data loader
    kaggle_root = args.kaggle_root
    long_term_root = args.long_term_root
    loader = UnsupervisedPretrainLoader(kaggle_root, long_term_root, args.cutoff)
    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn_unsupervised_pretrain,
    )
    
    return train_loader

def prepare_val_dataloader(args):
    root = "/home/workplace/thomas/BIOT/unlabeled_val"
    loader = UnsupervisedPretrainLoader(root, None, args.cutoff)
    val_loader = torch.utils.data.DataLoader(loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn_unsupervised_pretrain,
    )

    return val_loader

def pretrain(args):
    # get data loaders
    train_loader = prepare_dataloader(args)

    # define the trainer
    # N_version = (
    #     len(os.listdir(os.path.join("log-pretrain"))) + 1
    # )
    N_version = args.name
    # define the model
    save_path = f"log-pretrain/{N_version}-unsupervised/checkpoints"

    model = LitModel_supervised_pretrain(args, save_path)
    set_base_shapes(model, "/home/workplace/thomas/BIOT/base_shapes_LitModel.bsh")

    logger = TensorBoardLogger(
        save_dir="/home/workplace/thomas/scaling_law_tests",
        version=f"{N_version}/checkpoints",
        name="3epochs_with_long_term",
    )
    trainer = pl.Trainer(
        devices=[1],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
    )

    # train the model
    trainer.fit(model, train_loader)

class Args:
    epochs = 3
    lr = 0.0097
    encoder_var = 445.72
    weight_decay = 1e-5
    batch_size = 64
    num_workers = 32
    kaggle_root = "/home/workplace/thomas/BIOT/unlabeled_preprocessed" 
    long_term_root = "/home/workplace/thomas/BIOT/unlabeled_long_term_movement"
    emb_size = 32
    depth = 4
    heads = 8
    cutoff = 4.82
    name = "tuned_hyperparameter_test_32emb"


if __name__=="__main__":
    args = Args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_size", type=int, default=100, help="size of hidden layers")
    a = parser.parse_args()
    args.emb_size = a.emb_size
    args.name = f"tuned_hyperparameter_test_wiht_long_term{args.emb_size}emb"
    pretrain(args)
    
