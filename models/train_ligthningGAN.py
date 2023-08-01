#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Train Lightning GAN for transcriptomics


import os
import sys
sys.path.append("../")

import pickle as pkl
import argparse

import torch
from models.lightning_GAN import GAN
import pytorch_lightning as pl 
from torchvision import transforms
import torchvision.datasets as dset

### -------------- Pathes ------------------

path_to_data = "/projects/minos/jeremie/data/images"
path_to_log = "/projects/minos/jeremie/data/logs"
path_save = "/projects/minos/jeremie/data/GANresults"

### -------------- Hyperparameters ------------------

hparams = {
    "nz": 100,
    "ngf": 64,
    "ndf": 64,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
}

BATCH_SIZE = 64
EPOCHS = 10
def main():
    dcgan = GAN(**hparams)
    ### -------------- Load data -------------------------------
    preprocess = transforms.Compose(
        [
            transforms.Resize((300,300)),
            transforms.RandomCrop(300),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = dset.ImageFolder(root=path_to_data, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    print("Data loaded")

    ### -------------- Train -------------------------------
    loggers = [
        pl.loggers.CSVLogger(path_to_log),
        # pl.loggers.WandbLogger(project="GAN", log_model=True),
    ]
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=[0, 1],
        max_epochs=EPOCHS,
        logger=loggers,
    )
    trainer.fit(dcgan, train_dataloaders=dataloader)

### -------------- Main -------------------------------

if __name__ == "__main__":
    main()

