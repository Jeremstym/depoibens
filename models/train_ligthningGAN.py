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
from lightning.pytorch.loggers import NeptuneLogger
# from neptune import ANONYMOUS_API_TOKEN
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
    "lr": 0.002,
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

    neptune_logger = NeptuneLogger(
    # api_key=ANONYMOUS_API_TOKEN,  
    project="jeremstym/DC-GAN",  
    tags=["training", "ST-Net"],  # optional
    )
    loggers = [
        pl.loggers.CSVLogger(path_to_log),
        neptune_logger,
        # pl.loggers.WandbLogger(project="GAN", log_model=True),
    ]
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        devices=[0, 1],
        max_epochs=EPOCHS,
        logger=loggers,
        log_every_n_steps=1
    )
    trainer.fit(dcgan, train_dataloaders=dataloader)

### -------------- Main -------------------------------

if __name__ == "__main__":
    main()

