#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Vanilla GAN for transcriptomics

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms

import pickle as pkl
from PIL import Image

import matplotlib.pyplot as plt


### -------------- Constants ------------------

# Path to data


# Hyperparameters
nz = 100  # size of latent vector
ngf = 64  # size of feature maps in generator
ndf = 64  # size of feature maps in discriminator


### -------------- Image processing ------------------
preprocess = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

### ----------- Networks ----------------------------


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=ngf * 8,
                kernel_size=16,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=ngf * 8,
                out_channels=ngf * 4,
                kernel_size=32,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 60 x 60
            nn.ConvTranspose2d(
                in_channels=ngf * 4,
                out_channels=ngf * 2,
                kernel_size=32,
                stride=4,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 148 x 148
            nn.ConvTranspose2d(
                in_channels=ngf * 2,
                out_channels=ngf,
                kernel_size=16,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 308 x 308
            nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=4,
                bias=False,
            ),
            nn.Tanh()
            # output size. 1 x 300 x 300
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is 1 x 300 x 300
            nn.Conv2d(
                in_channels=1,
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 150 x 150
            nn.Conv2d(
                in_channels=ndf,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=4,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 37 x 37
            nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


### -------------- Brouillon -----------------------

# class Regressor(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Linear(1000, 640),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(640, 280),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(280, 64),
#             nn.LeakyReLU(0.2, inplace=True),
#             # Deconvolutional part
#             nn.ConvTranspose2d(64, 32, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 1, 1, 2, bias=False),
#         )

#     def forward(self, input):
#         return self.main(input)
