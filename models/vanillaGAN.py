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


### -------------- Image processing ------------------
preprocess = transforms.Compose([
    transforms.Resize(250),
    transforms.CenterCrop(250),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
