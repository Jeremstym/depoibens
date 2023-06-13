#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dataprocessing before GAN

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
import torch.utils.data as data

import pickle as pkl
from PIL import Image


### ---------------- Dataset of images ------------------------

class Phenotypes(data.Dataset):
    def __init__(self) -> None:
        super().__init__()

### ---------------- Pre-processing for images ------------------

batch_size = 16
preprocess = transforms.Compose([
    transforms.Resize(250),
    transforms.CenterCrop(250),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])