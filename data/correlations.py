#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Analysis of correlation

import os
import sys

# setting path
sys.path.append("../")
import pickle as pkl
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from models.inception_STnet import Regression_STnet

INPUT_SIZE = 900
OUTPUT_SIZE = 768
HIDDEN_SIZE = 1536
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Regression_STnet(
    input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE
)

model.load_state_dict(
    torch.load(
        "/projects/minos/jeremie/data/outputs/best_model_dino.pth", map_location=torch.device(device)
    )["model_state_dict"]
)
model.eval()
model.to(device)
print(model)
