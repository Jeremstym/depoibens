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

model = torch.load("../models/outputs/best_model_dino.pth")
model.eval()
print(model)