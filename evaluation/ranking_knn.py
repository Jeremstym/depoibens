#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Ranking k-NN

import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import pickle
from tqdm import tqdm
import importlib

sys.path.append('/import/bc_users/biocomp/stym/depoibens/stylegan3-repo')
generator = importlib.import_module("stylegan3-repo.gen_images")

# Constants

path_to_reals = "/projects/minos/jeremie/data/"
path_to_fakes = "/projects/minos/jeremie/data/generated_dict.pkl"
path_to_model = "/projects/minos/jeremie/data/styleGANresults/00078-stylegan2-styleImagesGen-gpus2-batch32-gamma0.2048/network-snapshot-021800.pkl"


