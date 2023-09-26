#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

print(os.getcwd())
import sys

# setting path
sys.path.append("../")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
import torch.utils.data as data
import torchvision
from torchvision.models.inception import Inception_V3_Weights

import pickle as pkl
import PIL
from PIL import Image
from glob import glob
import re
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = 933120000

# data constants
BATCH_SIZE = 64
VALID_SPLIT = 0.2
NUM_WORKERS = 4

tumor_path = "/projects/minos/jeremie/data/complete_concatenate_df.csv"
path_to_image = "/projects/minos/jeremie/data"

device = "cuda" if torch.cuda.is_available() else "cpu"

class TumoDataset(data.Dataset):
    def __init__(
        self,
        tumor_path: str,
        path_to_image: str,
    ) -> None:
        self.path = path_to_image
        self.dict = {}
        # self.dict_path = {}

        tumor_csv = pd.read_csv(tumor_path, index_col=0)
        self.tumor = tumor_csv["tumor"].apply(lambda x: (x == "tumor") * 1)

        os.chdir(self.path)
        with tqdm(glob("*/*/*.jpg"), unit="spot") as pbar:
            for image in pbar:
                img = Image.open(image)
                img_name = image[19:-4]
                img_preprocessed = self.preprocess(img)
                self.dict[img_name] = img_preprocessed
                # self.dict_path[img_name] = image

    def preprocess(self, image):
        size = self.size
        preprocess = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        preprocessed_image = preprocess(image)
        return preprocessed_image
    
    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx_number: int):
        index = list(self.dict.keys())[idx_number]
        return self.dict[index], self.tumor[index]
