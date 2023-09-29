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
        self.size = 256
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


def create_dataloader(
    tumor_path: str,
    path_to_image: str,
    batch_size: int = BATCH_SIZE,
    valid_split: float = VALID_SPLIT,
    num_workers: int = NUM_WORKERS,
    shuffle: bool = True,
):
    dataset = TumoDataset(tumor_path, path_to_image)
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = data.SubsetRandomSampler(train_indices)
    valid_sampler = data.SubsetRandomSampler(valid_indices)

    train_loader = data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )
    return train_loader, valid_loader
