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

sys.path.append('/import/bc_users/biocomp/stym/depoibens/stylegan3-repo')
import dnnlib
import legacy

PIL.Image.MAX_IMAGE_PIXELS = 933120000

# data constants
BATCH_SIZE = 64
VALID_SPLIT = 0.2
NUM_WORKERS = 4

tumor_path = "/projects/minos/jeremie/data/complete_concatenate_df.csv"
path_to_image = "/projects/minos/jeremie/data"
path_to_tsv = "/projects/minos/jeremie/data/tsv_concatened_allgenes.pkl"
path_to_generator = "/projects/minos/jeremie/data/styleGANresults/00078-stylegan2-styleImagesGen-gpus2-batch32-gamma0.2048/network-snapshot-021800.pkl"

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
                img_path = image
                pattern = 'B[A-Z][0-9]+_[A-Z0-9]+_[0-9]+x[0-9]+'
                img_match = re.search(pattern, image)
                if img_match:
                    img_name = img_match.group(0)
                else:
                    raise ValueError(f"Image name {image} does not match pattern {pattern}")
                # img_name = image[18:-4]
                # img_preprocessed = self.preprocess(img)
                self.dict[img_name] = img_path
                # self.dict_path[img_name] = image

    def preprocess(self, image_path: str):
        os.chdir(self.path)
        image = Image.open(image_path)
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
        img_preprocessed = self.preprocess(self.dict[index])
        return img_preprocessed, self.tumor[index]


class TumoGeneratedDataset(data.Dataset):
    def __init__(
        self,
        tumor_path: str,
        path_to_image: str,
        path_to_tsv: str,
        path_to_generator: str,
        nb_genes: int = 900
    ) -> None:
        self.path = path_to_image
        self.size = 256
        self.latent_dim = 512
        self.generator = self.load_generator(path_to_generator)
        self.tsv = self.load_tsv(path_to_tsv)

        tumor_csv = pd.read_csv(tumor_path, index_col=0)
        self.tumor = tumor_csv["tumor"].apply(lambda x: (x == "tumor") * 1)

    def load_generator(network_pkl: str):
        print('Loading networks from "%s"...' % network_pkl)
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        return G

    def generate_image(self, gene: np.ndarray = None):
        if gene is None:
            print("WARNING: No gene provided, generating random image")
        with torch.no_grad():
            latent_vector = torch.from_numpy(np.random.RandomState(42).randn(1, self.latent_dim)).to(device)
            generated_image = self.generator(latent_vector, gene, truncation_psi=1, noise_mode='const')
        return generated_image
    
    def load_tsv(path_to_tsv: str, nb_genes: int = 900):
        with open(path_to_tsv, "rb") as f:
            tsv = pkl.load(f)
        tsv = tsv.drop("tissue", axis=1)[
            tsv.columns[:nb_genes]
        ]
        return tsv

    def preprocess(self, image_path: str):
        os.chdir(self.path)
        image = Image.open(image_path)
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
        index = list(self.tsv.index)[idx_number]
        gene = self.tsv.loc[index].to_numpy()
        gen_image = self.generate_image(gene)
        img_preprocessed = self.preprocess(gen_image)
        return img_preprocessed, self.tumor[index]

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

def create_generated_dataloader(
    tumor_path: str = tumor_path,
    path_to_image: str = path_to_image,
    path_to_tsv: str = path_to_tsv,
    path_to_generator: str = path_to_generator,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    dataset = TumoGeneratedDataset(tumor_path, path_to_image, path_to_tsv, path_to_generator)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )

    return dataloader