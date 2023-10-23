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
from importlib.machinery import SourceFileLoader

import pickle as pkl
import PIL
from PIL import Image
from glob import glob
import re
from tqdm import tqdm

sys.path.append('/import/bc_users/biocomp/stym/depoibens/stylegan3-repo')
dnnlib = SourceFileLoader("dnnlib", "../stylegan3-repo/dnnlib/__init__.py").load_module()
legacy = SourceFileLoader("legacy.name", "../stylegan3-repo/legacy.py").load_module()
# import dnnlib
# import legacy

PIL.Image.MAX_IMAGE_PIXELS = 933120000

# data constants
BATCH_SIZE = 64
VALID_SPLIT = 0.2
NUM_WORKERS = 4

tumor_path = "/projects/minos/jeremie/data/complete_concatenate_df.csv"
path_to_image = "/projects/minos/jeremie/data"
path_to_tsv = "/projects/minos/jeremie/data/tsv_concatened_allgenes.pkl"
path_to_generator = "/projects/minos/jeremie/data/styleGANresults/00078-stylegan2-styleImagesGen-gpus2-batch32-gamma0.2048/network-snapshot-021800.pkl"
path_to_generator_test = "/projects/minos/jeremie/data/styleGANresults/00083-stylegan2-styleImagesGen_test-gpus2-batch32-gamma0.2048/network-snapshot-005800.pkl"
path_to_generator_test2 = "/projects/minos/jeremie/data/styleGANresults/00060-stylegan2-styleImagesGen-gpus2-batch32-gamma0.2048/network-snapshot-006800.pkl"
path_to_generated_image = "/projects/minos/jeremie/data/generated_dict.pkl"
path_to_generated_image_test = "/projects/minos/jeremie/data/generated_dict_test.pkl"

test_patient = ["BT24223_D2", "BT24223_E1", "BT24223_E2"]

device = "cuda" if torch.cuda.is_available() else "cpu"


class TumoDataset(data.Dataset):
    def __init__(
        self,
        tumor_path: str,
        path_to_image: str,
        testing: bool = False,
    ) -> None:
        self.path = path_to_image
        self.dict = {}
        self.size = 256
        # self.dict_path = {}

        tumor_csv = pd.read_csv(tumor_path, index_col=0)
        self.tumor = tumor_csv["tumor"].apply(lambda x: (x == "tumor") * 1)

        os.chdir(self.path)
        with tqdm(glob("images/*/*.jpg"), unit="spot") as pbar:
            for image in pbar:
                img_path = image
                pattern = 'B[A-Z][0-9]+_[A-Z0-9]+_[0-9]+x[0-9]+'
                img_match = re.search(pattern, image)
                if img_match:
                    img_name = img_match.group(0)
                else:
                    raise ValueError(f"Image name {image} does not match pattern {pattern}")
                if testing:
                    if not any([img_name.startswith(patient) for patient in test_patient]):
                        continue
                else:
                    if any([img_name.startswith(patient) for patient in test_patient]):
                        continue
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
        path_to_generated_image: str,
    ) -> None:
        print("Loading generated images...")
        self.dict = pd.read_pickle(path_to_generated_image)
        print("Loading tumor labels...")
        self.tumor = pd.read_csv(tumor_path, index_col=0)["tumor"].apply(lambda x: (x == "tumor") * 1)

    def __len__(self):
        return len(self.dict.keys())

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
    seed: int = 42,
    testing: bool = False,
):
    dataset = TumoDataset(tumor_path, path_to_image, testing=testing)
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    if shuffle:
        # set seed
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    if not testing:
        train_sampler = data.SubsetRandomSampler(train_indices)
        valid_sampler = data.SubsetRandomSampler(valid_indices)
        train_loader = data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
        )
        valid_loader = data.DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
        )
        return train_loader, valid_loader, valid_sampler
    else:
        test_loader = data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers
        )
        return test_loader
        

def create_generated_dataloader(
    tumor_path: str = tumor_path,
    path_to_generated_image: str = path_to_generated_image,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    sampler: data.Sampler = None
):
    if sampler is None:
        print("No sampler provided, using default sampler")

    dataset = TumoGeneratedDataset(
        tumor_path=tumor_path, 
        path_to_generated_image=path_to_generated_image, 
    )
    dataloader = data.DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers
    )

    return dataloader

def create_generated_images_dataset(path_to_tsv: str, network_pkl: str, nb_genes: int = 900, testing: bool = False):

    def load_generator(network_pkl: str):
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        return G

    def load_tsv(path_to_tsv: str):
        print("Loading tsv...")
        with open(path_to_tsv, "rb") as f:
            tsv = pkl.load(f)
        tsv = tsv.drop("tissue", axis=1)[
            tsv.columns[:nb_genes]
        ]
        return tsv

    def preprocess(image: PIL.Image.Image):
        # os.chdir(path_to_image)
        size = 256
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
    
    dataset_dict = {}
    generator = load_generator(network_pkl)
    tsv = load_tsv(path_to_tsv)
    with tqdm(tsv.index, unit="spot") as pbar:
        for index in pbar:
            if testing:
                if not any([index.startswith(patient) for patient in test_patient]):
                    continue
            gene = tsv.loc[index].values
            with torch.no_grad():
                gene = torch.from_numpy(gene).unsqueeze(0).to(device)
                latent_vector = torch.from_numpy(np.random.RandomState(42).randn(1, 512)).to(device)
                generated_image = generator(latent_vector, gene, truncation_psi=1, noise_mode='const')
                generated_image = (generated_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                generated_image = generated_image.cpu().numpy()
                generated_image = Image.fromarray(generated_image[0], 'RGB')
                generated_image = preprocess(generated_image)

            dataset_dict[index] = generated_image

    return dataset_dict

# ------------------- #
# if __name__ == "__main__":
#     generated_dict = create_generated_images_dataset(path_to_tsv, path_to_generator)
#     with open("generated_dict.pkl", "wb") as f:
#         pkl.dump(generated_dict, f)

# if __name__ == "__main__":
#     generated_dict_test_wrong = create_generated_images_dataset(path_to_tsv, path_to_generator_test2, testing=False)
#     os.chdir("/projects/minos/jeremie/data")
#     with open("generated_dict_test_wrong.pkl", "wb") as f:
#         pkl.dump(generated_dict_test_wrong, f)