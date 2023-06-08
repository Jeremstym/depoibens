#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dataset ot ST-Net

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
import torchvision
from torchvision.models.inception import Inception_V3_Weights
import torch.utils.data as data
import os

print(os.getcwd())
import sys

# setting path
sys.path.append("../")
import models.inception_STnet as inception_STnet

import pickle as pkl
from PIL import Image
from glob import glob
import re
from tqdm import tqdm

# model = inception_STnet.model
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
model.eval()
model.to(device)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model.fc = Identity()


### ---------------- Pre-processing for images ------------------


def image_embedding(path):
    cell = Image.open(path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(cell)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of 1 sample
    input_batch = input_batch.to(device)
    with torch.no_grad():
        output = model(input_batch)
    selection_tensor = torch.tensor(
        [[552, 1382, 1171, 699, 663, 1502, 588, 436, 1222, 617]]
    )
    return output[selection_tensor]


def embed_all_images(path):
    embeddings_dict = {}
    for sub_path in tqdm(glob(path + "/*/", recursive=True)):
        pbar = tqdm(glob(sub_path + "/*.jpg", recursive=True))
        for path_image in pbar:
            m = re.search("data/(.*)/(.*).jpg", path_image)
            if m:
                embeddings_dict[m.group(2)] = image_embedding(path_image)
            else:
                raise ValueError("Path not found")
            pbar.set_description(f"Processing {m.group(2)}")
    return embeddings_dict

# def embed_all_images(path):
#     embeddings = []
#     for sub_path in tqdm(glob(path + "/*/", recursive=True)):
#         for path_image in tqdm(glob(sub_path + "/*.jpg", recursive=True)):
#             embeddings.append(image_embedding(path_image))

#     return embeddings

# def image_embedding(path, model=model, device=device):
#     cell = Image.open(path)
#     preprocess = transforms.Compose(
#         [
#             transforms.Resize(299),
#             transforms.CenterCrop(299),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )
#     input_tensor = preprocess(cell)
#     input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of 1 sample
#     input_batch = input_batch.to(device)
#     with torch.no_grad():
#         output = model(input_batch)
#     return output

if __name__ == "__main__":
    path = "/import/pr_minos/jeremie/data"
    embeddings_dict = embed_all_images(path)
    with open(path + "/embeddings_dict.pkl", "wb") as f:
        pkl.dump(embeddings_dict, f)

### ---------------- Create dataset ------------------


class Phenotypes(data.Dataset):
    def __init__(self, path: str, model=model, device=device) -> None:
        super().__init__()
        self.path = path
        self.model = model
        self.device = device

        with open(path + "/embeddings_dict.pkl", "rb") as f:
            self.embeddings_dict = pkl.load(f)
        with open(path + "/std_genes_avg.pkl", "rb") as f:
            self.bestgene = list(pkl.load(f).index[:900])

        # self.preprocess = transforms.Compose(
        #     [
        #         transforms.Resize(299),
        #         transforms.CenterCrop(299),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )

        self.genotypes = self.concat_tsv()

    def __len__(self):
        return len(self.embeddings_dict)

    def __getitem__(self, index):
        return (
            torch.tensor(self.genotypes.iloc[index].values),
            self.embeddings_dict[index],
        )

    def tsv_processing(self, tissue_name: str, df: pd.DataFrame) -> pd.DataFrame:
        # mask = list(df.filter(regex="ambiguous"))
        # filtered_df = df[df.columns.drop(mask)]
        filtered_df = df[self.bestgene]
        filtered_df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
        filtered_df["id"] = filtered_df["id"].apply(lambda x: tissue_name + "_" + x)
        filtered_df.set_index("id", inplace=True)

        return filtered_df

    def concat_tsv(self):
        df = pd.DataFrame()
        pbar = tqdm(glob(self.path + "/*.tsv", recursive=True))
        for path_tsv in pbar:
            m = re.search("data/(.*)/(.*).tsv", path_tsv)
            if m:
                tissue_name = m.group(2)
                with open(path_tsv, "rb") as f:
                    df_tsv = pd.read_csv(f, sep="\t")
                df_tsv = self.tsv_processing(tissue_name, df_tsv)
                df = pd.concat([df, df_tsv])
            else:
                raise ValueError("Path not found")
            pbar.set_description(f"Processing {tissue_name}")
        return df

    # def image_embedding(self, path):
    #     cell = Image.open(path)
    #     input_tensor = self.preprocess(cell)
    #     input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of 1 sample
    #     input_batch = input_batch.to(self.device)
    #     with torch.no_grad():
    #         output = self.model(input_batch)
    #     selection_tensor = torch.tensor(
    #         [[552, 1382, 1171, 699, 663, 1502, 588, 436, 1222, 617]]
    #     )
    #     return output[selection_tensor]

    # def embed_all_images(self):
    #     embeddings_dict = {}
    #     for sub_path in tqdm(glob(self.path + "/*/", recursive=True)):
    #         pbar = tqdm(glob(sub_path + "/*.jpg", recursive=True))
    #         for path_image in tqdm(glob(sub_path + "/*.jpg", recursive=True)):
    #             m = re.search("data/(.*)/(.*).jpg", path_image)
    #             if m:
    #                 embeddings_dict[m.group(2)] = self.image_embedding(path_image)
    #             else:
    #                 raise ValueError("Path not found")
    #             pbar.set_description(f"Processing {m.group(2)}")
    #     return embeddings_dict


# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     st_set = Phenotypes(path)
#     torch.save(st_set, "data/st_set.pt")
