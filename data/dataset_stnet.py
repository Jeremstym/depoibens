#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dataset ot ST-Net

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
import models.inception_STnet as inception_STnet

import pickle as pkl
from PIL import Image
from glob import glob
import re
from tqdm import tqdm

model = inception_STnet.model
model.eval()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

### ---------------- Pre-processing for images ------------------

# selection_tensor = torch.tensor(
#         [[552, 1382, 1171, 699, 663, 1502, 588, 436, 1222, 617]]
#     )
# selection_tensor = selection_tensor.to(device)


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
    return output


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


# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     embeddings_dict = embed_all_images(path)
#     with open(path + "/embeddings_dict.pkl", "wb") as f:
#         pkl.dump(embeddings_dict, f)

### ---------------- Create dataset ------------------


class Phenotypes(data.Dataset):
    def __init__(self, path: str, model=model, device=device) -> None:
        super().__init__()
        self.path = path
        self.model = model
        self.device = device
        self.selection_tensor = torch.tensor(
            [[552, 1382, 1171, 699, 663, 1502, 588, 436, 1222, 617]]
        ).to(self.device)

        with open(path + "/embeddings_dict.pkl", "rb") as f:
            self.embeddings_dict = pkl.load(f)
        with open(path + "/std_genes_avg.pkl", "rb") as f:
            self.bestgene = list(pkl.load(f).index[:900])

        self.genotypes = self.concat_tsv()

    def __len__(self):
        return len(self.embeddings_dict)

    def __getitem__(self, index):
        return (
            torch.tensor(self.genotypes.iloc[index].values),
            self.embeddings_dict[index][self.selection_tensor],
        )

    def tsv_processing(self, tissue_name: str, df: pd.DataFrame) -> pd.DataFrame:
        mask = list(df.filter(regex="ambiguous"))
        filtered_df = df[df.columns.drop(mask)]
        filtered_df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
        filtered_df["id"] = filtered_df["id"].apply(lambda x: tissue_name + "_" + x)
        filtered_df.set_index("id", inplace=True)

        return filtered_df[self.bestgene]

    def concat_tsv(self):
        df = pd.DataFrame()
        pbar = tqdm(glob(self.path + "/*/*[0-9].tsv", recursive=True))
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
        print(df)
        return df


if __name__ == "__main__":
    path = "/import/pr_minos/jeremie/data"
    st_set = Phenotypes(path)
    torch.save(st_set, "data/st_set.pt")

### ---------------- Brouillon ------------------

# minipath = r"E:\ST-Net\data\hist2tscript\BRCA"
# for path_tsv in glob(minipath + "\*\*[0-9].tsv", recursive=True):
#     m = re.search(r"BRCA\\(.*)\\(.*).tsv", path_tsv)
#     if m:
#         tissue_name = m.group(2)
#         print(tissue_name)
#     break
