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
import torchvision
from torchvision.models.inception import Inception_V3_Weights

import pickle as pkl
import PIL
from PIL import Image
from glob import glob
import re
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = 933120000

### ---------------- Customized Inception model ------------------------

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model.fc = Identity()

model.eval()
model.to(device)
model.eval()

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
        pbar = tqdm(glob(sub_path + "/*/*.jpg", recursive=True))
        for path_image in pbar:
            m = re.search("data/(.*)/(.*).jpg", path_image)
            if m:
                embeddings_dict[m.group(2)] = image_embedding(path_image)
            else:
                raise ValueError("Path not found")
            pbar.set_description(f"Processing {m.group(2)}")
    return embeddings_dict


def change_device_embedding(embeddings_dict):
    for key in embeddings_dict.keys():
        embeddings_dict[key] = embeddings_dict[key].to("cpu")
    return embeddings_dict


# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     embeddings_dict = embed_all_images(path)
#     embeddings_dict = change_device_embedding(embeddings_dict)
#     with open(path + "/embeddings_dict.pkl", "wb") as f:
#         pkl.dump(embeddings_dict, f)


# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     embeddings_dict = embed_all_images(path)
#     with open(path + "/embeddings_dict2.pkl", "wb") as f:
#         pkl.dump(embeddings_dict, f)

## Deprecated version below
# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     embeddings = embed_all_images(path)
#     with open(path + "/embeddings.pkl", "wb") as f:
#         pkl.dump(embeddings, f)

### ---------------- Pre-processing for tsv files ------------------


def tsv_processing(
    tissue_name: str,
    df: pd.DataFrame,
    true_index: pd.core.indexes.base.Index,
    bestgene: list,
) -> pd.DataFrame:
    mask = list(df.filter(regex="ambiguous", axis=1))
    filtered_df = df[df.columns.drop(mask)]
    filtered_df = filtered_df.rename(columns={"Unnamed: 0": "id"})
    filtered_df = filtered_df[filtered_df["id"].isin(true_index)]
    filtered_df["id"] = filtered_df["id"].apply(lambda x: tissue_name + "_" + x)
    filtered_df = filtered_df.set_index("id")
    filtered_df["tissue"] = tissue_name

    return filtered_df[bestgene + ["tissue"]]


def concat_tsv(path: str, bestgene: list) -> pd.DataFrame:
    df = pd.DataFrame()
    pbar = tqdm(glob(path + "/*/*[0-9].tsv", recursive=True))
    # pbar = tqdm(glob(path + "/*/*", recursive=True))
    for subpath in pbar:
        m = re.search("data/(.*)/(.*).tsv", subpath)
        m2 = re.sub(".tsv", "_complete.pkl", subpath)
        with open(m2, "rb") as f:
            df_complete = pkl.load(f)
            true_index = list(df_complete.index)
        if m:
            tissue_name = m.group(2)
            with open(subpath, "rb") as f:
                df_tsv = pd.read_csv(f, sep="\t")
            df_tsv = tsv_processing(tissue_name, df_tsv, true_index, bestgene)
            df = pd.concat([df, df_tsv])
        else:
            raise ValueError("Path not found for m")
        pbar.set_description(f"Processing {tissue_name}")

    return df


# if __name__ == "__main__":
#     path = "/projects/minos/jeremie/data"
#     with open(path + "/std_genes_avg.pkl", "rb") as f:
#         bestgene = list(pkl.load(f).index[:900])
#     tsv_concatened = concat_tsv(path, bestgene)
#     with open(path + "/tsv_concatened3.pkl", "wb") as f:
#         pkl.dump(tsv_concatened, f)

#     df = pd.DataFrame()
#     for sub_path in tqdm(glob(path + "/*/", recursive=True)):
#         pbar = tqdm(glob(sub_path + "/*.tsv", recursive=True))
#         for path_tsv in pbar:
#             m = re.search("data/(.*)/(.*).tsv", path_tsv)
#             if m:
#                 df = pd.concat(
#                     [
#                         df,
#                         tsv_processing(
#                             m.group(2), pd.read_csv(path_tsv, sep="\t"), bestgene
#                         ),
#                     ]
#                 )
#             else:
#                 raise ValueError("Path not found")
#             pbar.set_description(f"Processing {m.group(2)}")
#     return df


# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     with open(path + "/std_genes_avg.pkl", "rb") as f:
#         bestgene = list(pkl.load(f).index[:900])
#     df = concat_tsv(path, bestgene)
#     with open(path + "/tsv_concatened.pkl", "wb") as f:
#         pkl.dump(df, f)

### ---------------- Create dataset ------------------


class Phenotypes(data.Dataset):
    def __init__(self, tsv_concatened, embeddings_dict, model=model) -> None:
        super().__init__()
        self.genotypes = tsv_concatened
        self.embeddings_dict = embeddings_dict
        self.model = model
        # self.device = device
        self.selection_list = [552, 1382, 1171, 699, 663, 1502, 588, 436, 1222, 617]

    def __len__(self):
        return len(self.embeddings_dict)

    def __getitem__(self, idx_number: int):
        index = list(self.embeddings_dict.keys())[idx_number]
        return (
            torch.tensor(self.genotypes.loc[index].values),
            self.embeddings_dict[index][0, self.selection_list],
        )


### ---------------- Brouillon ------------------

# minipath = r"E:\ST-Net\data\hist2tscript\BRCA"
# for path_tsv in glob(minipath + "\*\*[0-9].tsv", recursive=True):
#     m = re.search(r"BRCA\\(.*)\\(.*).tsv", path_tsv)
#     if m:
#         tissue_name = m.group(2)
#         print(tissue_name)
#     break

# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     st_set = Phenotypes(path)
#     torch.save(st_set, path + "/st_set.pt")


# path = r"E:\ST-Net\data\hist2tscript\BRCA"
# def embed_all_images(path):
#     embeddings_dict = {}
#     for sub_path in glob(path + "/*/", recursive=True):
#         pbar = glob(sub_path + "/*/*.jpg", recursive=True)
#         for path_image in pbar:
#             m = re.search(r"hist2tscript\\(.*)\\(.*).jpg", path_image)
#             print(path_image)
#             if m:
#                 print(m.group(2))
#                 break
#                 # embeddings_dict[m.group(2)] = image_embedding(path_image)
#             else:
#                 raise ValueError("Path not found")


# embed_all_images(path)


# my_tensor = torch.randn(1, 10)
# my_tensor[0, [1, 2]]


# path = r"E:\ST-Net\data\hist2tscript\BRCA"

# with open(path + "\BC23270\BC23270_D2.tsv", "r") as f:
#     try_tsv = pd.read_csv(f, sep="\t")

# try_tsv.set_index("Unnamed: 0", inplace=True)

# try_tsv.sample(10, weights=try_tsv.index)

# try_tsv[try_tsv["Unnamed: 0"].str.endswith("x34")]

# with open(path + "\BC23270\BC23270_D2.spots.txt", "r") as f:
#     try_coords = pd.read_csv(f, sep=",")

# try_coords.set_index("Unnamed: 0", inplace=True)
# try_coords
# try_coords.loc['18x18']

# with open(path + "\BC23270\BC23270_D2_complete.pkl", "rb") as f:
#     try_coords = pkl.load(f)

# try_tsv.loc[try_coords.index]

# try_coords.loc['18x18']

# try_tsv[~try_tsv.index.str.endswith('x12')]
# list(try_tsv[try_tsv.index.str.endswith('x12')].index)

# my_dict = {"oui":0, "non":1, "peut-être":2}

# {k: my_dict[k] for k in set(list(my_dict.keys())) - set(["oui"])}
