#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dataset ot ST-Net

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
import torch.utils.data as data

from inception_STnet import model, device

import pickle as pkl
from PIL import Image
from glob import glob
import re
from tqdm import tqdm


class Phenotypes(data.Dataset):
    def __init__(self, path, model=model, device=device) -> None:
        super().__init__()
        self.path = path
        self.model = model
        self.device = device

        with open(path + "/std_genes_avg.pkl", "rb") as f:
            self.bestgene = list(pkl.load(f).index[:900])
            
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.embeddings_dict = self.embed_all_images()
        self.genotypes = self.concat_tsv()

    def __len__(self):
        return len(self.embeddings_dict)

    def __getitem__(self, index):
        return torch.tensor(self.genotypes.iloc[index].values), self.embeddings_dict[index]

    def tsv_processing(self, tissue_name: str, df: pd.DataFrame) -> pd.DataFrame:
        # mask = list(df.filter(regex="ambiguous"))
        # filtered_df = df[df.columns.drop(mask)]
        filtered_df = df[self.bestgene]
        filtered_df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
        filtered_df["id"] = filtered_df["id"].apply(lambda x: tissue_name + '_' + x)
        filtered_df.set_index("id", inplace=True)

        return filtered_df
    
    def concat_tsv(self):
        df = pd.DataFrame()
        for path_tsv in tqdm(glob(self.path + "/*.tsv", recursive=True)):
            m = re.search("data/(.*)/(.*).tsv", path_tsv)
            if m:
                tissue_name = m.group(2)
                with open(path_tsv, "rb") as f:
                    df_tsv = pd.read_csv(f, sep="\t")
                df_tsv = self.tsv_processing(tissue_name, df_tsv)
                df = pd.concat([df, df_tsv])
            else:
                raise ValueError("Path not found")
        return df

    def image_embedding(self, path):
        cell = Image.open(path)
        input_tensor = self.preprocess(cell)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch of 1 sample
        input_batch = input_batch.to(self.device)
        with torch.no_grad():
            output = self.model(input_batch)
        selection_tensor = torch.tensor(
            [[552, 1382, 1171, 699, 663, 1502, 588, 436, 1222, 617]]
        )
        return output[selection_tensor]

    def embed_all_images(self):
        embeddings_dict = {}
        for sub_path in tqdm(glob(self.path + "/*/", recursive=True)):
            for path_image in tqdm(glob(sub_path + "/*.jpg", recursive=True)):
                m = re.search("data/(.*)/(.*).jpg", path_image)
                if m:
                    embeddings_dict[m.group(2)] = self.image_embedding(path_image)
                else:
                    raise ValueError("Path not found")
        return embeddings_dict