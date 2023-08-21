#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Preprocessing before styleGAN2 training

import os
import sys

sys.path.append("../")
from glob import glob
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd
import re
import json

### ---------- Pathes and constants -------------

path_to_image = "/projects/minos/jeremie/data/images"
path_to_complete = "/projects/minos/jeremie/data/complete_concatenate_df.csv"
path_to_tsv = "/projects/minos/jeremie/data/tsv_concatened_allgenes.pkl"

### ---------- Functions -------------


def create_dict_label(path: str, df_complete: pd.DataFrame) -> dict:
    os.chdir(path)
    list_image = glob("*/*.jpg")

    pattern = ".*/(.*).jpg"
    cnt = 0
    for path in list_image:
        match = re.match(pattern, path)
        idx = match.group(1)
        label = (df_complete.loc[idx]["tumor"] == "tumor") * 1
        list_image[cnt] = [path, label]
        cnt += 1

    label_dict = {}
    label_dict["labels"] = list_image

    return label_dict


def create_dict_tsv_label(
    path: str, tsv_df: pd.DataFrame, nb_genes=900
) -> dict:
    os.chdir(path)
    list_image = glob("*/*.jpg")

    pattern = ".*/(.*).jpg"
    cnt = 0
    for path in list_image:
        match = re.match(pattern, path)
        idx = match.group(1)
        label_tsv = tsv_df.loc[idx].values[:nb_genes]
        list_image[cnt] = [path, label_tsv]
        cnt += 1

    label_dict = {}
    label_dict["labels"] = list_image

    return label_dict


def export_json(dict_label: dict, path: str) -> None:
    os.chdir(path)
    with open("dataset.json", "w") as f:
        json.dump(dict_label, f)

def export_pickle(dict_label: dict, path: str) -> None:
    os.chdir(path)
    with open("dataset_genes.pkl", "wb") as f:
        pickle.dump(dict_label, f)

### ---------- Programmes -------------

# if __name__ == "__main__":
#     df_complete = pd.read_csv(path_to_complete, index_col=0)
#     tsv_df = pd.read_pickle(path_to_tsv)
#     dict_label = create_dict_tsv_label(path_to_image, tsv_df)
#     export_pickle(dict_label, path_to_image)

if __name__ == "__main__":
    df_complete = pd.read_csv(path_to_complete, index_col=0)
    dict_label = create_dict_label(path_to_image, df_complete)
    export_json(dict_label, path_to_image)
