#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Preprocessing before styleGAN2 training

import os
import sys
sys.path.append("../")
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import re
import json

path_to_image = "/projects/minos/jeremie/data/images"
path_to_complete = "/projects/minos/jeremie/data/complete_concatenate_df.csv"

def create_dict_label(path: str, df_complete: pd.DataFrame) -> dict:
    os.chdir(path)
    list_image = glob("*/*.jpg")

    pattern = ".*/(.*).jpg"
    cnt = 0
    for path in list_image:
        match = re.match(pattern, path)
        idx = match.group(1)
        print(idx)
        label = (df_complete.loc[idx]["tumor"] == "tumor") * 1
        list_image[cnt] = [path, label]
        cnt += 1
        
    for elt in list_image:
        assert type(elt) == list, "Error: list_image is not a list of list"

    label_dict = {}
    label_dict["labels"] = list_image

def export_json(dict_label: dict, path: str) -> None:
    os.chdir(path)
    with open("labels.json", "w") as f:
        json.dump(dict_label, f)


if __name__ == "__main__":
    df_complete = pd.read_csv(path_to_complete, index_col=0)
    dict_label = create_dict_label(path_to_image, df_complete)
    export_json(dict_label, path_to_image)