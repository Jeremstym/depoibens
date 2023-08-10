#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Preprocessing before styleGAN2 training

import os
import sys
sys.path.append("../")
from glob import glob

import numpy as np
import pandas as pd
import re
import json

path_to_image = "/projects/minos/jeremie/data/images"
path_to_complete = ""

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
        
    for elt in list_image:
        assert type(elt) == list, "Error: list_image is not a list of list"
        
    label_dict = {}
    label_dict["labels"] = list_image
