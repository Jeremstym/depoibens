#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Analysis of correlation

import os
import sys

# setting path
sys.path.append("../")
import pickle as pkl
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import R2Score, PearsonCorrCoef
from models.inception_STnet import Regression_STnet
from dataset_stnet import Phenotypes, create_dataloader

INPUT_SIZE = 900
OUTPUT_SIZE = 768
HIDDEN_SIZE = 1536
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_PATIENT = "BC23209"
path_to_dino_dict = "/projects/minos/jeremie/data/dino_features.pkl"
path_to_tsv = "/projects/minos/jeremie/data/tsv_concatened_allgenes.pkl"
path_to_csv = "/projects/minos/jeremie/data/outputs/test_correlations.csv"
path_to_data = "/projects/minos/jeremie/data/" + TEST_PATIENT


def load_model(path_to_model: str) -> nn.Module:
    model = Regression_STnet(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE
    )

    model.load_state_dict(
        torch.load(path_to_model, map_location=torch.device(device))["model_state_dict"]
    )
    model.eval()
    model.to(device)
    return model


def data_loader(path_to_tsv: str, path_to_dino_dict: str) -> DataLoader:
    with open(path_to_dino_dict, "rb") as f:
        dino_dict = pkl.load(f)

    with open(path_to_tsv, "rb") as f:
        tsv = pkl.load(f)

    dataset = Phenotypes(tsv, dino_dict, nb_genes=INPUT_SIZE, embd_size=OUTPUT_SIZE)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    return dataloader, dataset


def create_df_corr(
    model: nn.Module, path_to_dino_dict=path_to_dino_dict, path_to_tsv=path_to_tsv
) -> pd.DataFrame:
    model.eval()
    model.to(device)

    criterion = nn.MSELoss()
    r2 = R2Score(multioutput="variance_weighted").to(device)
    pearson = PearsonCorrCoef().to(device)

    df_corr = pd.DataFrame(columns=["loss", "r2", "pearson"])
    # dataloader, dataset = data_loader(path_to_tsv, path_to_dino_dict)
    _, _, testloader = create_dataloader(
        path_to_tsv,
        path_to_dino_dict,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        train_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        num_workers=4,
        test_patient="BC23209",
    )

    with tqdm(enumerate(testloader), total=len(testloader)) as pbar:
        for i, data in pbar:
            genotype, embd = data
            genotype, embd = genotype.float(), embd.squeeze(1)
            genotype, embd = genotype.to(device), embd.to(device)

            outputs = model(genotype)
            # print(outputs.shape, embd.shape)
            loss = criterion(outputs, embd)
            r2_score = r2(outputs.T, embd.T)
            pearson_score = pearson(outputs.squeeze(0), embd.squeeze(0))

            idx_name = testloader.dataset.get_index_name(i)
            df_corr.loc[idx_name] = [loss.item(), r2_score.item(), pearson_score.item()]

    return df_corr


def concatenate_dfcomplete(path: str) -> pd.DataFrame:
    df = pd.DataFrame()
    os.chdir(path)
    file_pattern = "./*_complete.pkl"
    for file_name in glob(file_pattern):
        inter_df = pd.read_pickle(file_name)
        file_name = file_name[2:12]
        inter_df.reset_index(inplace=True)
        inter_df["id"] = inter_df["id"].apply(lambda x: file_name + "_" + x)
        inter_df.set_index("id", inplace=True)
        df = pd.concat([df, inter_df])
        droppings = ["title", "lab", "tumor"]
    return df.drop(droppings, axis=1)


### ------------------- MAIN ----------------------

# if __name__ == "__main__":
#     path_to_model = "/projects/minos/jeremie/data/outputs/best_model_dino.pth"
#     model = load_model(path_to_model)
#     df_corr = create_df_corr(model)
#     df_corr.to_csv("/projects/minos/jeremie/data/outputs/test_correlations.csv")
#     print(df_corr)

# if __name__ == "__main__":
#     df = concatenate_dfcomplete(path_to_data)
#     print(df)

if __name__ == "__main__":
    df_complete = concatenate_dfcomplete(path_to_data)
    df_corr = pd.read_csv(path_to_csv, index_col=0)
    df_corr = df_corr.join(df_complete, how="inner")
    df_corr.to_csv("/projects/minos/jeremie/data/outputs/saptial_score.csv")
