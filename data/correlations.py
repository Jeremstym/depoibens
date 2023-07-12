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
from PIL import Image
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import R2Score, PearsonCorrCoef
from models.inception_STnet import Regression_STnet
from data.dataset_stnet import Phenotypes, create_dataloader
from data.dataexploration import complete_processing

INPUT_SIZE = 900
OUTPUT_SIZE = 768
HIDDEN_SIZE = 1536
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_PATIENT = "BC23209"
path_to_dino_dict = "/projects/minos/jeremie/data/dino_features.pkl"
path_to_tsv = "/projects/minos/jeremie/data/tsv_concatened_allgenes.pkl"
path_to_csv = "/projects/minos/jeremie/data/outputs/test_correlations.csv"
path_to_score = "/projects/minos/jeremie/data/outputs/saptial_score.csv"
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
        test_patient=TEST_PATIENT,
        num_workers=4,
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


def color_score(score: float) -> int:
    # res = 100 * round(score, 1)
    if score > 0.80:
        res = 95
    elif score > 0.60:
        res = 50
    else:
        res = 30
    return int(res)


def color_spot(path: str, df_score: pd.DataFrame) -> None:
    os.chdir(path)
    file_pattern = "*_complete.pkl"
    for df in glob(file_pattern):
        df_complete = pd.read_pickle(df)
        df_complete = complete_processing(df_complete)
        tissue_img_loc = re.sub("_complete.pkl", ".jpg", df)
        tissue_img = Image.open(tissue_img_loc)
        tissue_img = tissue_img.convert("RGBA")
        tissue_name = re.sub("_complete.pkl", "", df)
        with tqdm(df_complete.index, total=len(df_complete.index), unit="spot") as pbar:
            pbar.set_description(f"Coloring spots of {tissue_name}")
            for idx in pbar:
                crop_name = tissue_name + "_" + idx
                # Note that the axis are purposely inversed below
                coord = df_complete.loc[idx][["Y", "X"]].values
                coord = list(map(round, list(coord)))
                gaps = df_complete.loc[idx][["gapY", "gapX"]].values
                gaps = list(map(round, list(gaps)))
                posY = coord[0] + int(gaps[0] / 2)
                posX = coord[1] - int(gaps[1] / 2)

                score = df_score.loc[crop_name]["pearson"]
                color = color_score(score)
                green = (0, 255, 0, color)
                size = int((gaps[0] + gaps[1]) / 2)
                green_image = Image.new("RGBA", (size, size), green)
                # tissue_img = Image.alpha_composite(tissue_img, green_image)
                tissue_img.paste(green_image, (posX, posY), green_image)
        green_box_3 = Image.new("RGBA", (150,150), (0, 255, 0, 95))
        green_box_2 = Image.new("RGBA", (150,150), (0, 255, 0, 50))
        green_box_1 = Image.new("RGBA", (150,150), (0, 255, 0, 30))
        tissue_img.paste(green_box_3, (6000, 8000), green_box_3)
        tissue_img.paste(green_box_2, (6000, 7800), green_box_2)
        tissue_img.paste(green_box_1, (6000, 7600), green_box_1)
        tissue_img.save(tissue_name + "_score.png", "PNG")


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

# if __name__ == "__main__":
#     df_complete = concatenate_dfcomplete(path_to_data)
#     df_corr = pd.read_csv(path_to_csv, index_col=0)
#     df_corr = df_corr.join(df_complete, how="inner")
#     df_corr.to_csv("/projects/minos/jeremie/data/outputs/saptial_score.csv")

if __name__ == "__main__":
    df_corr = pd.read_csv(path_to_csv, index_col=0)
    color_spot(path_to_data, df_corr)
    print("Done")


### ------------------- Brouillon ----------------------

# img = Image.new("RGB", (100, 100), color=(73, 0, 0))
# img.show()

# local_path = r"E:\ST-Net\data\hist2tscript\BRCA\BC23209"

# img = Image.open(local_path + r"\BC23209_C2.jpg")
# newsize = (300, 300)
# img = img.resize(newsize)
# Shows the image in image viewer
# img.show()
# img_array = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)

# img_array.shape

# img = img.convert("RGBA")
# green = (0, 255, 0, 85)
# x = 100  # x-coordinate of the top-left corner of the square
# y = 200  # y-coordinate of the top-left corner of the square
# size = 50  # size of the square in pixels

# green_image = Image.new("RGBA", (size, size), green)
# img.paste(green_image, (x, y), green_image)


# img.show()

# # Color the square with the transparent green color
# pixels = img.load()
# for i in range(x, x + size):
#     for j in range(y, y + size):
#         pixels[i, j] = green
