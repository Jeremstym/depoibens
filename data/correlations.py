#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Analysis of correlation

import os
import sys
from io import BytesIO

# setting path
sys.path.append("../")
import pickle as pkl
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import R2Score, PearsonCorrCoef
from models.inception_STnet import Regression_STnet
from data.dataset_stnet import create_dataloader, Dataset_STnet
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
path_to_save = "/projects/minos/jeremie/data/Preprocessing_results"


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

    dataset = Dataset_STnet(tsv, dino_dict, nb_genes=INPUT_SIZE, embd_size=OUTPUT_SIZE)
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
        res = (0, 255, 0, 95)
    elif score > 0.60:
        res = (0, 0, 255, 45)
    else:
        res = (255, 0, 0, 35)
    return res


def create_red_stripes_black_background() -> None:
    path_red_hatch = (
        "/import/bc_users/biocomp/stym/depoibens/data/patterns/red_stripes.jpg"
    )

    img = Image.open(path_red_hatch).convert("RGBA")

    datas = img.getdata()

    newData = []
    for item in datas:
        if (
            item[0] in list(range(190, 256))
            and item[1] in list(range(190, 256))
            and item[2] in list(range(190, 256))
        ):
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(
        "/import/bc_users/biocomp/stym/depoibens/data/patterns/blackred_stripes.png",
        "PNG",
    )


def color_spot(path: str, df_score: pd.DataFrame) -> None:
    os.chdir(path)
    path_red_hatch = (
        "/import/bc_users/biocomp/stym/depoibens/data/patterns/blackred_stripes.png"
    )
    path_arial = "/import/bc_users/biocomp/stym/depoibens/data/arial.ttf"
    file_pattern = "*_complete.pkl"
    with open(path_red_hatch, "rb") as file:
        bytes_red_hatch = BytesIO(file.read())
        hatch_image = Image.open(bytes_red_hatch).convert("RGBA")
        hatch_image.putalpha(64)
    with open(path_arial, "rb") as file:
        bytes_font = BytesIO(file.read())
    font = ImageFont.truetype(bytes_font, 100)

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
                is_tumor = df_complete.loc[idx]["tumor"]
                score = df_score.loc[crop_name]["pearson"]
                color = color_score(score)
                size = int((gaps[0] + gaps[1]) / 2)
                colored_image = Image.new("RGBA", (size, size), color)
                # tissue_img = Image.alpha_composite(tissue_img, colored_image)
                tissue_img.paste(colored_image, (posX, posY), colored_image)
                if is_tumor == "tumor":
                    hatch_image = hatch_image.resize((size, size))
                    tissue_img.paste(hatch_image, (posX, posY), hatch_image)
                # draw = ImageDraw.Draw(tissue_img)
                # draw.text((posX, posY), str(round(score, 2)), font=font, fill=(0, 0, 0, 255))

        colored_box_3 = Image.new("RGBA", (150, 150), (0, 255, 0, 95))
        colored_box_2 = Image.new("RGBA", (150, 150), (0, 0, 255, 45))
        colored_box_1 = Image.new("RGBA", (150, 150), (255, 0, 0, 35))
        hatched_box = hatch_image.resize((150, 150))
        hatched_box.putalpha(255)
        tissue_img.paste(colored_box_3, (6000, 8500), colored_box_3)
        tissue_img.paste(colored_box_2, (6000, 8300), colored_box_2)
        tissue_img.paste(colored_box_1, (6000, 8100), colored_box_1)
        tissue_img.paste(hatched_box, (6000, 7900), hatched_box)

        draw = ImageDraw.Draw(tissue_img)
        # font = ImageFont.truetype("data/arial.ttf", 100)
        # default_font = ImageFont.load_default()

        text3 = "Pearson > 0.80"
        text2 = "Pearson > 0.60"
        text1 = "Pearson < 0.60"
        text4 = "Tumor"

        draw.text((6200, 8500), text3, font=font, fill=(0, 0, 0, 255), align="center")
        draw.text((6200, 8300), text2, font=font, fill=(0, 0, 0, 255), align="center")
        draw.text((6200, 8100), text1, font=font, fill=(0, 0, 0, 255), align="center")
        draw.text((6200, 7900), text4, font=font, fill=(0, 0, 0, 255), align="center")

        tissue_img.save(tissue_name + "_score_hatched.png", "PNG")


def test_color_spot_1_spot(path: str, df_score: pd.DataFrame) -> None:
    os.chdir(path)
    path_red_hatch = (
        "/import/bc_users/biocomp/stym/depoibens/data/patterns/blackred_stripes2.png"
    )
    file_pattern = "*_complete.pkl"
    with open(path_red_hatch, "rb") as file:
        bytes_red_hatch = BytesIO(file.read())
        hatch_image = Image.open(bytes_red_hatch).convert("RGBA")
        hatch_image.putalpha(128)

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
                is_tumor = df_complete.loc[idx]["tumor"]
                score = df_score.loc[crop_name]["pearson"]
                color = color_score(score)
                size = int((gaps[0] + gaps[1]) / 2)
                colored_image = Image.new("RGBA", (size, size), color)
                tissue_img.paste(colored_image, (posX, posY), colored_image)
                if is_tumor == "tumor":
                    hatch_image = hatch_image.resize((size, size))
                    tissue_img.paste(hatch_image, (posX, posY), hatch_image)
                    tumor_detected = True
                if tumor_detected:
                    break
            tissue_img.save(tissue_name + "_score_hatched.png", "PNG")
            if tumor_detected:
                break


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

if __name__ == '__main__':
    create_red_stripes_black_background()

# if __name__ == "__main__":
#     df_corr = pd.read_csv(path_to_csv, index_col=0)
#     color_spot(path_to_data, df_corr)
#     print("Done")


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
