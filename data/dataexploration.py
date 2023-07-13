#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Data exploration

import os
import re
from tqdm import tqdm
import multiprocessing as mp

import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mping

import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000

def colname_fixer(parent_path: str):
    """Fixe column name issues with previous datasets
    CAREFUL, IT CAN DAMAGE DATA. USE ONLY IF NECESSARY

    Args:
        parent_path (str): path to the datasets

    Raises:
        ValueError: If datasets already processed
    """

    files_list = os.listdir(parent_path)
    files_list.remove(".DS_Store")
    files_list.remove("._.DS_Store")

    for file in files_list:
        print(file)
        essai = os.listdir(parent_path + "\\" + file)
        essaistr = "\n".join(essai)
        df_list = re.findall(".*Coords.pkl", essaistr)

        for df in df_list:
            print(df)
            child_path = "\\" + file + "\\" + df
            print(parent_path + child_path)

            with open(parent_path + child_path, "rb") as f:
                mat = pkl.load(f)

            if "Unnamed: 4" not in mat.columns:
                raise ValueError("Data already processed")

            mat = mat.rename(
                columns={
                    "xcoord": "title",
                    "ycoord": "xcoord",
                    "lab": "ycoord",
                    "tumor": "lab",
                    "Unnamed: 4": "tumor",
                }
            )

            with open(parent_path + child_path, "wb") as f:
                pkl.dump(mat, f)


def zero_counter(df: pd.DataFrame) -> pd.Series:
    """pre-processing: count all zero values per column

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.Series:
    """
    return df.drop("Unnamed: 0", axis=1).isin([0]).sum(axis=0).sort_values()


def filtering_ambiguous(df: pd.DataFrame, with_index=False) -> pd.DataFrame:
    """Remove the "ambiguous" genes from the datasets

    Args:
        df (pd.DataFrame): datasets with "ambiguous" columns

    Returns:
        pd.DataFrame: New clean dataset
    """
    mask = list(df.filter(regex="ambiguous"))
    filtered_df = df[df.columns.drop(mask)]
    if with_index:
        return filtered_df.rename(columns={"Unnamed: 0":"id"}).set_index("id")
    else:
        return filtered_df.drop("Unnamed: 0", axis=1)


def common_genes(path: str):
    """Count number of genes for all patients

    Args:
        path (str): path to the datasets

    Returns:
        collections.Counter: OrderedDict with number of appearance
        for each gene
    """

    gene_list = []
    counter = 0

    files_list = os.listdir(path)
    files_list.remove(".DS_Store")
    files_list.remove("._.DS_Store")
    files_list.remove("std_genes.pkl")
    files_list.remove("std_genes_avg.pkl")

    for file in tqdm(files_list):
        essai = os.listdir(path + "\\" + file)
        essaistr = "\n".join(essai)
        df_list = re.findall(".*[0-9].tsv", essaistr)

        for df in df_list:
            child_path = "\\" + file + "\\" + df

            with open(path + child_path, "rb") as f:
                df_brut = pd.read_csv(f, sep="\t")
            counter += 1
            df_filter = filtering_ambiguous(df_brut)
            gene_list += list(df_filter.columns.values)

    gene_counter = Counter(gene_list)

    print(counter)
    return gene_counter


def most_common_gene(dict_count: Counter) -> list:
    """Gives the genes that appears in all the cells of the dataset.
    Meaning: 68 times

    Args:
        dict_count (Counter): OrderedDict that count each gene appearance

    Returns:
        list: List of the most frequent genes
    """
    maxi = max(dict_count.values())
    filtered_dict = dict(filter(lambda x: x[1] == maxi, dict_count.items()))
    return list(filtered_dict.keys())


def get_gene_std(path: str, gene_list: list, method="avg") -> pd.Series:
    """Gives the standard deviation for the gene distribution for all tissues
    (or for each tissue then average? QUESTION)

    Args:
        path (str): path to all datasets
        gene_list (list): list of genes present in all tissues

    Returns:
        pd.Series: a ranked serie of all genes with
        the biggest standard deviation
    """

    final_df = pd.DataFrame()
    final_avg = pd.DataFrame()

    files_list = os.listdir(path)
    files_list.remove(".DS_Store")
    files_list.remove("._.DS_Store")
    files_list.remove("std_genes.pkl")
    files_list.remove("std_genes_avg.pkl")

    for file in tqdm(files_list):
        essai = os.listdir(path + "\\" + file)
        essaistr = "\n".join(essai)
        list_df = re.findall(".*[0-9].tsv", essaistr)

        for df in list_df:
            child_path = "\\" + file + "\\" + df

            with open(path + child_path, "rb") as f:
                df_brut = pd.read_csv(f, sep="\t")
            df_filtered = filtering_ambiguous(df_brut)[gene_list]

            if method == "avg":
                std_serie = df_filtered.std(axis=0)
                final_avg = pd.concat([final_avg, std_serie], axis=1, ignore_index=True)
            elif method == "all":
                final_df = pd.concat([final_df, df_filtered], ignore_index=True)

    if method == "avg":
        gene_std = final_avg.mean(axis=1).sort_values(ascending=False)
    elif method == "all":
        gene_std = final_df.std(axis=0).sort_values(ascending=False)

    return gene_std


def summing_genes(path: str, gene_list: list) -> pd.Series:
    final_df = pd.DataFrame()

    files_list = os.listdir(path)
    files_list.remove(".DS_Store")
    files_list.remove("._.DS_Store")
    files_list.remove("std_genes.pkl")
    files_list.remove("std_genes_avg.pkl")

    for file in tqdm(files_list):
        essai = os.listdir(path + "\\" + file)
        essaistr = "\n".join(essai)
        list_df = re.findall(".*[0-9].tsv", essaistr)

        for df in list_df:
            child_path = "\\" + file + "\\" + df

            with open(path + child_path, "rb") as f:
                df_brut = pd.read_csv(f, sep="\t")
            df_filtered = filtering_ambiguous(df_brut)[gene_list]

            sum_serie = df_filtered.sum(axis=0)
            final_df = pd.concat([final_df, sum_serie], axis=1, ignore_index=True)

    gene_sum = final_df.mean(axis=1).sort_values(ascending=False)
    return gene_sum


def make_index(df: pd.DataFrame) -> pd.DataFrame:
    """Create index from the position of the spot (line x column)
    Round the coordinates of each spot and create
        an index name from it

    Args:
        df (pd.DataFrame): Dataframe (coordonates with decimals),
        tumor indicator, lab

    Returns:
        pd.DataFrame: Return same dataframe with additional column: "Id"
    """
    df["id"] = (
        df["xcoord"].apply(lambda x: str(round(x)))
        + "x"
        + df["ycoord"].apply(lambda x: str(round(x)))
    )

    return df


def inner_genes(path):
    """Join dataframe with new index (line x column) for each spot, with
    a dataframe containing true coordinates for each pixels.

    Args:
        path (_type_): Path to all datasets
    """

    files_list = os.listdir(path)
    files_list.remove(".DS_Store")
    files_list.remove("._.DS_Store")
    files_list.remove("std_genes.pkl")
    files_list.remove("std_genes_avg.pkl")

    for file in tqdm(files_list):
        essai = os.listdir(path + "\\" + file)
        essaistr = "\n".join(essai)
        df_list = re.findall(".*Coords.pkl", essaistr)

        for df in df_list:
            child_path = "\\" + file + "\\" + df

            with open(path + child_path, "rb") as f:
                df_coords = pkl.load(f)

            df_coords = make_index(df_coords)

            df2 = re.sub("_Coords.pkl", ".spots.txt", df)
            with open(path + "\\" + file + "\\" + df2, "rb") as f:
                df_spots = pd.read_csv(f, sep=",")

            df_spots.set_index("Unnamed: 0", inplace=True)

            df_join = df_coords.join(df_spots, on="id").set_index("id")
            df_join.drop(columns=["xcoord", "ycoord"], inplace=True)
            df3 = re.sub(".spots.txt", "_complete.pkl", df2)
            with open(path + "\\" + file + "\\" + df3, "wb") as f:
                pkl.dump(df_join, f)


def counting_frame(path: str, gene_list: list) -> pd.DataFrame:
    """Create a dataframe with the number of counts for each gene

    Args:
        path (str): path to the folder containing all datasets
        gene_list (list): list of genes to count (best to use std_genes.pkl)

    Returns:
        pd.DataFrame: dataframe with the number of counts for each gene
    """
    counting_df = pd.DataFrame()

    files_list = os.listdir(path)
    files_list.remove(".DS_Store")
    files_list.remove("._.DS_Store")
    files_list.remove("std_genes.pkl")
    files_list.remove("std_genes_avg.pkl")

    for file in tqdm(files_list):
        essai = os.listdir(path + "\\" + file)
        essaistr = "\n".join(essai)
        list_df = re.findall(".*[0-9].tsv", essaistr)

        for df in list_df:
            child_path = "\\" + file + "\\" + df

            with open(path + child_path, "rb") as f:
                df_brut = pd.read_csv(f, sep="\t")
            df_filtered = filtering_ambiguous(df_brut)[gene_list]

            sum_serie = df_filtered.sum()
            sum_df = pd.DataFrame(sum_serie).T
            sum_name = df.replace(".tsv", "")
            sum_df.index = [sum_name]
            counting_df = pd.concat([counting_df, sum_df])

    return counting_df


def complete_processing(df: pd.DataFrame) -> pd.DataFrame:
    df["idX"] = df.index.map(lambda x: int(x.split("x")[0]))
    df["idY"] = df.index.map(lambda x: int(x.split("x")[1]))
    df.sort_values(["idX", "idY"], inplace=True)
    df["gapY"] = df["Y"].shift(-1, fill_value=0) - df["Y"]
    df["gapX"] = df["X"].shift(-1, fill_value=0) - df["X"]
    df["gapX"] = df["idX"].apply(lambda x: df[df["idX"] == x]["gapX"].max())
    df["gapX"] = df["gapX"].apply(lambda x: 250 if x < 250 else x)
    df["gapY"] = df["gapY"].apply(lambda x: 250 if x < 250 else x)

    return df


def get_cropped_image(path: str, format=".jpg") -> None:
    """Stock cropped image for each tissue, in each file (by spot)

    Args:
        path (str): path to all datasets
        format (str, optional): Format of the image (.png or .tif). Defaults to ".png".
    """
    files_list = os.listdir(path)
    files_list.remove(".DS_Store")
    files_list.remove("._.DS_Store")
    files_list.remove("std_genes.pkl")
    files_list.remove("std_genes_avg.pkl")

    for file in tqdm(files_list):
        essai = os.listdir(path + "/" + file)
        essaistr = "\n".join(essai)
        df_list = re.findall(".*complete.pkl", essaistr)

        for df in df_list:
            child_path = "/" + file + "/" + df
            with open(path + child_path, "rb") as f:
                df_complete = pkl.load(f)

            df_complete = complete_processing(df_complete)
            tissue_img_loc = re.sub("_complete.pkl", ".tif", df)
            tissue_img = cv2.imread(
                path + "/" + file + "/" + tissue_img_loc, cv2.COLOR_BGR2RGB
            )
            tissue_name = re.sub("_complete.pkl", "", df)
            os.mkdir(path + "/" + file + "/" + tissue_name)

            for idx in df_complete.index:
                crop_name = tissue_name + "_" + idx
                # Note that the axis are purposely inversed below
                coord = df_complete.loc[idx][["Y", "X"]].values
                coord = list(map(round, list(coord)))
                gaps = df_complete.loc[idx][["gapY", "gapX"]].values
                gaps = list(map(round, list(gaps)))
                img_crop = tissue_img[
                    coord[0] - int(gaps[0] / 2) : coord[0] + int(gaps[0] / 2),
                    coord[1] - int(gaps[1] / 2) : coord[1] + int(gaps[1] / 2),
                ]
                cv2.imwrite(
                    path + "/" + file + "/" + tissue_name + "/" + crop_name + format,
                    img_crop,
                )


### ------------- Programmes ----------------------

# if __name__ == "__main__":
#     path = r"E:\ST-Net\data\hist2tscript\BRCA"
#     try_count = common_genes(path)
#     gene_used = most_common_gene(try_count)
#     gene_std = get_gene_std(path, gene_used)


# ### Already done below:
# # colname_fixer(path)
# # inner_genes(path)

# if __name__ == "__main__":
#     path = r"E:\ST-Net\data\hist2tscript\BRCA"
#     files_list = os.listdir(path)
#     files_list.remove(".DS_Store")
#     files_list.remove("._.DS_Store")
#     files_list.remove("std_genes.pkl")
#     files_list.remove("std_genes_avg.pkl")

#     # counting = 0
#     listing = []
#     listing_by_patient = []
#     for file in tqdm(files_list):
#         essai = os.listdir(path + "\\" + file)
#         essaistr = "\n".join(essai)
#         df_list = re.findall(".*spots.txt", essaistr)

#         for df in df_list:
#             child_path = "\\" + file + "\\" + df

#             with open(path + child_path, "rb") as f:
#                 df_coords = pd.read_csv(f, sep=",")

#             listing.append(len(df_coords))
#         listing_by_patient.append(np.sum(listing))
#         listing = []

# if __name__ == "__main__":
#     path = r"E:\ST-Net\data\hist2tscript\BRCA"
#     try_count = common_genes(path)
#     gene_used = most_common_gene(try_count)
#     gene_sum = summing_genes(path, gene_used)

# if __name__ == "__main__":
#     path = r"E:\ST-Net\data\hist2tscript\BRCA"
#     try_count = common_genes(path)
#     gene_used = most_common_gene(try_count)
#     df_sum = counting_frame(path, gene_used)

if __name__ == "__main__":
    path = "/import/pr_minos/jeremie/data"
    get_cropped_image(path)



### ------------ Brouillon -----------------------

# with open(r"E:\ST-Net\data\hist2tscript\BRCA\BC23270\BC23270_E2.tsv", "rb") as f:
#     df_gene = pd.read_csv(f, sep="\t")
# # with open(r"E:\ST-Net\data\hist2tscript\BRCA\BC23209\BC23209_C2.tsv", "rb") as f:
# #     df_gene2 = pd.read_csv(f, sep="\t")

# df_gene = filtering_ambiguous(df_gene, with_index=True)
# df_gene[gene_used].iloc[0].value_counts

# with open(r"E:\ST-Net\data\hist2tscript\BRCA\std_genes_avg.pkl", "rb") as f:
#     df_std = pkl.load(f)

# bestgene = list(df_std.index[:1000])

# df_gene[bestgene]

# df_sum.to_csv(r"C:\Jérémie\Stage\IBENS\depo\data\df_sum.csv", index_label="id")

# with open(path + '\\' + 'std_genes.pkl', "rb") as f:
#     hello = pkl.load(f)

# with open(path + '\\' + 'std_genes_avg.pkl', "rb") as f:
#     bye = pkl.load(f)

# selected_cols = list(bye.head(10).index)
# df_sum[selected_cols].head(10).to_excel(r"C:\Jérémie\Stage\IBENS\depo\data\df_sum.xlsx")

# df_gene.sum().sort_values(ascending=False).head(10).plot.pie(autopct='%1.1f%%')
# df_gene2 = filtering_ambiguous(df_gene2)
# df_gene2.sum().sort_values(ascending=False).head(10).plot.pie(autopct='%1.1f%%')

# gene_sum.head(10)

# df_gene.set_index("Unnamed: 0").rename(columns={"Unnamed: 0":"index"})

# len(set(list(bye.index[:1000])).intersection(list(hello.index[:1000])))

# try_list = list(try_count.keys())

# "ENSG00000270951" in gene_sum.index # false
# "ENSG00000276722" in try_list # False


# with open(r"E:\ST-Net\data\hist2tscript\BRCA\BC23209\BC23209_C1_Coords.pkl", "rb") as f:
#     hello = pkl.load(f)

# hello

# with open(r"E:\ST-Net\data\hist2tscript\BRCA\BC23209\BC23209_C1.spots.txt", "r") as f:
#     df_try = pd.read_csv(f, sep=',')

# df_try

# with open(r"E:\ST-Net\data\hist2tscript\BRCA\BC23209\BC23209_C1.tsv", "r") as f:
#     df_tsv = pd.read_csv(f, sep='\t')

# df_tsv

# with open(
#     r"E:\ST-Net\data\hist2tscript\BRCA\BC23270\BC23270_D2_complete.pkl", "rb"
# ) as f:
#     df_try = pkl.load(f)

# df_try["idX"] = df_try.index.map(lambda x: int(x.split("x")[0]))
# df_try["idY"] = df_try.index.map(lambda x: int(x.split("x")[1]))
# df_try.sort_values(["idX", "idY"], inplace=True)
# df_try["gapY"] = df_try["Y"].shift(-1, fill_value=0) - df_try["Y"]
# df_try["gapX"] = df_try["X"].shift(-1, fill_value=0) - df_try["X"]
# df_try["gapX"] = df_try["idX"].apply(lambda x: df_try[df_try["idX"] == x]["gapX"].max())
# df_try["gapX"] = df_try["gapX"].apply(lambda x: 300 if x < 0 else x)
# df_try["gapY"] = df_try["gapY"].apply(lambda x: 300 if x < 0 else x)
# print(df_try.to_string())

# coord = df_try.loc["7x22"][["gapX", "gapY"]].values
# coord = list(map(round, list(coord)))
# coord
# c1tif_path = path + '\\' + 'BC23270\BC23270_D2.tif'

# finder = Image.open(c1tif_path)
# finder.size

# plt.imshow(finder)
# plt.show()

# img = cv2.imread(c1tif_path, cv2.COLOR_BGR2RGB)
# cropped = img[coord[0]:coord[0]+300, coord[1]:coord[1]+300]
# # imS = cv2.resize(img, (1000, 1000))
# cropped = img[5207:5207+300, 1161:1161+300]
# cv2.imshow('original', cropped)

# files_list = os.listdir(path)
# files_list.remove(".DS_Store")
# files_list.remove("._.DS_Store")
# files_list.remove("std_genes.pkl")
# files_list.remove("std_genes_avg.pkl")

# for file in tqdm(files_list):
#     essai = os.listdir(path + '\\' + file)
#     essaistr = '\n'.join(essai)
#     df_list = re.findall('.*complete.pkl', essaistr)
#     break

# df_list

# re.findall("(.*)_complete.pkl", "\n".join(df_list))

# re.sub("_complete.pkl", "", df_list[0])


# with open(r"E:\ST-Net\data\hist2tscript\BRCA\BC23209\BC23209_C1.spots.txt", "rb") as f:
#     dimC1 = pd.read_csv(f, sep=',')

# dimC1.sort_values('Y')

# with open(r"E:\ST-Net\data\hist2tscript\BRCA\BC23209\BC23209_C1.tsv", "rb") as f:
#     geneC1_ambiguous = pd.read_csv(f, sep='\t')

# genC1 = filtering_ambiguous(geneC1_ambiguous)
# genC1.std(axis=0).sort_values(ascending=False)

# d = OrderedDict(sorted(try_count.items(), key=itemgetter(1), reverse=True))

# with open(r"E:\ST-Net\data\hist2tscript\BRCA\BC23209\BC23209_C2.tsv", "rb") as f:
#     geneC2_ambiguous = pd.read_csv(f, sep='\t')

# genC2 = filtering_ambiguous(geneC2_ambiguous)
# genC2.std(axis=0).sort_values(ascending=False).index.sort_values()

# genC1[gene_used]
# genC2[gene_used]

# pd.concat([genC1[gene_used], genC2[gene_used]], ignore_index=True)
