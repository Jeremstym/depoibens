#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Perform PCA on data embeddings

import os

print(os.getcwd())
import sys

# setting path
sys.path.append("../")
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.inception_STnet import Regression_STnet

import torch

# constants
path_to_features = "/projects/minos/jeremie/data/features_std.pkl"
path_to_dict = "/projects/minos/jeremie/data/embeddings_dict.pkl"
path_to_tsv = "/projects/minos/jeremie/data/tsv_concatened_allgenes.pkl"
path_to_std = "/projects/minos/jeremie/data/std_genes_avg.pkl"
path_to_model = "/projects/minos/jeremie/data/outputs/best_model4_norm.pth"

INPUT_SIZE = 900
OUTPUT_SIZE = 2048
HIDDEN_SIZE = 3056
PATIENT_TISSUE = "BT23944_E2"

# PATIENT_TISSUE = "BT23450_E1"

### ------------------- Preprocessing -------------------


def get_embeddings_from_dict(path_to_dict: str, patient=PATIENT_TISSUE) -> pd.DataFrame:
    """
    Function to get the embeddings of a given patient.
    """
    print("Loading embeddings dict...")
    with open(path_to_dict, "rb") as f:
        embeddings_dict = pkl.load(f)

    list_patient = [spot for spot in embeddings_dict.keys() if spot.startswith(patient)]
    dict_df = {spot: embeddings_dict[spot].squeeze(0) for spot in list_patient}

    df = pd.DataFrame(dict_df).T
    return df


def get_embeddings_from_tsv(path_to_tsv: str, patient=PATIENT_TISSUE) -> pd.DataFrame:
    """
    Function to get the embeddings of a given patient.
    """
    print("Loading embeddings tsv...")
    with open(path_to_tsv, "rb") as f:
        tsv = pkl.load(f)

    df = tsv[tsv["tissue"] == patient].drop(columns=["tissue"])
    return df


### ------------------- Embeddings with regression model -------------------
# load data
def embedding_tsv(tsv: pd.DataFrame, path_to_model=path_to_model) -> np.ndarray:
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # load model
    model = Regression_STnet(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout=0.6,
        batch_norm=True,
    )
    print(f"Loading model...")
    model.load_state_dict(torch.load(path_to_model)["model_state_dict"])
    model.to(device)
    model.eval()

    # get embeddings
    # embeddings = model["model_state_dict"]["fc2.weight"].cpu().numpy()
    tsv_tensor = torch.tensor(tsv.values, dtype=torch.float32).to(device)
    embeddings = model(tsv_tensor)
    embeddings = embeddings.detach().cpu().numpy()

    print(f"Embeddings: {embeddings.shape}")
    return embeddings


### ------------------- PCA -------------------


def pca(data_0, data_1=None, n_components=2) -> np.ndarray:
    """
    Function to perform PCA on the data
    """
    if data_1 is not None:
        data = np.concatenate((data_0, data_1), axis=0)
        print(f"Loaded data: {data.shape}")
    else:
        data = data_0
        print(f"Loaded data: {data.shape}")

    # standardize data
    data_std = StandardScaler().fit_transform(data)

    # perform PCA
    pca = PCA(n_components=n_components)
    pca_fit = pca.fit(data_std)
    data_pca = pca_fit.transform(data_std)
    print(f"PCA explained variance: {pca_fit.explained_variance_ratio_}")
    print(f"PCA data: {data_pca.shape}")
    if data_1 is not None:
        data_pca_0 = data_pca[: data_0.shape[0]]
        data_pca_1 = data_pca[data_0.shape[0] :]
        return data_pca_0, data_pca_1
    else:
        return data_pca, pca_fit


def plot_pca(data_pca: np.ndarray, name: str, color_index: int) -> None:
    colors = ["turquoise", "darkorange"]
    plt.scatter(
        data_pca[:, 0],
        data_pca[:, 1],
        c=colors[color_index],
        lw=2,
        alpha=0.5,
        label=name,
    )


# if __name__ == "__main__":
#     print("Loading std...")
#     with open(path_to_std, "rb") as f:
#         std_tsv = pkl.load(f)
#     tsv = get_embeddings_from_tsv(path_to_tsv)
#     tsv = tsv[std_tsv.index[:900]]
#     tsv_embed = embedding_tsv(tsv)
#     print("Loading features...")
#     with open(path_to_features, "rb") as f:
#         features = pkl.load(f).squeeze(0).to("cpu").tolist()
#     embds = get_embeddings_from_dict(path_to_dict)
#     embds = embds[features[:2048]].values

#     pca_res0, pca_res1 = pca(embds, tsv_embed)
#     plot_pca(pca_res0, "PCA on data", 0)
#     plot_pca(pca_res1, "PCA on Regression output", 1)
#     print("Plotting PCA...")
#     plt.legend(loc="best", shadow=False, scatterpoints=1)
#     plt.title(f"PCA of ST-Net dataset {PATIENT_TISSUE}")
#     plt.savefig("/projects/minos/jeremie/data/outputs/PCA_final2.png")

### Optional savings below ###
### To add to the pca function ###
# # save PCA data
# np.savetxt(path + "/data/embeddings_pca.csv", data_pca, delimiter=",")
# print(f"Saved PCA data to: {path}/data/embeddings_pca.csv")

# # save PCA model
# np.savetxt(path + "/models/pca_model.csv", pca.components_, delimiter=",")
# print(f"Saved PCA model to: {path}/models/pca_model.csv")

# # save PCA explained variance ratio
# np.savetxt(
#     path + "/models/pca_explained_variance_ratio.csv",
#     pca.explained_variance_ratio_,
#     delimiter=",",
# )
# print(
#     f"Saved PCA explained variance ratio to: {path}/models/pca_explained_variance_ratio.csv"
# )

# # save PCA singular values
# np.savetxt(
#     path + "/models/pca_singular_values.csv",
#     pca.singular_values_,
#     delimiter=",",
# )
# print(f"Saved PCA singular values to: {path}/models/pca_singular_values.csv")

# # save PCA mean
# np.savetxt(path + "/models/pca_mean.csv", pca.mean_, delimiter=",")
# print(f"Saved PCA mean to: {path}/models/pca_mean.csv")

# # save PCA noise variance
# np.savetxt(
#     path + "/models/pca_noise_variance.csv",
#     pca.noise_variance_,
#     delimiter=",",
# )
# print(f"Saved PCA noise variance to: {path}/models/pca_noise_variance.csv")

# # save PCA n_components
# np.savetxt(
#     path + "/models/pca_n_components.csv",
#     pca.n_components_,
#     delimiter=",",
# )
# print(f"Saved PCA n_components to: {path}/models/pca_n_components.csv")

# # save PCA explained variance
# np.savetxt(
#     path + "/models/pca_explained_variance.csv",
#     pca
# )

### ------------------- Brouillon -------------------

# from sklearn import datasets
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# iris = datasets.load_iris()

# X = iris.data
# y = iris.target
# target_names = iris.target_names

# pca = PCA(n_components=2)
# X_r = pca.fit_transform(X)

path = "/projects/minos/jeremie/data"

from glob import glob

path_list = glob(path + "/*/*/")
import re

text = " ".join(path_list)

pattern = "data/(.*?)/(.*_[CDE][0-9]?)/"
tissues = re.findall(pattern, text)
tissues = [t[0] for t in tissues]
tissues = list(set(tissues))
tissues.sort()
print(tissues)