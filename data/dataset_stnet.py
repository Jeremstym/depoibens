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

# data constants
BATCH_SIZE = 64
VALID_SPLIT = 0.2
NUM_WORKERS = 4

tsv_path = "/projects/minos/jeremie/data/tsv_concatened_allgenes.pkl"
embeddings_path = "/projects/minos/jeremie/data/embeddings_dict.pkl"
embeddings_path_dino = "/projects/minos/jeremie/data/dino_features.pkl"
selection_tensor_path = "/projects/minos/jeremie/data/features_std.pkl"
path_to_image = "/projects/minos/jeremie/data"


### ---------------- Customized Inception model ------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
inception = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


inception.fc = Identity()

### ---------------- Pre-processing for images ------------------

# selection_tensor = torch.tensor(
#         [[552, 1382, 1171, 699, 663, 1502, 588, 436, 1222, 617]]
#     )
# selection_tensor = selection_tensor.to(device)


def image_embedding(path: str, pre_trained: nn.Module, device=device):
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
    pre_trained.eval()
    pre_trained.to(device)
    with torch.no_grad():
        output = pre_trained(input_batch)
    return output


def embed_all_images(path: str, pre_trained=inception, device=device):
    embeddings_dict = {}
    for sub_path in tqdm(glob(path + "/*/", recursive=True)):
        with tqdm(glob(sub_path + "/*/*.jpg", recursive=True), unit="spot") as pbar:
            for path_image in pbar:
                m = re.search("data/(.*)/(.*).jpg", path_image)
                if m:
                    embeddings_dict[m.group(2)] = image_embedding(
                        path_image, pre_trained, device
                    ).to("cpu")
                else:
                    raise ValueError("Path not found")
                pbar.set_description(f"Processing {m.group(2)}")
    return embeddings_dict


def change_device_embedding(embeddings_dict):
    for key in embeddings_dict.keys():
        embeddings_dict[key] = embeddings_dict[key].to("cpu")
    return embeddings_dict


# def compute_biggest_std(embeddings_dict):
#     stds = []
#     for key in embeddings_dict.keys():
#         stds.append(embeddings_dict[key])
#     stds = torch.stack(stds)
#     stds = torch.std(stds, dim=0)
#     return stds.topk(10, largest=True, sorted=True).indices


def list_stds(embeddings_dict):
    stds = []
    for key in tqdm(embeddings_dict.keys(), unit="spot"):
        stds.append(embeddings_dict[key])
    stds = torch.stack(stds)
    stds = torch.std(stds, dim=0)
    return stds.argsort(descending=True)


# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     dino_dict = embed_all_images(path, dino, device)
#     with open(path + "/dino_features.pkl", "wb") as f:
#         pkl.dump(dino_dict, f)


# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     with open(path + "/embeddings_dict2.pkl", "rb") as f:
#         embeddings_dict = pkl.load(f)
#     stds = list_stds(embeddings_dict)
#     with open(path + "/features_std.pkl", "wb") as f:
#         pkl.dump(stds, f)

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
#         bestgene = list(pkl.load(f).index)
#     df = concat_tsv(path, bestgene)
#     with open(path + "/tsv_concatened_allgenes.pkl", "wb") as f:
#         pkl.dump(df, f)

### ---------------- Create dataset ------------------


class Dataset_STnet(data.Dataset):
    def __init__(
        self,
        tsv_concatened,
        embeddings_dict,
        selection_tensor=None,
        nb_genes=900,
        embd_size=2048,
    ) -> None:
        super().__init__()
        self.genotypes = tsv_concatened.drop("tissue", axis=1)[
            tsv_concatened.columns[:nb_genes]
        ]
        self.embeddings_dict = embeddings_dict
        if selection_tensor:
            self.selection_list = (
                selection_tensor[:, :embd_size].sort(descending=True).values.tolist()
            )
        else:
            self.selection_list = list(range(embd_size))

    def __len__(self):
        return len(self.embeddings_dict)

    def __getitem__(self, idx_number: int):
        index = list(self.embeddings_dict.keys())[idx_number]
        return (
            torch.tensor(self.genotypes.loc[index].values),  # (nb_genes,)
            self.embeddings_dict[index][0, self.selection_list],  # (embd_size,)
        )

    def get_index_name(self, idx_number: int):
        return list(self.embeddings_dict.keys())[idx_number]


class Phenotype(data.Dataset):
    def __init__(
        self,
        path_to_image: str,
        size=299,
    ) -> None:
        self.path = path_to_image
        self.size = size
        self.data = pd.DataFrame(columns=["name", "image"])
        os.chdir(self.path)
        for image in glob("*/*/*.jpg"):
            img = Image.open(image)
            img_name = image[19:-4]
            img_preprocessed = self.preprocess(img)
            self.data = pd.concat(
                [self.data, pd.DataFrame({"name": img_name, "image": img_preprocessed})]
            )

    def preprocess(self, image):
        size = self.size
        preprocess = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        preprocessed_image = preprocess(image)
        return preprocessed_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx_number: int):
        return self.data.iloc[idx_number]["image"]


### ---------------- Create dataloader ------------------------


def create_dataloader(
    tsv_path=tsv_path,
    embeddings_path=embeddings_path,
    selection_tensor_path=None,
    input_size=900,
    output_size=2048,
    train_batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    test_patient="BC23270",
    test_batch_size=16,
) -> data.DataLoader:
    """
    Create dataloader for images
    """
    with open(tsv_path, "rb") as f:
        tsv_concatened = pkl.load(f)
    with open(embeddings_path, "rb") as f:
        embeddings_dict = pkl.load(f)
    if selection_tensor_path:
        with open(selection_tensor_path, "rb") as f:
            selection_tensor = pkl.load(f)
    else:
        selection_tensor = None

    # # number if validation images
    # train_dataset_size = len(tsv_concatened)

    # validation size = number of validation images
    valid_size = VALID_SPLIT

    if test_patient:
        tsv_train = tsv_concatened[~tsv_concatened.index.str.startswith(test_patient)]
        tsv_validation = tsv_train.groupby("tissue").sample(
            frac=valid_size, random_state=42
        )
        tsv_train = tsv_train.drop(tsv_validation.index)
        tsv_test = tsv_concatened[tsv_concatened.index.str.startswith(test_patient)]

        train_list = list(tsv_train.index)
        validation_list = list(tsv_validation.index)
        test_list = list(tsv_test.index)
        embeddings_train = {k: embeddings_dict[k] for k in train_list}
        embedding_validation = {k: embeddings_dict[k] for k in validation_list}
        embeddings_test = {k: embeddings_dict[k] for k in test_list}

        trainset = Dataset_STnet(
            tsv_train,
            embeddings_train,
            selection_tensor,
            nb_genes=input_size,
            embd_size=output_size,
        )
        validationset = Dataset_STnet(
            tsv_validation,
            embedding_validation,
            selection_tensor,
            nb_genes=input_size,
            embd_size=output_size,
        )
        testset = Dataset_STnet(
            tsv_test,
            embeddings_test,
            selection_tensor,
            nb_genes=input_size,
            embd_size=output_size,
        )
        trainloader = data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers
        )
        validationloader = data.DataLoader(
            validationset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        testloader = data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
        )
        return trainloader, validationloader, testloader

    else:
        raise ValueError("test_patient must be specified")


def create_GAN_dataloader(
    image_path=path_to_image,
    train_batch_size=BATCH_SIZE,
    # test_patient="BC23270",
    # test_batch_size=16,
) -> data.DataLoader:
    trainset = Phenotype(image_path)
    trainloader = data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=4
    )
    return trainloader


# if __name__ == "__main__":
#     trainloader, validationloader, testloader = create_dataloader(
#         train_batch_size=16, num_workers=4, test_patient="BC23270"
#     )
#     for i, (genotypes, images_embd) in enumerate(trainloader):
#         print("train", genotypes.shape)
#         print("train", images_embd.shape)
#         break
#     for i, (genotypes, images_embd) in enumerate(validationloader):
#         print("validation", genotypes.shape)
#         print("validation", images_embd.shape)
#         break
#     for i, (genotypes, images_embd) in enumerate(testloader):
#         print("test", genotypes.shape)
#         print("test", images_embd.shape)
#         break

# if __name__ == "__main__":
#     train_loader, test_loader = create_dataloader(
#         train_batch_size=16, num_workers=4, test_patient="BC23270"
#     )
#     for i, (genotypes, images_embd) in enumerate(train_loader):
#         print(genotypes.index)  # (16, 900)
#         print(images_embd.keys())
#         print(list(genotypes.index) == list(images_embd))  # (16, 10)
#         break
# for i, (genotypes, images_embd) in enumerate(test_loader):
#     print(genotypes.shape) # (4, 900)
#     print(images_embd.shape) # (4, 10)
#     break


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
#     st_set = Dataset_STnet(path)
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
