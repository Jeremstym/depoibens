#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
print(os.getcwd())
import sys

# setting path
sys.path.append("../")
import pickle as pkl
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from data.dataset_stnet import Phenotypes

import PIL
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.models.inception import Inception_V3_Weights


tsv_path = "/projects/minos/jeremie/data/tsv_concatened.pkl"
embeddings_path = "/projects/minos/jeremie/data/embeddings_dict2.pkl"

### ------------ Network ---------------

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
model.eval()
model.to(device)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model.fc = Identity()

### ---------------- Create dataloader ------------------------


def create_dataloader(
    tsv_path=tsv_path, embeddings_path=embeddings_path, batch_size=16, num_workers=4
) -> data.DataLoader:
    """
    Create dataloader for images
    """
    with open(tsv_path, "rb") as f:
        tsv_concatened = pkl.load(f)
    with open(embeddings_path, "rb") as f:
        embeddings_dict = pkl.load(f)
    dataset = Phenotypes(tsv_concatened, embeddings_dict, model=model, device="cpu")
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    dataloader = create_dataloader(batch_size=16, num_workers=4)
    for i, (images, labels) in enumerate(dataloader):
        print(images.shape)
        print(labels.shape)
        break


### --------------- Brouillon ---------------

# if __name__ == "__main__":
#     path = "/import/pr_minos/jeremie/data"
#     with open(path + "/embeddings.pkl", "rb") as f:
#         embeddings = pkl.load(f)
#     all_emebeddings = torch.cat(embeddings, dim=0)
#     embedding_std = torch.std(all_emebeddings, dim=0, keepdim=True)
#     print(torch.topk(embedding_std, 10).indices)

# try_path = "/projects/minos/jeremie/data/BT23269/BT23269_C1/BT23269_C1_25x25.jpg"
# path = r"E:\ST-Net\data\hist2tscript\BRCA\BC23270"
# m = re.search("/projects/minos/jeremie/data/(.*)/(.*).jpg", try_path)
# m.group(2)
# for path_image in glob(path + "\*\*.jpg", recursive=True):
#     print(path_image)


# yield tensor([[ 552, 1382, 1171,  699,  663, 1502,  588,  436, 1222,  617]])

# preprocess = transforms.Compose(
#     [
#         transforms.Grayscale(num_output_channels=1),
#         transforms.Resize(30),
#         transforms.CenterCrop(30),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485], std=[0.229]),
#         transforms.Lambda(lambda x: torch.flatten(x)),
#     ]
# )

# preprocess(cell1).size()

# path = r"E:\ST-Net\data\hist2tscript\BRCA\BC23270"
# cell1 = Image.open(path + "\BC23270_E2.jpg")
# cell1 = cv2.imread(path + "\BC23270_E2.jpg", cv2.COLOR_BGR2RGB)
# # cell1 = cell1.resize((300,300))
# cell1.shape
# tensor = torch.tensor(np.transpose(cell1, (2, 0, 1)))
# tensor.size()
# # cv2.imshow("hello", cv2.resize(cell1, (300,300)))
# # plt.show()


# tensor = torch.randn((1, 28, 28)).unsqueeze(0)
# tensor = nn.Conv2d(1, 32, 3, 1)(tensor)
# tensor = nn.Conv2d(32, 64, 3, 1)(tensor)
# tensor = nn.Dropout2d(0.25)(tensor)
# tensor = F.max_pool2d(tensor, 2)
# tensor.size()
# nn.ConvTranspose2d()

# torch.flatten(tensor, 1).size()

# tensor = torch.randn((1, 28, 28)).unsqueeze(0)
# transforms.CenterCrop(28)(tensor)

# from torchvision.datasets import MNIST
# import torch.utils.data as data


# batch_size = 128
# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
# )
# train_set = MNIST(
#     r"C:\Jérémie\Stage\IBENS\depo\dataplus",
#     train=True,
#     transform=transform,
#     download=True,
# )
# train_loader = data.DataLoader(
#     train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
# )


# with open(path + "\BC23270_D2_complete.pkl", "rb") as f:
#     df_complete = pkl.load(f)


# class PandasDataset(data.Dataset):
#     def __init__(self, dataframe):
#         self.dataframe = dataframe

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()
#         coords = list(self.dataframe.iloc[index][:-1])
#         label = self.dataframe.iloc[index][-1]
#         return [coords, label]


# df_complete["label"] = df_complete["tumor"].apply(lambda x: 1 if x == "tumor" else 0)
# df_complete.drop(columns=["title", "lab", "tumor"], inplace=True)

# df = PandasDataset(df_complete)

# df[1]
# # df_complete = PandasDataset(df_complete)

# train_loader = data.DataLoader(df, batch_size=4)
# next(iter(train_loader))[1]

# for X, y in train_loader:
#     print(X)
#     print(y)
#     break

# df_complete.to_numpy()

# from collections import OrderedDict

# from ignite.engine import Engine
# # from ignite.handlers import *
# # from ignite.metrics import *
# # from ignite.utils import *
# # from ignite.contrib.metrics.regression import *
# # from ignite.contrib.metrics import

# # create default evaluator for doctests

# def eval_step(engine, batch):
#     return batch

# default_evaluator = Engine(eval_step)

# # create default optimizer for doctests

# param_tensor = torch.zeros([1], requires_grad=True)
# default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# # create default trainer for doctests
# # as handlers could be attached to the trainer,
# # each test must define his own trainer using `.. testsetup:`

# def get_default_trainer():

#     def train_step(engine, batch):
#         return batch

#     return Engine(train_step)

# # create default model for doctests

# default_model = nn.Sequential(OrderedDict([
#     ('base', nn.Linear(4, 2)),
#     ('fc', nn.Linear(2, 1))
# ]))


# from ignite.metrics import FID

# metric = FID(num_features=1, feature_extractor=default_model)
# metric.attach(default_evaluator, "fid")
# y_true = torch.ones(10, 4)
# y_pred = torch.ones(10, 4)
# state = default_evaluator.run([[y_pred, y_true]])
# print(state.metrics["fid"])
