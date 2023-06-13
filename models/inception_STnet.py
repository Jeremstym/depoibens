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
from torcheval.metrics import R2Score
from data.dataset_stnet import Phenotypes

import PIL
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.models.inception import Inception_V3_Weights

# data constants
BATCH_SIZE = 64
VALID_SPLIT = 0.2
NUM_WORKERS = 4

tsv_path = "/projects/minos/jeremie/data/tsv_concatened2.pkl"
embeddings_path = "/projects/minos/jeremie/data/embeddings_dict.pkl"

### ------------ Network ---------------

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
inception = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
inception.eval()
inception.to(device)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


inception.fc = Identity()

### ---------------- Create dataloader ------------------------


def create_dataloader(
    tsv_path=tsv_path,
    embeddings_path=embeddings_path,
    train_batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    test_patient="BC23270",
    test_batch_size=4,
) -> data.DataLoader:
    """
    Create dataloader for images
    """
    with open(tsv_path, "rb") as f:
        tsv_concatened = pkl.load(f)
    with open(embeddings_path, "rb") as f:
        embeddings_dict = pkl.load(f)

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
        embeddings_train = {
            k: embeddings_dict[k] for k in embeddings_dict.keys() if k in train_list
        }
        embedding_validation = {
            k: embeddings_dict[k] for k in test_list if k in validation_list
        }
        embeddings_test = {k: embeddings_dict[k] for k in test_list}

        trainset = Phenotypes(tsv_train, embeddings_train, model=inception)
        validationset = Phenotypes(
            tsv_validation, embedding_validation, model=inception
        )
        testset = Phenotypes(tsv_test, embeddings_test, model=inception)
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


if __name__ == "__main__":
    trainloader, validationloader, testloader = create_dataloader(
        train_batch_size=16, num_workers=4, test_patient="BC23270"
    )
    for i, (genotypes, images_embd) in enumerate(trainloader):
        print(genotypes.shape)
        print(images_embd.shape)
        break
    for i, (genotypes, images_embd) in enumerate(validationloader):
        print(genotypes.shape)
        print(images_embd.shape)
        break
    for i, (genotypes, images_embd) in enumerate(testloader):
        print(genotypes.shape)
        print(images_embd.shape)
        break

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


### --------------- Neural Network ---------------


class Regression_STnet(nn.Module):
    def __init__(self, input_size=900, hidden_size=512, output_size=10):
        super(Regression_STnet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


### --------------- Training ---------------


def train(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        metric = R2Score().to(device)
        with tqdm(dataloader, unit="batch") as pbar:
            running_loss = 0.0
            for genotypes, images_embd in pbar:
                pbar.set_description(f"Epoch {epoch}")
                genotypes = genotypes.to(device)
                genotypes = genotypes.float()
                images_embd = images_embd.to(device)
                optimizer.zero_grad()
                outputs = model(genotypes)
                loss = criterion(outputs, images_embd)
                metric.update(outputs, images_embd)
                # metric.compute()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item(), score=metric.compute().item())
                # if i % 100 == 99:
                #     print(
                #         "[%d, %5d] loss: %.3f"
                #         % (epoch + 1, i + 1, running_loss / 100)
                #     )
                #     running_loss = 0.0
    print("Finished Training")


def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for genotypes, images_embd in testloader:
            genotypes = genotypes.to(device)
            genotypes = genotypes.float()
            images_embd = images_embd.to(device)
            outputs = model(genotypes)
            loss = criterion(outputs, images_embd)
            test_loss += loss.item() * genotypes.size(0)
        print(f"Testing Loss:{test_loss/len(testloader)}")


# if __name__ == "__main__":
#     train_loader, test_loader = create_dataloader(
#         train_batch_size=16, num_workers=4, test_patient="BC23270"
#     )
#     model = Regression_STnet()
#     model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     train(model, train_loader, criterion, optimizer, device, epochs=10)
#     test(model, test_loader, criterion, device)

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
