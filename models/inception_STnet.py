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
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

# from torcheval.metrics import R2Score
from torchmetrics import R2Score
from torchmetrics import PearsonCorrCoef
from data.dataset_stnet import create_dataloader

if __name__ == "__main__":
    from utils import SaveBestModel, save_model, save_plots

import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported

PENALIZATION = 0.5
list_patients = [
    "BC23209",
    "BT23268",
    "BT23288",
    "BT23567",
    "BT23944",
    "BC23270",
    "BT23269",
    "BT23377",
    "BT23810",
    "BT24044",
    "BC23803",
    "BT23272",
    "BT23450",
    "BT23895",
    "BT24223",
    "BC24105",
    "BT23277",
    "BT23506",
    "BT23901",
    "BC24220",
    "BT23287",
    "BT23508",
    "BT23903",
]

### --------------- Neural Network ---------------


# class Regression_STnet(nn.Module):
#     def __init__(self, input_size=900, hidden_size=650, output_size=10):
#         super(Regression_STnet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc3 = nn.Linear(hidden_size // 2, output_size)
#         self.dropout = nn.Dropout(p=0.2)

#     def forward(self, x):
#         x = self.dropout(F.gelu(self.fc1(x)))
#         x = self.dropout(F.gelu(self.fc2(x)))
#         x = self.fc3(x)
#         return x


class Regression_STnet(nn.Module):
    def __init__(
        self,
        input_size=900,
        hidden_size=2048,
        output_size=2048,
        dropout=0.2,
        batch_norm=True,
    ):
        super(Regression_STnet, self).__init__()
        self.p = dropout
        if batch_norm:
            print("Using batch norm")

        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_size) if batch_norm else nn.Identity(),
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(self.p),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.BatchNorm1d(2 * hidden_size),
            nn.GELU(),
            nn.Dropout(self.p),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(self.p),
            nn.Linear(hidden_size, output_size),
            # No activation function here because we're doing regression
        )

    def forward(self, x):
        return self.layers(x)


class DummyRegression_STnet(nn.Module):
    def __init__(self, output_size=10):
        super(DummyRegression_STnet, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return torch.randint(0, 30, (x.shape[0], self.output_size))


### --------------- Training ---------------


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    current_epoch,
    run=None,
):
    model.train()
    print("Start Training")
    # for epoch in range(epochs):
    # metric_unif = R2Score(multioutput="uniform_average").to(device)
    with tqdm(dataloader, unit="batch") as pbar:
        running_loss = 0.0
        running_r2score_wght = 0.0
        running_pearson_coef = 0.0
        counter = 0
        # reg_loss = 0
        pbar.set_description(f"Epoch {current_epoch+1}")
        for genotypes, images_embd in pbar:
            counter += 1
            genotypes = genotypes.float()
            genotypes = genotypes.to(device)
            images_embd = images_embd.squeeze(1)
            images_embd = images_embd.to(device)
            optimizer.zero_grad()
            outputs = model(genotypes)
            # reg_loss += nn.L1Loss(size_average=False)(outputs, images_embd)
            # factor = PENALIZATION
            loss = criterion(outputs, images_embd)
            # loss += factor * reg_loss
            # metric_unif.update(outputs, images_embd)
            pearson = PearsonCorrCoef(num_outputs=genotypes.size(0)).to(device)
            pearson_coefs = pearson(outputs.T, images_embd.T)
            pearson_coef = torch.mean(pearson_coefs)
            r2 = R2Score(
                num_outputs=genotypes.size(0), multioutput="variance_weighted"
            ).to(device)
            # metric_wght.update(outputs, images_embd)
            metric_wght = r2(outputs.T, images_embd.T)
            if run and counter % 30 == 0:
                # run["train/batch/r2score_unif"].append(metric_unif.compute().item())
                run["train/batch/loss"].append(loss.item())
                run["train/batch/pearson"].append(pearson_coef.item())
                run["train/batch/r2score_wght"].append(metric_wght.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # running_r2score_unif += metric_unif.compute().item()
            running_pearson_coef += pearson_coef.item()
            running_r2score_wght += metric_wght.item()
            pbar.set_postfix(
                loss=loss.item(), score=metric_wght.item(), pearson=pearson_coef.item()
            )
            # if i % 100 == 99:
            #     print(
            #         "[%d, %5d] loss: %.3f"
            #         % (epoch + 1, i + 1, running_loss / 100)
            #     )
            #     running_loss = 0.0
        # print("Finished Training")
        epoch_loss = running_loss / counter
        epoch_r2score_wght = running_r2score_wght / counter
        epoch_pearson_coef = running_pearson_coef / counter
        # epoch_r2score_unif = running_r2score_unif / counter
        if run:
            run["train/epoch/loss"].append(epoch_loss)
            run["train/epoch/r2score_wght"].append(epoch_r2score_wght)
            run["train/epoch/pearson"].append(epoch_pearson_coef)
            # run["train/epoch/r2score_unif"].append(epoch_r2score_unif)
        return epoch_loss, epoch_r2score_wght, epoch_pearson_coef


def validate(model, dataloader, criterion, device, run=None):
    model.eval()
    print("Validation")
    valid_running_loss = 0.0
    running_r2score_wght = 0.0
    running_pearson_coef = 0.0
    # running_r2score_unif = 0.0
    counter = 0
    # metric_unif = R2Score(multioutput="uniform_average").to(device)
    with torch.no_grad():
        for genotypes, images_embd in dataloader:
            counter += 1
            genotypes = genotypes.float()
            genotypes = genotypes.to(device)
            images_embd = images_embd.squeeze(1)
            images_embd = images_embd.to(device)
            outputs = model(genotypes)
            loss = criterion(outputs, images_embd)
            valid_running_loss += loss.item()
            # metric_unif.update(outputs, images_embd)
            pearson = PearsonCorrCoef(num_outputs=genotypes.size(0)).to(device)
            pearson_coefs = pearson(outputs.T, images_embd.T)
            pearson_coef = torch.mean(pearson_coefs)
            # metric_wght.update(outputs, images_embd)
            r2 = R2Score(
                num_outputs=genotypes.size(0), multioutput="variance_weighted"
            ).to(device)
            metric_wght = r2(outputs.T, images_embd.T)
            running_r2score_wght += metric_wght.item()
            running_pearson_coef += pearson_coef.item()
            # running_r2score_unif += metric_unif.compute().item()
        epoch_loss = valid_running_loss / counter
        epoch_r2score_wght = running_r2score_wght / counter
        epoch_pearson_coef = running_pearson_coef / counter
        # epoch_r2score_unif = running_r2score_unif / counter
        if run:
            run["valid/epoch/loss"].append(epoch_loss)
            run["valid/epoch/r2score_wght"].append(epoch_r2score_wght)
            run["valid/epoch/pearson"].append(epoch_pearson_coef)
            # run["valid/epoch/r2score_unif"].append(epoch_r2score_unif)
        print(f"Validation Loss:{epoch_loss}")
        print(f"Validation Score:{epoch_r2score_wght}")
        print(f"Validation Pearson:{epoch_pearson_coef}")
        return epoch_loss, epoch_r2score_wght, epoch_pearson_coef


def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_r2score_wght = 0.0
    test_pearson_coef = 0.0
    counter = 0
    with torch.no_grad():
        for genotypes, images_embd in testloader:
            counter += 1
            genotypes = genotypes.to(device)
            genotypes = genotypes.float()
            images_embd = images_embd.squeeze(1)
            images_embd = images_embd.to(device)
            outputs = model(genotypes)
            loss = criterion(outputs, images_embd)
            test_loss += loss.item()
            r2 = R2Score(
                num_outputs=genotypes.size(0), multioutput="variance_weighted"
            ).to(device)
            metric_wght = r2(outputs.T, images_embd.T)
            # metric_wght.update(outputs, images_embd)
            pearson = PearsonCorrCoef(num_outputs=genotypes.size(0)).to(device)
            pearson_coefs = pearson(outputs.T, images_embd.T)
            pearson_coef = torch.mean(pearson_coefs)
            # test_r2score_wght += metric_wght.compute().item() * genotypes.size(0)
            test_r2score_wght += metric_wght.item()
            test_pearson_coef += pearson_coef.item()
        print(
            f"Testing Loss:{test_loss/counter}, Score:{test_r2score_wght/counter}, Pearson:{test_pearson_coef/counter}"
        )
        return (
            test_loss / counter,
            test_r2score_wght / counter,
            test_pearson_coef / counter,
        )


### --------------- Main ---------------

# construct the argument parser

parser = argparse.ArgumentParser()
parser.add_argument(
    "-use_neptune",
    "--neptune",
    type=bool,
    default=False,
    help="Use neptune to log the training",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=20,
    help="number of epochs to train our network for",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=0.0001,
    help="Learning rate for the training",
)
parser.add_argument(
    "-dropout",
    "--dropout",
    type=float,
    default=0.6,
    help="Dropout rate for the training",
)
parser.add_argument(
    "-dummy",
    "--dummy",
    type=bool,
    default=False,
    help="Use dummy model for testing purpose",
)
parser.add_argument(
    "-batch_norm",
    "--batch_norm",
    type=bool,
    default=True,
    help="Use batch normalization for input",
)
parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=256,
    help="Batch size for the training",
)
parser.add_argument(
    "-name",
    "--model-name",
    type=str,
    default="STNet-regression",
    help="Name of the model",
)
parser.add_argument(
    "-test",
    "--test",
    type=bool,
    default=False,
    help="Test the model",
)
args = vars(parser.parse_args())


def main(
    path_saving="/import/pr_minos/jeremie/data",
    lr=args["learning_rate"],
    epochs=args["epochs"],
    dummy=args["dummy"],
    dropout=args["dropout"],
    input_size=900,
    hidden_size=3056,
    output_size=2048,
    is_single_test=args["test"],
):
    params = {
        "lr": lr,
        "bacth_size": args["batch_size"],
        "test_bacth_size": 16,
        # "input_sz": 32 * 32 * 3,
        # "n_classes": 10,
        "model_filename": args["model_name"],
        # "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "epochs": epochs,
        "dropout": dropout,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
    }
    if args["neptune"]:
        run = neptune.init_run(
            project="jeremstym/STNet-Regression",
        )
        run["parameters"] = params
    else:
        run = None

    # learning_parameters
    lr = params["lr"]
    epochs = params["epochs"]
    # computation device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    # if dummy:
    #     device = torch.device("cpu")
    #     model = DummyRegression_STnet()
    if is_single_test:
        model = Regression_STnet(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            batch_norm=args["batch_norm"],
        )
        model.load_state_dict(
            torch.load("/projects/minos/jeremie/data/outputs/best_model4_norm.pth")[
                "model_state_dict"
            ]
        )
        model.to(device)
        model.eval()
    else:
        model = Regression_STnet(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            batch_norm=args["batch_norm"],
        )
        model.to(device)

    # npt_logger = NeptuneLogger(
    #     run=run,
    #     model=model,
    #     log_model_diagram=True,
    #     log_gradients=True,
    #     log_parameters=True,
    #     log_freq=30,
    # )

    # run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(params)
    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.\n")
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.1, patience=10, verbose=True
    # )
    # loss function
    criterion = nn.MSELoss()
    # initialize SaveBestModel class
    save_best_model = SaveBestModel()

    test_frame = pd.DataFrame(columns=["test_loss", "r2_score", "pearson"])
    for test_patient in list_patients:
        print(f"Test patient: {test_patient}")
        # create dataloader
        train_loader, valid_loader, test_loader = create_dataloader(
            train_batch_size=params["bacth_size"],
            test_batch_size=params["test_bacth_size"],
            input_size=params["input_size"],
            output_size=params["output_size"],
            test_patient=test_patient,
        )
        # initiate model
        model = Regression_STnet(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            batch_norm=args["batch_norm"],
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # start training
        train_loss, valid_loss = [], []
        train_pearson_list, valid_pearson_list = [], []
        train_r2_wght, valid_r2_wght = [], []
        for epoch in range(epochs):
            print(f"[INFO]: Epoch {epoch+1} of {epochs}")
            if not dummy:
                train_epoch_loss, train_r2score_wght, train_pearson = train(
                    model, train_loader, criterion, optimizer, device, epoch, run
                )
            else:
                train_epoch_loss, train_r2score_wght, train_pearson = 0, 0, 0

            valid_epoch_loss, valid_r2score_wght, valid_pearson = validate(
                model, valid_loader, criterion, device, run
            )
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_pearson_list.append(train_pearson)
            valid_pearson_list.append(valid_pearson)
            train_r2_wght.append(train_r2score_wght)
            valid_r2_wght.append(valid_r2score_wght)
            print(
                f"Training loss: {train_epoch_loss:.3f}, training r2_wght: {train_r2score_wght:.3f}, training pearson: {train_pearson:.3f}"
            )
            print(
                f"Validation loss: {valid_epoch_loss:.3f}, validation r2_wght: {valid_r2score_wght:.3f}, validation pearson: {valid_pearson:.3f}"
            )
            # save the best model till now if we have the least loss in the current epoch
            if not dummy:
                save_best_model(
                    path_saving,
                    valid_epoch_loss,
                    epoch,
                    model,
                    optimizer,
                    criterion,
                )
            # scheduler.step(valid_epoch_loss)
            print("-" * 50)
        if run:
            run.stop()
        if not dummy:
            # save the trained model weights for a final time
            save_model(path_saving, epochs, model, optimizer, criterion)
            # save the loss and accuracy plots
            save_plots(
                path_saving, train_r2_wght, valid_r2_wght, train_loss, valid_loss
            )
        print("TRAINING COMPLETE")

        # test the model
        if not dummy:
            test_loss, r2_test, pearson_test = test(
                model, test_loader, criterion, device
            )
            test_frame.loc[test_patient] = [test_loss, r2_test, pearson_test]
            print("TESTING COMPLETE")

    print(test_frame)
    test_frame.to_csv("/projects/minos/jeremie/data/outputs/test_results.csv")
    # additional save of the test results
    with open("/projects/minos/jeremie/data/outputs/test_results.pkl", "w") as f:
        pkl.dump(test_frame, f)


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     lr_list = np.geomspace(1e-3, 1e-5, num=10)
#     dropout_list = np.linspace(0.2, 0.7, num=6)
#     input_size_list = [850, 800, 600, 400, 200, 100, 50, 25, 10]
#     output_size_list = [1024, 512, 256, 128, 64, 32, 10]
#     main()
#     print("Iterate on dropout")
#     for dp in dropout_list:
#         main(dropout=dp)
#     print("Iterate on input size")
#     for input_size in input_size_list:
#         main(input_size=input_size)
#     print("Iterate on output size")
#     for output_size in output_size_list:
#         main(output_size=output_size)
#     print("Iterate on learning rate")
#     for lr in lr_list:
#         main(lr=lr)
#     print("End of the script")

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


# target = torch.tensor([[0.5, 1, 0.5], [-1, 1, 1], [7, -6, 3]])
# preds = torch.tensor([[0, 2, 4], [-1, 2, 7], [8, -5, 0]])
# r2score = R2Score(num_outputs=3, multioutput='raw_values')
# r2score(preds, target)

# target.shape

# t = torch.tensor([[-0.1321,  1.9615, -0.3195],
#         [-0.1669,  0.9056,  1.0235],
#         [ 0.0698, -0.9476, -0.9968],
#         [ 1.7891,  0.9675, -1.2795]])

# t1 = torch.tensor([[ 0.8018,  0.9627,  0.0036],
#         [ 0.3374, -0.3799,  0.4940],
#         [-0.0533,  0.7327,  1.0212],
#         [-1.0302,  1.4651,  0.3806]])

# r2 = R2Score(num_outputs=3, multioutput='variance_weighted')
# r2(t, t1)
