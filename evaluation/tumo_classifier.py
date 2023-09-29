#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Create a binary classifier for the ST-Dataset using a CNN
# To detect tumours in the cell tissue patches

import os
import sys

import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from tumo_dataset import create_dataloader, create_generated_dataloader

# data constants

tumor_path = "/projects/minos/jeremie/data/complete_concatenate_df.csv"
path_to_image = "/projects/minos/jeremie/data"
path_to_classifier = "/projects/minos/jeremie/data/tumo.ckpt"


# create a binary CNN classifier
class TumoClassifier(nn.Module):
    def __init__(self):
        super(TumoClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=8)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# Train the model
# Set the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Create the model
    model = TumoClassifier().to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # import dataloader
    train_loader, valid_loader = create_dataloader(
        tumor_path=tumor_path, path_to_image=path_to_image
    )

    # Train the model
    # total_step = len(train_loader)
    for epoch in range(5):
        with tqdm(train_loader, unit="batch") as pbar:
            pbar.set_description(f"Epoch {epoch+1}")
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.float().unsqueeze(1)

                # Forward pass
                outputs = model(images.float())
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (i + 1) % 10 == 0:
                #     print(
                #         "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                #             epoch + 1, 5, i + 1, total_step, loss.item()
                #         )
                #     )
                pbar.set_postfix(loss=loss.item())

            

    # Evaluate the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        print("Evaluating model...")
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.float().unsqueeze(1)
            outputs = model(images.float())
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Test Accuracy of the model on the test images: {} %".format(
                100 * correct / total
            )
        )

    # Save the model checkpoint
    torch.save(model.state_dict(), "model_tumo.ckpt")

def load_and_evaluate_model():

    # import dataloader
    _train_loader, valid_loader = create_dataloader(
        tumor_path=tumor_path, path_to_image=path_to_image
    )
    # Create the model
    model = TumoClassifier().to(device)
    model.load_state_dict(torch.load("model_tumo.ckpt"))

    # Evaluate the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        print("Evaluating model...")
        count = 0
        score = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.float().unsqueeze(1)
            outputs = model(images.float())
            metric = BinaryF1Score().to(device)
            score += metric(outputs.T, labels.T).item()
            count += 1
        print(
            "F1 score of the model on the test images: {} %".format(
                100 * score / count
            )
        )

def load_and_evaluate_generated_model():

    # import dataloader
    dataloader = create_generated_dataloader()
    # Create the model
    model = TumoClassifier().to(device)
    model.load_state_dict(torch.load(path_to_classifier))

    # Evaluate the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        print("Evaluating model...")
        count = 0
        score = 0
        with tqdm(dataloader, unit="batch") as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.float().unsqueeze(1)
                outputs = model(images.float())
                metric = BinaryF1Score().to(device)
                score += metric(outputs.T, labels.T).item()
                count += 1
                pbar.set_postfix(f1_score=score/count)
        print(
            "F1 score of the model on the test images: {} %".format(
                100 * score / count
            )
        )

if __name__ == "__main__":
    load_and_evaluate_generated_model()