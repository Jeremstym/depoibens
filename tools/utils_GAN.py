#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utils for GAN

# Utils for ST-Net regression

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image

plt.style.use("ggplot")

import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.utils as vutils


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss

    def __call__(self, path, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                path + "/outputs/bestGAN.pth",
            )


def save_model_generator(path, epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        path + "/outputs/netG.pth",
    )


def save_model_discriminator(path, epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        path + "/outputs/netD.pth",
    )


def plot_grid(img_batch, path_save: str, epoch: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                img_batch[0].to(device)[:64],
                padding=5,
                normalize=True,
            ).cpu(),
            (1, 2, 0),
        )
    )        
    plt.savefig(path_save + f"/fake_images_epoch{epoch}.jpg")


def plot_final_grid(real_batch, img_list, path_save) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=2, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig("/outputs/real_images.png")

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                img_list[-1],
                (1, 2, 0),
            )
        )
    )
    plt.savefig("/outputs/fake_images.png")
