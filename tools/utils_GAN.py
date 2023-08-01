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
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from typing import List


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
        path + "/final_outputs/netG.pth",
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
        path + "/final_outputs/netD.pth",
    )


def show_tensor_images(
    image_tensor: torch.Tensor,
    path_save: str,
    epoch: int,
    num_images=25,
    size=(3, 300, 300),
    real=False,
) -> None:
    """Function to display the images."""
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = (
        make_grid(image_unflat[:num_images], nrow=5)
        .mul(0.5)
        .add(0.5)
        .mul(255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    # plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()
    if not real:
        image_grid = to_pil_image(image_grid)
        image_grid.save(path_save + f"/fake_images_epoch{epoch}.jpg")
    else:
        Image.fromarray(image_grid).save(path_save + f"/true_images_epoch{epoch}.jpg")


def show_final_grid(
    real_batch: List[torch.Tensor],
    img_list: List[torch.Tensor],
    path_save: str,
    num_image=25,
    size=(3, 300, 300),
) -> None:
    """Function to display the final grid of images.

    Args:
        real_batch (List[torch.Tensor]): batch of real images
        img_list (List[torch.Tensor]): batch of fake images
        path_save (str): path to save the images
        num_image (int, optional): number of image per grid. Defaults to 25.
        size (tuple, optional): size of each image. Defaults to (3, 300, 300).
    """
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    image_unflat = real_batch[0].detach().cpu().view(-1, *size)
    image_grid = (
        make_grid(image_unflat[:num_image], nrow=5)
        .mul(0.5)
        .add(0.5)
        .mul(255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    Image.fromarray(image_grid).save(path_save + "/real_final_images.jpg")

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    image_unflat = img_list[-1].detach().cpu().view(-1, *size)
    image_grid = (
        make_grid(image_unflat[:num_image], nrow=5)
        .mul(0.5)
        .add(0.5)
        .mul(255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    Image.fromarray(image_grid).save(path_save + "/fake_final_images.jpg")


def plot_losses(
    path_save: str,
    generator_losses: List[float],
    discriminator_losses: List[float],
) -> None:
    """Function to plot the losses of the generator and discriminator.

    Args:
        path_save (str): path to save the images
        epochs (int): number of epochs
        generator_losses (List[float]): list of generator losses
        discriminator_losses (List[float]): list of discriminator losses
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_losses, label="G", color="red", linestyle="dashed", linewidth=2)
    plt.plot(discriminator_losses, label="D", color="blue", linewidth=2)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path_save + "/losses.png")


### ------------------- Brouillon -------------------

# def plot_final_grid(real_batch, img_list, path_save, num_image=25) -> None:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     plt.figure(figsize=(15, 15))
#     plt.subplot(1, 2, 1)
#     plt.axis("off")
#     plt.title("Real Images")
#     plt.imshow(
#         np.transpose(
#             vutils.make_grid(
#                 real_batch[0].to(device)[:num_image], nrow=5, padding=2, normalize=True
#             ).cpu(),
#             (1, 2, 0),
#         )
#     )
#     plt.savefig("/final_outputs/real_images.png")

#     # Plot the fake images from the last epoch
#     plt.subplot(1, 2, 2)
#     plt.axis("off")
#     plt.title("Fake Images")
#     plt.imshow(
#         np.transpose(
#             vutils.make_grid(
#                 img_list[-1],
#                 (1, 2, 0),
#             )
#         )
#     )
#     plt.savefig("/final_outputs/fake_images.png")

# def plot_grid(img_batch: torch.Tensor, path_save: str, epoch: int) -> None:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     plt.figure(figsize=(8, 8))
#     plt.axis("off")
#     plt.title("Fake Images")
#     plt.imshow(
#         np.transpose(
#             vutils.make_grid(
#                 img_batch[0].to(device)[:64],
#                 padding=5,
#                 normalize=True,
#             ).cpu(),
#             (1, 2, 0),
#         )
#     )
#     plt.savefig(path_save + f"/fake_images_epoch{epoch}.jpg")
