#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Utils for ST-Net regression

import os
import torch
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss

    def __call__(self, path, current_valid_loss, epoch, model, optimizer, criterion, test_patient):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            
            if not os.path.exists(path + "/outputs"):
                os.makedirs(path + "/outputs")
            
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                path + f"/outputs/best_model_dino_{test_patient}.pth",
            )


def save_model(path, epochs, model, optimizer, criterion, test_patient):
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
        path + f"/outputs/final_model_dino_{test_patient}.pth",
    )


def save_plots(path, train_score, valid_score, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_score, color="green", linestyle="-", label="train accuracy")
    plt.plot(valid_score, color="blue", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("R2 score")
    plt.legend()
    plt.savefig(path + "/outputs/r2_weighted_dino.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path + "/outputs/loss_dino.png")
