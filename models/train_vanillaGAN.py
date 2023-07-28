#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle as pkl
import logging

sys.path.append("../")

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from bioval.metrics.conditional_evaluation import ConditionalEvaluation

from vanillaGAN import Generator, Discriminator, weights_init

# from data.dataset_stnet import create_GAN_dataloader
from tools.utils_GAN import (
    save_model_generator,
    save_model_discriminator,
    show_tensor_images,
    show_final_grid
)

### -------------- Constants ------------------

# Path to data

path_to_data = "/projects/minos/jeremie/data/images"
path_to_log = "/projects/minos/jeremie/data/logs"
path_save = "/projects/minos/jeremie/data/GANresults"

## Hyperparameters
nz = 100  # size of latent vector
ngf = 64  # size of feature maps in generator
ndf = 64  # size of feature maps in discriminator
mean = (0.485, 0.456, 0.406) # normalize images
std = (0.229, 0.224, 0.225) # normalize images

BATCH_SIZE = 128  # Batch size during training

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

parser = argparse.ArgumentParser(description="Vanilla GAN for transcriptomics")
parser.add_argument(
    "-ngpu",
    "--ngpu",
    type=int,
    default=1,
    help="Number of GPUs available. Use 0 for CPU mode.",
)
parser.add_argument(
    "-epochs",
    "--epochs",
    type=int,
    default=10,
    help="Number of epochs to train for",
)
args = vars(parser.parse_args())
ngpu = args["ngpu"]
num_epochs = args["epochs"]


def main():
    ### -------------- Load data -------------------------------
    preprocess = transforms.Compose(
        [
            transforms.Resize((300,300)),
            transforms.RandomCrop(300),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = dset.ImageFolder(root=path_to_data, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    print("Data loaded")
    ### -------------- Initialize models -----------------------

    device = torch.device("cuda" if torch.cuda.is_available() and ngpu > 0 else "cpu")

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == "cuda") and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == "cuda") and (ngpu > 1):
        netD = nn.DataParallel(netD, device_ids=list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    ### -------------- Loss function -----------------------
    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    ### -------------- Training -----------------------

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    real_batch = next(iter(dataloader))

    # Sample data
    # fake = torch.randn(64, 3, 300, 300)
    noise = torch.randn(64, nz, 1, 1)
    fake = netG(noise).detach().cpu()
    # Test plot_grid function for fake image
    show_tensor_images(fake, path_save, 0)
    # Test plot_grid function for real image
    show_tensor_images(real_batch[0], path_save, -1)


    print("Starting Training Loop...")
    # For each epoch

    for epoch in range(num_epochs):
        with tqdm(enumerate(dataloader), total=len(dataloader), unit="batch") as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
            # For each batch in the dataloader
            for i, data in pbar:
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full(
                    (b_size,), real_label, dtype=torch.float, device=device
                )
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # metrics bioval
                topk = ConditionalEvaluation()
                score = topk(fake.detach(), real_cpu.detach())

                # Output training stats
                if i % 50 == 0:
                    pbar.set_postfix(
                        loss_D=errD.item(),
                        loss_G=errG.item(),
                        D_x=D_x,
                        D_G_z1=D_G_z1,
                        D_G_z2=D_G_z2,
                    )
                    logger = logging.getLogger("score")
                    logger.setLevel(logging.INFO)
                    formatter = logging.Formatter(
                        "%(asctime)s | %(levelname)s | %(message)s"
                    )
                    file_handler = logging.FileHandler(
                        os.path.join(path_save, "score.log")
                    )
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                    logger.info(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f, score: %.4f",
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                        score,
                    )

                    # print(
                    #     "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    #     % (
                    #         epoch,
                    #         num_epochs,
                    #         i,
                    #         len(dataloader),
                    #         errD.item(),
                    #         errG.item(),
                    #         D_x,
                    #         D_G_z1,
                    #         D_G_z2,
                    #     )
                    # )

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or (
                    (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    show_tensor_images(fake, path_save, epoch+1)

                iters += 1

            # Save fake images
            # plot_grid(fake, path_save, epoch)
            # show_tensor_images(fake, path_save, epoch+1)

    ### -------------- Save models -----------------------

    # Save models
    save_model_generator(path_save, num_epochs, netG, optimizerG, criterion)
    save_model_discriminator(path_save, num_epochs, netD, optimizerD, criterion)

    ### -------------- Plot -----------------------

    # plot_final_grid(real_batch, img_list, path_save)
    show_final_grid(real_batch, img_list, path_save)
    print("Done!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        filename=path_to_log + "/training.log",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        main()
    except KeyboardInterrupt as keyb:
        logging.exception(str(keyb))
    except Exception as e:
        logging.exception(str(e))

# if __name__ == "__main__":
#     # main()
#     dataloader = create_GAN_dataloader(image_path=path_to_data, train_batch_size=16)
#     with open("dataloaderGAN.pkl", "wb") as f:
#         pkl.dump(dataloader, f)
