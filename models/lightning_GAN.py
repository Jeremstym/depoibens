#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Lightning GAN for transcriptomics

import os 
import sys
import pickle as pkl
import argparse
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
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

import pytorch_lightning as pl

path_save = "/projects/minos/jeremie/data/GANresults"



### ----------- Networks ----------------------------

class GAN(pl.LightningModule):
    def __init__(self, nz, ngf, ndf, lr, b1, b2):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.criterion = nn.BCELoss()

        self.example_input_array = torch.randn(1, 100, 1, 1)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    # def training_step(self, batch, batch_idx, optimizer_idx):
    #     real, _ = batch
    #     real = real.to(self.device)
    #     batch_size = real.size(0)
    #     label = torch.full((batch_size,), 1, device=self.device)
    #     fake_label = torch.full((batch_size,), 0, device=self.device)

    #     # Train Generator
    #     if optimizer_idx == 0:
    #         noise = torch.randn(batch_size, self.hparams["nz"], 1, 1, device=self.device)
    #         fake = self.generator(noise)
    #         output = self.discriminator(fake)
    #         loss = self.adversarial_loss(output, label)
    #         self.log("g_loss", loss)
    #         return loss

    #     # Train Discriminator
    #     if optimizer_idx == 1:
    #         noise = torch.randn(batch_size, self.hparams["nz"], 1, 1, device=self.device)
    #         fake = self.generator(noise).detach()
    #         output_real = self.discriminator(real)
    #         output_fake = self.discriminator(fake)
    #         loss_real = self.adversarial_loss(output_real, label)
    #         loss_fake = self.adversarial_loss(output_fake, fake_label)
    #         loss = (loss_real + loss_fake) / 2
    #         self.log("d_loss", loss)
    #         return loss

    # def configure_optimizers(self):
    #     lr = self.hparams["lr"]
    #     b1 = self.hparams["b1"]
    #     b2 = self.hparams["b2"]

    #     opt_g = optim.Adam(
    #         self.generator.parameters(), lr=lr, betas=(b1, b2)
    #     )
    #     opt_d = optim.Adam(
    #         self.discriminator.parameters(), lr=lr, betas=(b1, b2)
    #     )   
    #     return [opt_g, opt_d], []
    
    # def on_epoch_end(self):
    #     z = torch.randn(64, self.hparams["nz"], 1, 1, device=self.device)
    #     sample = self(z)
    #     grid = vutils.make_grid(sample, normalize=True)
    #     self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
    #     # show_tensor_images(sample)
    #     # show_final_grid(sample, self.current_epoch)
    #     # save_model_generator(self.generator, self.current_epoch)
    #     # save_model_discriminator(self.discriminator, self.current_epoch)
    #     # self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

    # def train_dataloader(self):
    #     return self.hparams["train_dataloader"]

    def training_step(self, batch):
        imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.nz, 1, 1)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        # sample_imgs = self.generated_imgs[:6]
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image("generated_images", grid, 0)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        # grid = torchvision.utils.make_grid(sample_imgs)
        show_tensor_images(sample_imgs, path_save, self.current_epoch)
        # self.logger.experiment.add_image("generated_images", grid, self.current_epoch)