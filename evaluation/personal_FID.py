#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Personal FID

from bioval.metrics.conditional_evaluation import ConditionalEvaluation

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models.inception import Inception_V3_Weights
from piq import FID

import pickle
from tqdm import tqdm
from glob import glob 

# Constants

path_to_reals = "/projects/minos/jeremie/data/"
path_to_fakes = "/projects/minos/jeremie/data/generated_dict.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare model
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

inception = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
# inception.Conv2d_1a_3x3.conv=nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False) # change input channels to 1
inception.fc = Identity() # remove fully connected layer, output size = 2048
inception.eval()
inception.to(device)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    # Preprocess image
    size = 256
    preprocess = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # default values for imagenet
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    image_processed = preprocess(image)
    # image_transposed = image_processed.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    image_batch = image_processed.unsqueeze(0)
    assert image_batch.shape == (1, 3, 256, 256)
    return image_batch


def preprocess_all_reals(path_to_reals: str) -> torch.Tensor:
    # Preprocess all real images
    reals = []
    os.chdir(path_to_reals)
    with tqdm(glob("images/*/*.jpg"), unit="spot") as pbar:
        for image_path in pbar:
            image = Image.open(image_path)
            image_batch = preprocess_image(image)
            reals.append(image_batch)
    reals = torch.cat(reals)
    print(f"reals.shape: {reals.shape}")
    assert reals.shape == (len(reals), 3, 256, 256)
    return reals


def preprocess_all_fakes(path_to_fakes: str) -> torch.Tensor:
    # Preprocess all f ake images
    print("Loading fakes...")
    with open(path_to_fakes, "rb") as f:
        fakes = pickle.load(f)
    fakes = list(fakes.values()) # list of tensors
    # fakes_transposed = [fake.permute(1, 2, 0) for fake in fakes] # (C, H, W) -> (H, W, C)
    fake_stacked = torch.stack(fakes) # stack tensors to create batch
    print(f"fake_stacked.shape: {fake_stacked.shape}")
    assert fake_stacked.shape == (len(fake_stacked), 3, 256, 256)
    return fake_stacked


def split_on_channels(concatenate_image: torch.Tensor) -> torch.Tensor:
    assert concatenate_image.shape == (len(concatenate_image), 3, 256, 256)
    # Split on channels
    splited_images = concatenate_image.split(1, dim=1)
    assert len(splited_images) == 3
    # Remove the channel dimension
    # splited_images = [image.squeeze(3) for image in splited_images]
    # assert splited_images[0].shape == (len(concatenate_image), 256, 256)
    # Create artificial 3 channels
    return splited_images

def embed_images(imageset: torch.Tensor) -> torch.Tensor:
    # Embed images
    print("Embedding images...")
    with torch.no_grad():
        embeddings = []
        with tqdm(imageset, unit="spot") as pbar:
            for image in pbar:
                image = image.to(device)
                embedding = inception(image)
                embeddings.append(embedding)
    embeddings = torch.cat(embeddings)
    print(f"embeddings.shape: {embeddings.shape}")
    assert embeddings.shape == (len(imageset), 2048)
    return embeddings
    


def main():
    # topk = ConditionalEvaluation(distributed_method="fid")
    reals = preprocess_all_reals(path_to_reals)
    fakes = preprocess_all_fakes(path_to_fakes)
    print("Computing FID...")
    # main_results = topk(reals, fakes, aggregated=False)
    ch1_reals, ch2_reals, ch3_reals = split_on_channels(reals)
    ch1_fakes, ch2_fakes, ch3_fakes = split_on_channels(fakes)
    print("Embedding reals...")
    ch1_reals = embed_images(ch1_reals.repeat(1,3,1,1))
    ch2_reals = embed_images(ch2_reals.repeat(1,3,1,1))
    ch3_reals = embed_images(ch3_reals.repeat(1,3,1,1))
    print("Embedding fakes...")
    ch1_fakes = embed_images(ch1_fakes.repeat(1,3,1,1))
    ch2_fakes = embed_images(ch2_fakes.repeat(1,3,1,1))
    ch3_fakes = embed_images(ch3_fakes.repeat(1,3,1,1))
    print("Computing main FID...")
    main_results = FID()(reals, fakes)
    print("Computing channel 1 FID...")
    ch1_results = FID()(ch1_reals, ch1_fakes)
    print("Computing channel 2 FID...")
    ch2_results = FID()(ch2_reals, ch2_fakes)
    print("Computing channel 3 FID...")
    ch3_results = FID()(ch3_reals, ch3_fakes)
    mean_results = torch.mean([ch1_results, ch2_results, ch3_results], dim=0)
    print("Main results:", main_results)
    print("Channel 1 results:", ch1_results)
    print("Channel 2 results:", ch2_results)
    print("Channel 3 results:", ch3_results)
    print("Mean results:", mean_results)
    return main_results, ch1_results, ch2_results, ch3_results, mean_results
    # print("Computing channel 1 FID...")
    # ch1_results = topk(ch1_reals, ch1_fakes, aggregated=False)
    # print("Computing channel 2 FID...")
    # ch2_results = topk(ch2_reals, ch2_fakes, aggregated=False)
    # print("Computing channel 3 FID...")
    # ch3_results = topk(ch3_reals, ch3_fakes, aggregated=False)
    # mean_results = torch.mean([ch1_results, ch2_results, ch3_results], dim=0)
    # print("Main results:", main_results)
    # print("Channel 1 results:", ch1_results)
    # print("Channel 2 results:", ch2_results)
    # print("Channel 3 results:", ch3_results)
    # print("Mean results:", mean_results)
    # return main_results, ch1_results, ch2_results, ch3_results, mean_results


if __name__ == "__main__":
    main_results, ch1_results, ch2_results, ch3_results, mean_results = main()
    with open("results_FID.pkl", "wb") as f:
        pickle.dump([main_results, ch1_results, ch2_results, ch3_results], f)
