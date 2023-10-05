#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Personal FID

from bioval.metrics.conditional_evaluation import ConditionalEvaluation

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import pickle
from tqdm import tqdm
from glob import glob 

# Constants

path_to_reals = "/projects/minos/jeremie/data/"
path_to_fakes = "/projects/minos/jeremie/data/generated_dict.pkl"


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
    image_transposed = image_processed.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    image_batch = image_transposed.unsqueeze(0)
    assert image_batch.shape == (1, 256, 256, 3)
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
    assert reals.shape == (len(reals), 256, 256, 3)
    return reals


def preprocess_all_fakes(path_to_fakes: str) -> torch.Tensor:
    # Preprocess all f ake images
    print("Loading fakes...")
    with open(path_to_fakes, "rb") as f:
        fakes = pickle.load(f)
    fakes = list(fakes.values()) # list of tensors
    fakes = torch.cat(fakes)
    assert fakes.shape == (len(fakes), 256, 256, 3)
    return fakes


def split_on_channels(concatenate_image: torch.Tensor) -> torch.Tensor:
    assert concatenate_image.shape == (len(concatenate_image), 256, 256, 3)
    # Split on channels
    splited_images = concatenate_image.split(1, dim=3)
    assert len(splited_images) == 3
    # Remove the channel dimension
    # splited_images = [image.squeeze(3) for image in splited_images]
    # assert splited_images[0].shape == (len(concatenate_image), 256, 256)
    return splited_images


def main():
    topk = ConditionalEvaluation(distributed_method="fid")
    reals = preprocess_all_reals(path_to_reals)
    fakes = preprocess_all_fakes(path_to_fakes)
    print("Computing FID...")
    main_results = topk(reals, fakes, aggregated=False)
    ch1_reals, ch2_reals, ch3_reals = split_on_channels(reals)
    ch1_fakes, ch2_fakes, ch3_fakes = split_on_channels(fakes)
    print("Computing channel 1 FID...")
    ch1_results = topk(ch1_reals, ch1_fakes, aggregated=False)
    print("Computing channel 2 FID...")
    ch2_results = topk(ch2_reals, ch2_fakes, aggregated=False)
    print("Computing channel 3 FID...")
    ch3_results = topk(ch3_reals, ch3_fakes, aggregated=False)
    mean_results = torch.mean([ch1_results, ch2_results, ch3_results], dim=0)
    print("Main results:", main_results)
    print("Channel 1 results:", ch1_results)
    print("Channel 2 results:", ch2_results)
    print("Channel 3 results:", ch3_results)
    print("Mean results:", mean_results)
    return main_results, ch1_results, ch2_results, ch3_results, mean_results


if __name__ == "__main__":
    main_results, ch1_results, ch2_results, ch3_results, mean_results = main()
    # with open("results.pkl", "wb") as f:
    #     pickle.dump([main_results, ch1_results, ch2_results, ch3_results], f)
