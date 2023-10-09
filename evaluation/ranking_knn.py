#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Ranking k-NN

import os
import sys
from importlib.machinery import SourceFileLoader

import numpy as np
from PIL import Image
from glob import glob

import torch
from torchvision import transforms

import click
from tqdm import tqdm
import pickle
from typing import List, Optional, Tuple, Union

sys.path.append('/import/bc_users/biocomp/stym/depoibens/stylegan3-repo')
generator = SourceFileLoader("gen_images.name", "../stylegan3-repo/gen_images.py").load_module()
dnnlib = SourceFileLoader("dnnlib", "../stylegan3-repo/dnnlib/__init__.py").load_module()
legacy = SourceFileLoader("legacy.name", "../stylegan3-repo/legacy.py").load_module()
train = SourceFileLoader("train.name", "../stylegan3-repo/train.py").load_module()

init_dataset_kwargs = train.init_dataset_kwargs

# Constants

path_to_images = "/projects/minos/jeremie/data/styleImagesGen"
path_to_reals = "/projects/minos/jeremie/data/"
path_to_fakes = "/projects/minos/jeremie/data/generated_dict.pkl"
path_to_dinodict = "/projects/minos/jeremie/data/label_dino.pkl"
path_to_model = "/projects/minos/jeremie/data/styleGANresults/00078-stylegan2-styleImagesGen-gpus2-batch32-gamma0.2048/network-snapshot-021800.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"
dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
dino.eval()
dino.to(device)


#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_labelized_embeddings(path: str, model=dino, device=device):
    training_set = import_dataset(genes=True, data=path, gene_size=900)
    os.chdir(path)
    dataloader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=False, num_workers=0)
    dict = {}
    with tqdm(dataloader, unit="spot", total=len(dataloader)) as pbar:
        for image, label in pbar:
            image = transforms.ToTensor()(image[0].numpy()) # prevent unit8 type error
            image = image.permute(1, 2, 0)
            image = image.unsqueeze(0).to(device)
            label = label.to(device)
            with torch.no_grad():
                dict[label] = model(image).cpu().numpy()

    return dict

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', type=int, help='Random seed', default=42, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
def rank_gene(
    network_pkl: str,
    seed: int,
    truncation_psi: float,
    noise_mode: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: int,
    data = path_to_images
):
    
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Labels.
    if class_idx is None:
        raise click.UsageError('--class must be specified when running on unconditional networks')
    if G.c_dim == 0:
        raise click.UsageError('--class cannot be specified when running on unconditional networks')

    training_set = import_dataset(genes=True, data=data, gene_size=G.c_dim)
    label = training_set.get_label(class_idx) 
    label = torch.from_numpy(label).unsqueeze(0).to(device)

    # Generate images.
    print('Generating image for seed %d ...' % (seed))
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    gen_img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    gen_img = (gen_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    gen_img = gen_img.cpu().numpy()
    gen_img = transforms.ToTensor()(gen_img[0])
    gen_img = gen_img.unsqueeze(0).to(device)
    embed_img = dino(gen_img)
    embed_img = embed_img.detach()

    # Load dino dict
    with open(path_to_dinodict, "rb") as f:
        dino_dict = pickle.load(f)

    # Get 100 nearest neighbors of generated image
    print("Getting 100 nearest neighbors of generated image...")
    distances = []
    for key in dino_dict.keys():
        distances.append(np.linalg.norm(dino_dict[key] - embed_img.cpu().numpy()))
    distances = np.array(distances)
    idx = np.argsort(distances)[:100].tolist() # dino_dict is ranged in the same order as training_set

    try:
        real_idx = idx.index(idx[class_idx])
        print(f"Position of the real image in the 100 nearest neighbors of generated image: {real_idx}")
    except IndexError:
        print("Real image not in the 100 nearest neighbors of generated image")
        return

    



def import_dataset(genes: bool, data:str, gene_size: int):
    # Training set.
    training_set_kwargs, _dataset_name = init_dataset_kwargs(data=data, is_pickle=genes)
    training_set_kwargs.use_labels = True
    training_set_kwargs.xflip = False
    training_set_kwargs.gene_size = gene_size
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    return training_set

#----------------------------------------------------------------------------

# if __name__ == "__main__":
#     label_dino = create_labelized_embeddings(path=path_to_images)
#     with open(path_to_reals + "/label_dino.pkl", "wb") as f:
#         pickle.dump(label_dino, f)

if __name__ == "__main__":
    rank_gene()

#----------------------------------------------------------------------------