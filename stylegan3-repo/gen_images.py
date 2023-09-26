# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

from train import init_dataset_kwargs

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

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

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=parse_range, help='Class label (optional list, unconditional if not specified, e.g., \'0,1,4-6\')')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--genes', help='Gene expression use', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--data', help='Training data', metavar='[ZIP|DIR]', type=str)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[List[int]],
    genes: bool,
    data: str
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        if genes is True and len(class_idx) == 1:
            training_set = import_dataset(genes=genes, data=data, gene_size=G.c_dim)
            label = training_set.get_label(class_idx[0])
            label = torch.from_numpy(label).unsqueeze(0).to(device)
            real_image = training_set[class_idx[0]][0]
        elif genes is True and len(class_idx) > 1:
            training_set = import_dataset(genes=genes, data=data, gene_size=G.c_dim)
            list_of_images = []
            for idx in class_idx:
                label = training_set.get_label(idx)
                label = torch.from_numpy(label).unsqueeze(0).to(device)
                real_image = training_set[idx][0]
                list_of_images.append([real_image, label])
        else:
            label[:, class_idx[0]] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    if genes is True:
        # grid = np.empty((1, 256, 256 * len(seeds), 3))
        print(list_of_images[0][0].shape)
        w, h = list_of_images[0][0].shape
        gw, gh = len(seeds), len(class_idx)
        canvas = PIL.Image.new('RGB', (h * gh, w * gw), 'white')
        list_of_PIL_images = []            
        for real_image, label in list_of_images:            
            # real_img = np.expand_dims(real_image.transpose(1, 2, 0), axis=0)
            real_img = real_image.transpose(1, 2, 0)
            list_of_PIL_images.append(PIL.Image.fromarray(real_img, 'RGB'))
            # combined_img = real_img
            # Generate images.
            for seed_idx, seed in enumerate(seeds):
                print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

                # Construct an inverse rotation/translation matrix and pass to the generator.  The
                # generator expects this matrix as an inverse to avoid potentially failing numerical
                # operations in the network.
                if hasattr(G.synthesis, 'input'):
                    m = make_transform(translate, rotate)
                    m = np.linalg.inv(m)
                    G.synthesis.input.transform.copy_(torch.from_numpy(m))

                # if len(seeds) > 1 and seed_idx != len(seeds) - 1:
                gen_img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                gen_img = (gen_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                gen_img = gen_img.cpu().numpy()
                # combined_img = np.concatenate((combined_img, gen_img), axis=2)
                list_of_PIL_images.append(PIL.Image.fromarray(gen_img[0], 'RGB'))
                # continue
                # elif len(seeds) == 1:
                # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                # combined_img = np.concatenate((real_img, img.cpu().numpy()), axis=2)
                # list_of_PIL_images.append(PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB'))


            
            # grid = np.concatenate((grid, combined_img), axis=1)

        # grid = grid[:, 256:, :, :]
        # PIL.Image.fromarray(grid[0], 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        for idx, img in enumerate(list_of_PIL_images):
            x = idx % gw
            y = idx // gw
            canvas.paste(img, (x * w, y * h))
        canvas.save(f'{outdir}_grid.png')
        # PIL.Image.fromarray(grid[0], 'RGB').save(f'{outdir}_grid.png')

    else:
        #  Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            # Construct an inverse rotation/translation matrix and pass to the generator.  The
            # generator expects this matrix as an inverse to avoid potentially failing numerical
            # operations in the network.
            if hasattr(G.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))

            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


def import_dataset(genes: bool, data:str, gene_size: int):
    # Training set.
    training_set_kwargs, _dataset_name = init_dataset_kwargs(data=data, is_pickle=genes)
    training_set_kwargs.use_labels = True
    training_set_kwargs.xflip = False
    training_set_kwargs.gene_size = gene_size
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    return training_set

def make_grid(imgs: List[PIL.Image.Image], grid_size: Tuple[int, int]) -> PIL.Image.Image:
    '''Make a grid of images.'''
    w, h = imgs[0].size
    gw, gh = grid_size
    canvas = PIL.Image.new('RGB', (w * gw, h * gh), 'white')
    for idx, img in enumerate(imgs):
        x = idx % gw
        y = idx // gw
        canvas.paste(img, (x * w, y * h))
    return canvas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
