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
from importlib.machinery import SourceFileLoader
import torchmetrics
from torchmetrics import PearsonCorrCoef

import legacy

from train import init_dataset_kwargs
dataset_stnet = SourceFileLoader("dataset_stnet.name", "../data/dataset_stnet.py").load_module()
create_dataloader = dataset_stnet.create_dataloader


# Constants

tsv_path = "/projects/minos/jeremie/data/tsv_concatened_allgenes.pkl"
selection_tensor_path = "/projects/minos/jeremie/data/features_std.pkl"

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
@click.option('--testing', help='Testing data', metavar='BOOL', type=bool, default=False, show_default=True)
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
    data: str,
    testing: bool
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
    with dnnlib.util.open_url(network_pkl) as f:
        print("Loading discriminator...")
        D = legacy.load_network_pkl(f)['D'].to(device) # type: ignore
        print("Discriminator loaded")

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        elif genes is True:
            training_set = import_dataset(genes=genes, data=data, gene_size=G.c_dim, testing=testing)
            list_of_images = []
            for idx in class_idx:
                assert idx < len(training_set), f"Class index {idx} is out of range for dataset of size {len(training_set)}"
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
        _c, w, h = list_of_images[0][0].shape
        gw, gh = len(seeds)+1, len(class_idx)
        print(f"grid shape: width: {gw}, height:{gh}")
        print(f"Number of labels: {len(list_of_images)}")
        print(f"Number of images per label: {len(seeds)}")
        canvas = PIL.Image.new('RGB', (w * gw, h * gh), 'white')
        list_of_PIL_images = [] 
        dict_results = {"probs": [], "correlation": []}           
        for real_image, label in list_of_images:            
            real_img = real_image.transpose(1, 2, 0)
            list_of_PIL_images.append(PIL.Image.fromarray(real_img, 'RGB'))
            # Generate images.
            list_pearson = []
            list_probs = []
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

                gen_img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                logits, regressor = D(gen_img, label)
                pearson = PearsonCorrCoef(num_outputs=1).to(device)
                correlation = pearson(regressor.squeeze(0), label.squeeze(0))
                print(f"clf: {torch.nn.functional.sigmoid(logits).cpu().item()}, correlation: {correlation}")
                list_probs.append(torch.nn.functional.sigmoid(logits).item())
                list_pearson.append(correlation)
                gen_img = (gen_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                gen_img = gen_img.cpu().numpy()
                list_of_PIL_images.append(PIL.Image.fromarray(gen_img[0], 'RGB'))
            
            dict_results["probs"].append(np.mean(list_probs))
            dict_results["correlation"].append(torch.stack(list_pearson).mean())

        for idx, img in enumerate(list_of_PIL_images):
            x = idx % gw
            y = idx // gw
            canvas.paste(img, (x * w, y * h))
        canvas.save(f'{outdir}_grid.png')

        if testing:
            true_labels = torch.zeros(len(list_of_images), dtype=torch.long)
            outputs = torch.tensor((np.array(dict_results["probs"]) > 0.5) * 1)
            print(outputs.shape)
            raise Exception
            accuracy = torchmetrics.functional.accuracy(torch.stack(dict_results["probs"]), true_labels, task="binary")
            print(f"Accuracy: {accuracy}")
            correlation = torch.stack(dict_results["correlation"]).mean()
            print(f"Correlation: {correlation}")

        else:
            true_labels = torch.ones(len(list_of_images), dtype=torch.long)
            accuracy = torchmetrics.functional.accuracy(torch.stack(dict_results["probs"]), true_labels, task="binary")
            print(f"Accuracy: {accuracy}")
            correlation = torch.stack(dict_results["correlation"]).mean()
            print(f"Correlation: {correlation}")


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


def import_dataset(genes: bool, data:str, gene_size: int, testing: bool = False):
    # Training set.
    if testing:
        data += '_patientout'
    training_set_kwargs, _dataset_name = init_dataset_kwargs(data=data, is_pickle=genes)
    training_set_kwargs.use_labels = True
    training_set_kwargs.xflip = False
    training_set_kwargs.gene_size = gene_size
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    return training_set


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
