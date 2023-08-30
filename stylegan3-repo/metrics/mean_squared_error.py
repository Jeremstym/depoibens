#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mean squared error (MSE) metric for evaluating generative models."""

import numpy as np
import scipy.linalg
from . import metric_utils

# ----------------------------------------------------------------------------

def compute_mse(opts):
    """Evaluate MSE for the discriminator of the latest snapshot.

    Returns:
        MSE.
    """
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    # mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
    #     opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
    #     rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=None).get_mean_cov()

    # mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
    #     opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
    #     rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=None).get_mean_cov()

    gene_expr = metric_utils.compute_gene_expr_for_dataset(opts=opts)
    gen_expr = metric_utils.compute_gene_expr_for_discriminator(opts=opts)


    if opts.rank != 0:
        return float('nan')

    mse = np.mean(np.square(gen_expr - gene_expr))
    return float(mse)