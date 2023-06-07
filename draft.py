#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Brouillon

import multiprocessing as mp
from multiprocessing import Pool
import os
import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        print(available_gpus)
    else:
        print("No GPU available")