#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Brouillon

import multiprocessing as mp
from multiprocessing import Pool
import os
import torch

if __name__ == '__main__':
    print(torch.cuda.device_count())
