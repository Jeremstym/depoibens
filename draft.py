#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Brouillon

import multiprocessing as mp
from multiprocessing import Pool
import os
import torch

if __name__ == '__main__':
    torch.cuda.device_count()
    torch.cuda.get_device_name(0)
    torch.cuda.is_available()
    torch.cuda.current_device()
    torch.cuda.device(0)
    torch.cuda.empty_cache()
    torch.cuda.memory_allocated()
    torch.cuda.memory_cached()
    torch.cuda.max_memory_allocated()
    torch.cuda.max_memory_cached()