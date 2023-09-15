#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Brouillon

import multiprocessing as mp
from multiprocessing import Pool
import os
import torch

import pandas as pd


t = torch.tensor([1,2,3]).to(torch.float32)
t2 = torch.tensor([[1,1,1],[2,2,2],[3,3,3]]).to(torch.float32)

torch.norm(t2, keepdim=True, dim=0)
torch.norm(t2, keepdim=True)
torch.norm(t2, dim=0)

torch.sqrt(torch.tensor([3]))

t.norm(float('inf')) # norme infinie
t.square().sum().sqrt() # norme euclidienne
t.square().sum().rsqrt() # norme euclidienne inverse

t2.unbind(1)

di_inception = {
        "lr": 1e-4,
        "bacth_size": 256,
        "test_bacth_size": 16,
        "epochs": 200,
        "dropout": 0.6,
        "input_size": 900,
        "hidden_size": 2048,
        "output_size": 2048,
    }

di_dino = {
        "lr": 1e-4,
        "bacth_size": 256,
        "test_bacth_size": 16,
        "epochs": 200,
        "dropout": 0.6,
        "input_size": 900,
        "hidden_size": 1536,
        "output_size": 768,
    }


print(pd.DataFrame(di_inception, index=[0]).T.to_latex())
print(pd.DataFrame(di_dino, index=[0]).T.to_latex())
