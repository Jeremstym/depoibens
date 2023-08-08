#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Brouillon

import multiprocessing as mp
from multiprocessing import Pool
import os
import torch


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