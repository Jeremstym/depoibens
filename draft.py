#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Brouillon

import multiprocessing as mp
import os

if __name__ == '__main__':
    os.chdir('/projects/minos/jeremie/data')
    print(os.getcwd())
    p = mp.Pool(mp.cpu_count())
    print(mp.cpu_count())
