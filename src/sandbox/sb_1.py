# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        sb1.py
 Description:
 Author:
 Author:       Joshua Poole
 Created on:   20251031
 Version:      1.0
===========================================================

 Notes:
    -

 Requirements:
    - Python >= 3.11
    - scipy
    - numpy
    - matplotlib
    - random

==================
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.io import loadmat
import os
from src.ext.nn_data_plot_app import plot_dataset


os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))
print(os.getcwd())

# optional example plot
data = loadmat('src/nn/seq/20251030_cnn_cnn_seq/outputs/vis/D2_vis.mat')

plot_dataset(data)
print()

