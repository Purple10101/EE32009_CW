# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls_utils.py
 Description:
 Author:       Joshua Poole
 Created on:   20251029
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

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.ext.data_loader_cls import Recording, plot_sample


def noise_plt_example(rec, snr_out):
    # plot the 0th capture with noise injection
    for snr in snr_out:
        noisy_un_norm = rec.noise_injection(rec.captures_training, snr)
        noisy_norm = rec.norm_data(noisy_un_norm)
        plot_sample(noisy_norm[0])

def prep_training_set(rec, snr_out=80):
    if snr_out == 80:
        captures = [d["Capture"] for d in rec.captures_train]
        cls = [d["Classification"] for d in rec.captures_train]
        # for example plot some normed time series
        for i in range(80):
            capture = rec.captures_train[i]
            if capture["Classification"] == 3:
                plot_sample(capture)
    else:
        noisy_un_norm = rec.noise_injection(rec.captures_train, snr_out)
        noisy_norm = rec.norm_data(noisy_un_norm)
        captures = [d["Capture"] for d in noisy_norm]
        cls = [d["Classification"] for d in noisy_norm]

    X = np.array(captures, dtype=np.float32)
    y = np.array(cls, dtype=np.int64)
    X = np.expand_dims(X, axis=1)
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=True)