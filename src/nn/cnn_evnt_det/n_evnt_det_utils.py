# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_evnt_det_utils.py
 Description:
 Author:       Joshua Poole
 Created on:   20251028
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

from scipy.io import loadmat
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.nn.cnn_evnt_det.n_evnt_det import SpikeNet


def plot_sample_with_binary(sample, binary_signal):
    """
    This plotting function will plot a time series
    accompanied by the binary decision signal
    """
    x_axis = np.arange(len(sample))

    # Check that both have same length
    if len(sample) != len(binary_signal):
        raise ValueError("`sample` and `binary_signal` must have the same length.")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- Top plot: waveform ---
    ax1.plot(x_axis, sample, color='b', label='Signal')
    ax1.set_title("Sample Waveform and Binary Signal")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Bottom plot: binary signal ---
    ax2.step(x_axis, binary_signal, where='mid', color='r', linewidth=2, label='Binary Signal')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['0', '1'])
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Binary")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show(block=False)

def prep_set_train(data, labels, window_size=200, stride=100):
    """
    Package the series and indexes into tensors with respect to the
    required window size and stride length.

    It has a sliding window (hence strid) that means the output
    tensors will represent 2*input_size - window_size 1D datapoints
    """
    X, y = [], []
    for i in range(0, len(data) - window_size, stride):
        X.append(data[i:i + window_size])
        y.append(labels[i:i + window_size])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, window_size)
    y = torch.tensor(y, dtype=torch.float32)  # (N, window_size)
    print(X.size(), y.size())
    return X, y


def prep_set_inf(data, labels, window_size=200):
    """
    This version id for preparing the inference tensors and will not slide!
    """
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, window_size):
        X.append(data[i:i + window_size])
        y.append(labels[i:i + window_size])

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, window_size)
    y = torch.tensor(y, dtype=torch.float32)  # (N, window_size)
    print(X.size(), y.size())
    return X, y
