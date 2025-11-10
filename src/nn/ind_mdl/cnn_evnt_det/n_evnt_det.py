# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_evnt_det.py
 Description:
 Author:       Joshua Poole
 Created on:   20251103
 Version:      2.0
===========================================================

Notes:
    - Batch Normalization is implemented in this version.
    - Each convolutional layer output is normalized across the batch dimension.
        - This ensures each feature channel has a stable mean and variance
          before being passed to the next layer.
        - Implemented with nn.BatchNorm1d(...).
    - Dilation is used to expand the convolution’s receptive field exponentially.
        - This allows the network to capture a wider temporal context
          without increasing the number of parameters.


 Requirements:
    - Python >= 3.11
    - scipy
    - numpy
    - matplotlib
    - random

==================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class SpikeNet(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, dilation=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, dilation=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, dilation=8, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
