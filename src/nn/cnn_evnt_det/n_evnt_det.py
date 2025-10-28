# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_evnt_det.py
 Description:
 Author:       Joshua Poole
 Created on:   20251027
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

import torch
import torch.nn as nn


class SpikeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, 5, padding=2),  # keep 1 output channel
            nn.Sigmoid()                     # output probability per sample
        )

    def forward(self, x):
        # x: (batch, 1, window_size)
        out = self.net(x)  # (batch, 1, window_size)
        return out.squeeze(1)  # -> (batch, window_size)
