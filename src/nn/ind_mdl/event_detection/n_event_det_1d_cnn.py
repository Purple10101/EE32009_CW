# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_event_det_1d_cnn.py
 Description:
 Author:       Joshua Poole
 Created on:   20251113
 Version:      1.0
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
    - torch.nn

==================
"""
import torch.nn as nn

class SpikeNet(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, dilation=1, padding=3),
            nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, dilation=2, padding=4),
            nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, dilation=4, padding=4),
            nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, dilation=8, padding=8),
            nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)