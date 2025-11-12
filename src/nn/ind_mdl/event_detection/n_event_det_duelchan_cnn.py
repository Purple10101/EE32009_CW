# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_event_det_duelchan_cnn.py
 Description:
 Author:       Joshua Poole
 Created on:   20251112
 Version:      1.0
===========================================================

 Notes:
    - Duel Channel cnn that takes the raw window on one chan
      and the corresponding spectral power on the other
    - Maybe you run this model in tandem with a more typical
      peak detector 1D CNN that over guesses
      Use this to confirm if that 1D CNNs output is legit?

 Requirements:
    - torch.nn

==================
"""

import torch.nn as nn
import torch.nn.functional as F

class DualInputCNN(nn.Module):
    # This CNN will say if a window contains a spike.
    # To get its index we can use signal processing.
    def __init__(self, input_length, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * (input_length // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 2, input_length)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x