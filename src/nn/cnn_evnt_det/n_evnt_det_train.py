# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_evnt_det_train.py
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
from src.nn.cnn_evnt_det.n_evnt_det_utils import plot_sample_with_binary, prep_set_train


os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(os.getcwd())

data = loadmat('data\D1.mat')

raw_data = data['d'][0]
idx_lst = data['Index'][0]
tr_to_tst_r=0.8

labels_bin = []
for y in range(raw_data.shape[0]):
    if y in idx_lst:
        labels_bin.append(1)
    else:
        labels_bin.append(0)

split_index_raw = int(len(raw_data) * tr_to_tst_r)

raw_data_train = raw_data
idx_bin_train = labels_bin

# training data
X_tensor, y_tensor = prep_set_train(raw_data_train, idx_bin_train)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# model and training
model = SpikeNet()
criterion = nn.BCELoss()  # binary cross entropy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch, y_batch
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)  # accumulate total loss

    avg_loss = train_loss / len(loader.dataset)  # compute average loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.8f}")

torch.save(model.state_dict(), "src/nn/models/20251028_neuron_event_det_cnn.pt")
