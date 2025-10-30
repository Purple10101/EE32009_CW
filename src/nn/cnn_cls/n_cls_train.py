# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls_train.py
 Description:
 Author:       Joshua Poole
 Created on:   20251024
 Version:      1.1
===========================================================

 Notes:
    - So things that could be hampering the performance:
        - Normalisation?
            - Right now you norm each capture independently.
              This could be an issue!
        - More variations of the degraded training set?
        - Shuffling the training set?
        - Criterion and optimiser?
        - Could you randomly deteriorate some samples?
          so that its being fed one at 80 then one at 0 then one at 60

        - !!!! Norm the whole thing. Different neurons have different amplitudes
          you might actually be stupid for not picking up on that bro icl

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.ext.data_loader_cls import RecordingTrain, plot_sample
from src.nn.cnn_cls.n_cls import NeuronCNN
from src.nn.cnn_cls.n_cls_utils import noise_plt_example, prep_training_set


os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(os.getcwd())

# data prep
print(os.getcwd())
data = loadmat('data\D1.mat')
data_inf = loadmat('data\D2.mat')

rec = RecordingTrain(data['d'], data['Index'], data['Class'])

snr_lst = [80] # add back 0 and -10

# show example of manual degradation
#noise_plt_example(rec, snr_lst)

# now create test sets for each SNR
training_sets = []
for snr in snr_lst:
    training_sets.append(prep_training_set(rec, snr))

# model and training
model = NeuronCNN(5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    total_batches = 0

    for dataloader in training_sets:
        for X_batch, y_batch in dataloader:
            y_batch -= 1
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

    avg_loss = epoch_loss / total_batches
    print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

torch.save(model.state_dict(), "src/nn/models/20251030_neuron_total_norm.pt")