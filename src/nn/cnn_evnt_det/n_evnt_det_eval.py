# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_evnt_det_eval.py
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
from src.nn.cnn_evnt_det.n_evnt_det_utils import plot_sample_with_binary, prep_set_inf

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

raw_data_test = raw_data[split_index_raw:]
idx_bin_test = labels_bin[split_index_raw:]
print(len(idx_bin_test))

#
sample_dataset_raw_data = raw_data_test[-20000: -500]
sample_dataset_idx_bin = idx_bin_test[-20000: -500]

# validation data
X_tensor_val, y_tensor_val = prep_set_inf(raw_data_test, idx_bin_test)
dataset_val = TensorDataset(X_tensor_val, y_tensor_val)
loader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)
# sample data for visualisation
X_sample = torch.tensor(sample_dataset_raw_data, dtype=torch.float32).unsqueeze(1)
y_sample = torch.tensor(sample_dataset_idx_bin, dtype=torch.float32)

# plot sample data
plot_sample_with_binary(sample_dataset_raw_data, sample_dataset_idx_bin)

print(X_tensor_val.shape, y_tensor_val.shape)

# load model and evaluate performance
model = SpikeNet()
model.load_state_dict(torch.load("src/nn/models/20251028_neuron_event_det_cnn.pt"))
model.eval()

scorecard = []
infered_events = []
known_events = []

prediction_count = 0

model.eval()
with torch.no_grad():
    incorrect_count = 0
    for X_batch, y_batch in loader_val:
        outputs = model(X_batch)
        preds = (outputs > 0.5).float()
        infered_events.extend(preds.flatten().tolist())
        known_events.extend(y_batch.flatten().tolist())
        prediction_count += preds.size(0)*preds.size(1)
        print()

# see how we did % wise, bear in mind we are currently at 80dB
y_true = np.array(infered_events)
y_pred = np.array(known_events)
num_wrong = sum(int(a) ^ int(b) for a, b in zip(y_true, y_pred))
print(100*(1-(num_wrong/len(y_true))))

# create the list of spike idx
output_indexes = []
for idx, val in enumerate(infered_events):
    if val == 1:
        output_indexes.append(idx)

with torch.no_grad():
    X_sample = X_sample.unsqueeze(0)
    X_sample = X_sample.permute(0, 2, 1)
    outputs = model(X_sample)
    preds = (outputs > 0.5).float()

plot_sample_with_binary(sample_dataset_raw_data, preds.squeeze().tolist())

print()