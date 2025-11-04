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
from src.nn.cnn_evnt_det.n_evnt_det_utils import plot_sample_with_binary, prep_set_val, norm_data

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = loadmat('data\D1.mat')
data_d2 = loadmat('data\D2.mat')
data_d3 = loadmat('data\D3.mat')
data_d4 = loadmat('data\D4.mat')
data_d5 = loadmat('data\D5.mat')
data_d6 = loadmat('data\D6.mat')

inf_datasets = [data_d2, data_d3, data_d4, data_d5, data_d6]


raw_data = data['d'][0]
raw_data = norm_data(raw_data)
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

# val data
raw_data_val = raw_data[split_index_raw:]
idx_bin_val = labels_bin[split_index_raw:]
X_v, y_v = prep_set_val(raw_data_val, idx_bin_val)
dataset_v = TensorDataset(X_v, y_v)
loader_v = DataLoader(dataset_v, batch_size=64, shuffle=True)

# plotting sample data
sample_dataset_raw_data = raw_data_val
sample_dataset_idx_bin = idx_bin_val
X_sample = torch.tensor(sample_dataset_raw_data, dtype=torch.float32).unsqueeze(1)
y_sample = torch.tensor(sample_dataset_idx_bin, dtype=torch.float32)

# load model and evaluate performance
model = SpikeNet().to(device)
model.load_state_dict(torch.load("src/nn/models/20251103_neuron_event_det_cnn_dilation.pt"))
model.eval()

with torch.no_grad():
    X_sample = X_sample.unsqueeze(0)
    X_sample = X_sample.permute(0, 2, 1)
    outputs = model(X_sample.to(device))
    preds = (outputs > 0.75).float()
    print(len([x for x in preds.squeeze().tolist() if x != 0]))


plot_sample_with_binary(raw_data_test[-11000:-9000], preds.squeeze().tolist()[-11000:-9000])

for data in inf_datasets:
    pred_count = 0
    raw_data = data['d'][0]
    raw_data = norm_data(raw_data)
    X_sample = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(1)
    X_sample = X_sample.unsqueeze(0)
    X_sample = X_sample.permute(0, 2, 1)
    outputs = model(X_sample.to(device))
    preds = (outputs > 0.71).float().squeeze().tolist()
    print(len([x for x in preds if x != 0]))
    plot_sample_with_binary(raw_data[-20000:-10000], preds[-20000:-10000])




print()