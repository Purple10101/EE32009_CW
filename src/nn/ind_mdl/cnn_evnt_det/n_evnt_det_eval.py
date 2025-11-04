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

from src.nn.ind_mdl.cnn_evnt_det.n_evnt_det import SpikeNet
from src.nn.ind_mdl.cnn_evnt_det.n_evnt_det_utils import (plot_sample_with_binary,
                                                          prep_set_val, degrade,
                                                          norm_data, nonmax_rejection,
                                                          denoise_fft)

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

d2_denoise_norm = denoise_fft(norm_data(data2['d'][0]))
d2_denoise_norm = np.ascontiguousarray(d2_denoise_norm)

# raw datasets
raw_data_80 = norm_data(data['d'][0])
denoise_data_80 = denoise_fft(raw_data_80)

raw_data_60 = norm_data(degrade(data['d'][0], data2['d'][0], 0.25))
denoise_data_60 = denoise_fft(raw_data_60)

raw_data_40 = norm_data(degrade(data['d'][0], data3['d'][0], 0.4))
raw_data_20 = norm_data(degrade(data['d'][0], data4['d'][0], 0.6))
raw_data_0 = norm_data(degrade(data['d'][0], data5['d'][0], 0.8))
raw_data_sub0 = norm_data(degrade(data['d'][0], data6['d'][0], 1))

# pred truth list
idx_lst = data['Index'][0]
tr_to_tst_r=0.8

labels_bin = []
for y in range(raw_data_80.shape[0]):
    if y in idx_lst:
        labels_bin.append(1)
    else:
        labels_bin.append(0)

split_index_raw = int(len(raw_data_80) * tr_to_tst_r)

raw_data_test = raw_data_60[split_index_raw:]
idx_bin_test = labels_bin[split_index_raw:]

# val data
raw_data_val = raw_data_60[split_index_raw:]
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
model.load_state_dict(torch.load(
    "src/nn/ind_mdl/models/D2/20251104_neuron_event_det_cnn_D2.pt"))
model.eval()

with torch.no_grad():
    X_sample = X_sample.unsqueeze(0)
    X_sample = X_sample.permute(0, 2, 1)
    outputs = model(X_sample.to(device))
    preds = nonmax_rejection(outputs.squeeze().tolist(), 0.7)
    print(len([x for x in preds if x != 0]))


#plot_sample_with_binary(raw_data_test[-11000:-9000], preds.squeeze().tolist()[-11000:-9000])

pred_count = 0
X_sample = torch.tensor(d2_denoise_norm, dtype=torch.float32).unsqueeze(1)
X_sample = X_sample.unsqueeze(0)
X_sample = X_sample.permute(0, 2, 1)
outputs = model(X_sample.to(device))
preds = nonmax_rejection(outputs.squeeze().tolist(), 0.6)
print(len([x for x in preds if x != 0]))
plot_sample_with_binary(data2['d'][0][-12000:-10000], preds[-12000:-10000])
#plot_sample_with_binary(raw_data_60[-20000:-10000], labels_bin[-20000:-10000])
print()