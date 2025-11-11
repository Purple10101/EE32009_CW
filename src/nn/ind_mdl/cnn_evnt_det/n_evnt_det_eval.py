# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        nsig_evnt_det_eval.py
 Description:
 Author:       Joshua Poole
 Created on:   20251111
 Version:      1.0
===========================================================

 Notes:
    - This version requires dimensionality to be maintained
      in the prediction output.

 Requirements:
    - Python >= 3.11
    - scipy
    - numpy
    - matplotlib
    - random

==================
"""

import numpy as np
from scipy.io import loadmat
import os
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.nn.ind_mdl.cnn_evnt_det.n_evnt_det import SpikeNet
from src.nn.ind_mdl.cnn_evnt_det.n_evnt_det_utils_nn import *
from src.nn.ind_mdl.cnn_evnt_det.n_evnt_det_utils_sig import *


dir_name = "EE32009_CW"
p = Path.cwd()
while p.name != dir_name:
    if p.parent == p:
        raise FileNotFoundError(f"Directory '{dir_name}' not found above {Path.cwd()}")
    p = p.parent
os.chdir(p)
print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data1 = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

# raw datasets
raw_data_80 = norm_data(data1['d'][0])
raw_data_60 = norm_data(degrade(data1['d'][0], data2['d'][0],0.25))
raw_data_40 = norm_data(degrade(data1['d'][0], data3['d'][0], 0.4))
raw_data_20 = norm_data(degrade(data1['d'][0], data4['d'][0], 0.6))
raw_data_0 = norm_data(degrade(data1['d'][0], data5['d'][0], 0.8))
raw_data_sub0 = norm_data(degrade(data1['d'][0], data6['d'][0], 1))


# pred truth list
idx_lst = data1['Index'][0]
tr_to_tst_r=0.8

labels_bin = []
for y in range(data1['d'][0].shape[0]):
    if y in idx_lst:
        labels_bin.append(1)
    else:
        labels_bin.append(0)
labels_bin = widen_labels(np.array(labels_bin))
split_index_raw = int(len(data1['d'][0]) * tr_to_tst_r)


# val data
raw_data_val = raw_data_60[split_index_raw:]
idx_bin_val = labels_bin[split_index_raw:]
X_v, y_v, index_map_v = prep_set_val(raw_data_val, idx_bin_val)
dataset_v = TensorDataset(X_v, y_v)
loader_v = DataLoader(dataset_v, batch_size=64, shuffle=False)

# inf data on d2 to visualise
data2_norm = norm_data(data2['d'][0])
X_i, index_map_i = prep_set_inf(data2['d'][0])
dataset_i = TensorDataset(X_i)
loader_i = DataLoader(dataset_i, batch_size=64, shuffle=False)

# load model and evaluate performance
model = SpikeNet().to(device)
model.load_state_dict(torch.load(
    "src/nn/ind_mdl/models/D2/20251111_neuron_event_det_cnn_window_norm_d2.pt"))
model.eval()

# Do forward pass on the _val data using the index map to
# reconstruct the predictions
all_outputs = []
with torch.no_grad():  # disables gradient computation (saves memory)
    for X_batch, _ in loader_v:
        X_batch = X_batch.to(device)
        output = model(X_batch)  # shape: (batch_size, 1, window_size) or (batch_size, window_size)
        output = output.squeeze(1).cpu().numpy()  # shape: (batch_size, window_size)
        all_outputs.append(output)

# Stack all batches back together
outputs_v = np.concatenate(all_outputs, axis=0)  # (num_windows, window_size)
# construct our outputs list
n_total = len(raw_data_val)
final_probs = np.zeros(n_total)
counts = np.zeros(n_total)

for i in range(outputs_v.shape[0]):
    final_probs[index_map_v[i]] += outputs_v[i]
    counts[index_map_v[i]] += 1

# Average overlapping predictions
final_probs /= np.maximum(counts, 1)
preds = nonmax_rejection(final_probs, 0.7)
print(len([x for x in preds if x != 0]))
plot_sample_with_binary(raw_data_val[-11000:], preds[-11000:])

metrics = tolerant_binary_metrics(preds, idx_bin_val, tol=50)
print(f"Accuracy:  {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1:        {metrics['f1']:.3f}")
print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
print(f"Zero agreement: {metrics['zero_agreement']:.3f}")


# Do forward pass on the _inf data using the index map to
# reconstruct the predictions
all_outputs = []
with torch.no_grad():  # disables gradient computation (saves memory)
    for X_batch, in loader_i:
        X_batch = X_batch.to(device)
        output = model(X_batch)  # shape: (batch_size, 1, window_size) or (batch_size, window_size)
        output = output.squeeze(1).cpu().numpy()  # shape: (batch_size, window_size)
        all_outputs.append(output)

# Stack all batches back together
outputs_i = np.concatenate(all_outputs, axis=0)  # (num_windows, window_size)
# construct our outputs list
n_total = len(data2_norm)
final_probs = np.zeros(n_total)
counts = np.zeros(n_total)

for i in range(outputs_i.shape[0]):
    final_probs[index_map_i[i]] += outputs_i[i]
    counts[index_map_i[i]] += 1

# Average overlapping predictions
final_probs /= np.maximum(counts, 1)
preds = nonmax_rejection(final_probs, 0.7)
print(len([x for x in preds if x != 0]))
plot_sample_with_binary(data2_norm[-11000:], preds[-11000:])
print()