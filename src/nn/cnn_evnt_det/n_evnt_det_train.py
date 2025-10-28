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

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(os.getcwd())

data = loadmat('data\D1.mat')

def plot_sample_with_binary(sample, binary_signal):
    """
    Plot a sample waveform with a corresponding binary signal.

    Args:
        sample (array-like): The main signal (e.g., raw data).
        binary_signal (array-like): Binary 0/1 sequence of the same length as `sample`.
    """
    x_axis = np.arange(len(sample))

    # Check that both have same length
    if len(sample) != len(binary_signal):
        raise ValueError("`sample` and `binary_signal` must have the same length.")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- Top plot: waveform ---
    ax1.plot(x_axis, sample, color='b', label='Signal')
    ax1.set_title("Sample Waveform and Binary Signal")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Bottom plot: binary signal ---
    ax2.step(x_axis, binary_signal, where='mid', color='r', linewidth=2, label='Binary Signal')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['0', '1'])
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Binary")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show(block=False)

def prep_set(data, labels, window_size=200, stride=100):
    X, y = [], []
    for i in range(0, len(data) - window_size, stride):
        X.append(data[i:i + window_size])
        y.append(labels[i:i + window_size])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, window_size)
    y = torch.tensor(y, dtype=torch.float32)  # (N, window_size)
    print(X.size(), y.size())
    return X, y


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

raw_data_train = raw_data[:split_index_raw]
idx_bin_train = labels_bin[:split_index_raw]

raw_data_test = raw_data[split_index_raw:]
idx_bin_test = labels_bin[split_index_raw:]

#
sample_dataset_raw_data = raw_data_train[-700: -500]
sample_dataset_idx_bin = idx_bin_train[-700: -500]

# training data
X_tensor, y_tensor = prep_set(raw_data_train, idx_bin_train)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
# validation data
X_tensor_val, y_tensor_val = prep_set(raw_data_test, idx_bin_test)
dataset_val = TensorDataset(X_tensor_val, y_tensor_val)
loader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)
# sample data for visualisation
X_sample = torch.tensor(sample_dataset_raw_data, dtype=torch.float32).unsqueeze(1)
y_sample = torch.tensor(sample_dataset_idx_bin, dtype=torch.float32)

# plot sample data
plot_sample_with_binary(sample_dataset_raw_data, sample_dataset_idx_bin)

print(X_tensor_val.shape, y_tensor_val.shape)

# model and training

"""model = SpikeNet()
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

torch.save(model.state_dict(), "src/nn/models/20251028_neuron_event_det_cnn.pt")"""

# separate training and inference, make n_cls_inf.py

# load model and evaluate performance
model = SpikeNet()
model.load_state_dict(torch.load("src/nn/models/20251028_neuron_event_det_cnn.pt"))
model.eval()

scorecard = []
output_events = []

model.eval()
with torch.no_grad():
    incorrect_count = 0
    for X_batch, y_batch in loader_val:
        outputs = model(X_batch)
        threshold = 0.3
        preds = (outputs > 0.5).float()
        incorrect_count += (preds.bool() ^ y_batch.bool()).sum().item()
print(incorrect_count/len(loader_val))


y_true_list = []
y_pred_list = []

with torch.no_grad():
    X_sample = X_sample.unsqueeze(0)
    X_sample = X_sample.permute(0, 2, 1)
    outputs = model(X_sample)
    preds = (outputs > 0.5).float()

plot_sample_with_binary(sample_dataset_raw_data, preds.squeeze().tolist())

print()