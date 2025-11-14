# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_event_det_train.py
 Description:
 Author:       Joshua Poole
 Created on:   20251113
 Version:      1.0
===========================================================

 Notes:
    -

 Requirements:
    - torch
    - TensorDataset, DataLoader
    - src.nn.ind_mdl.noise_suppression.noise_suppression_utils

==================
"""
import torch
import torch.nn as nn

from scipy.io import loadmat
import os

from src.nn.ind_mdl.event_detection.n_event_det_data_prep import *
from src.nn.ind_mdl.event_detection.n_event_det_1d_cnn import SpikeNet

dir_name = "EE32009_CW"
p = Path.cwd()
while p.name != dir_name:
    if p.parent == p:
        raise FileNotFoundError(f"Directory '{dir_name}' not found above {Path.cwd()}")
    p = p.parent
os.chdir(p)
print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data1 = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

# Imma make a general training algo for d2 but it can be generalised for each dataset

########################################################################################################################
# DATA #
########################################################################################################################

# make index binary
idx_bin = np.isin(np.arange(data1['d'][0].shape[0]), data1['Index'][0]).astype(int)

# get an 80% training set
split_index = int(len(data1['d'][0]) * 0.8)

data1_train = data1['d'][0][:split_index]
# we need d2 to be the same size so unfortunately we lose some resolution
data_unknown_train = data6['d'][0][:split_index]
idx_train = idx_bin[:split_index]

training_set = TrainingData(data1_train, data_unknown_train, idx_train)

plot_sample_with_binary(training_set.wavelet_degraded_80dB_data, training_set.wavelet_degraded_80dB_data)

print()
########################################################################################################################
# MODEL SETUP, TRAINING PARAMS #
########################################################################################################################

model = SpikeNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

training_threshold = 0.7
num_epochs = 100

print()
########################################################################################################################
# TRAINING LOOP #
########################################################################################################################

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    y_true_train, y_pred_train = [], []

    for X_batch, y_batch in training_set.loader_t:
        try:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


            train_loss += loss.item()
        except torch.AcceleratorError or RuntimeError:
            print(outputs.min().item(), outputs.max().item())
            print("X sample:\n", X_batch[0, 0, :])  # first example, first channel, first 10 values
            print("Y sample:\n", y_batch[0, :])  # corresponding label
            print()

        # store results for metrics
        preds = (torch.sigmoid(outputs) > training_threshold).float()
        y_true_train.extend(y_batch.cpu().numpy().flatten())
        y_pred_train.extend(preds.cpu().numpy().flatten())

    # average loss
    train_loss /= len(training_set.loader_t)

    # compute train metrics
    y_true = np.array(y_true_train)
    y_pred = np.array(y_pred_train)


    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    train_acc = (TP + TN) / (TP + TN + FP + FN)
    train_prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    train_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
    train_f1   = 2 * train_prec * train_rec / (train_prec + train_rec) if (train_prec + train_rec) > 0 else 0

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(
        f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
    print("-" * 70)

print()
########################################################################################################################
# .PT FILE SAVE #
########################################################################################################################

torch.save(model.state_dict(),
           "src/nn/ind_mdl/event_detection/models/D6/20251114_neuron_event_det_cnn.pt")

print()

