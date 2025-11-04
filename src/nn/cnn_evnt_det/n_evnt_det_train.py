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
import copy
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.ops import sigmoid_focal_loss


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.nn.cnn_evnt_det.n_evnt_det import SpikeNet
from src.nn.cnn_evnt_det.n_evnt_det_utils import (plot_sample_with_binary,
                                                  prep_set_train, prep_set_val,
                                                  norm_data, degrade)

from sklearn.metrics import precision_recall_curve


os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(os.getcwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

# raw datasets
raw_data_80 = norm_data(data['d'][0])
raw_data_60 = norm_data(degrade(data['d'][0], data2['d'][0], 0.25))
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

# training data
raw_data_train_80 = raw_data_80[:split_index_raw]
raw_data_train_60 = raw_data_60[:split_index_raw]
raw_data_train_40 = raw_data_40[:split_index_raw]
raw_data_train_20 = raw_data_20[:split_index_raw]
raw_data_train_0 = raw_data_0[:split_index_raw]
#raw_data_train_sub0 = raw_data_sub0[:split_index_raw]
raw_data_lists = [raw_data_train_80, raw_data_train_60, raw_data_train_40,
                  raw_data_train_20, raw_data_train_0]

idx_bin_train = labels_bin[:split_index_raw]

"""# remove for run: plotting
plot_sample_with_binary(raw_data_train_80, idx_bin_train)
plot_sample_with_binary(raw_data_train_60, idx_bin_train)
plot_sample_with_binary(raw_data_train_40, idx_bin_train)
plot_sample_with_binary(raw_data_train_20, idx_bin_train)
plot_sample_with_binary(raw_data_train_0, idx_bin_train)
"""
#plot_sample_with_binary(degrade(data['d'][0], data6['d'][0])[-10000:], labels_bin[-10000:])
#plot_sample_with_binary(data6["d"][0][-10000:], data6["d"][0][-10000:])


X_t, y_t = prep_set_train(raw_data_lists, idx_bin_train)
dataset_t = TensorDataset(X_t, y_t)
loader_t = DataLoader(dataset_t, batch_size=64, shuffle=True)

# val data
raw_data_val = raw_data_80[split_index_raw:]
idx_bin_val = labels_bin[split_index_raw:]
X_v, y_v = prep_set_val(raw_data_val, idx_bin_val)
dataset_v = TensorDataset(X_v, y_v)
loader_v = DataLoader(dataset_v, batch_size=64, shuffle=True)

# plotting sample data
sample_dataset_raw_data = raw_data_val
sample_dataset_idx_bin = idx_bin_val
X_sample = torch.tensor(sample_dataset_raw_data, dtype=torch.float32).unsqueeze(1)
y_sample = torch.tensor(sample_dataset_idx_bin, dtype=torch.float32)

# model
model = SpikeNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


traning_start_th = 0.65

# training
num_epochs = 200

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    y_true_train, y_pred_train = [], []

    for X_batch, y_batch in loader_t:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # store results for metrics
        preds = (torch.sigmoid(outputs) > traning_start_th).float()
        y_true_train.extend(y_batch.cpu().numpy().flatten())
        y_pred_train.extend(preds.cpu().numpy().flatten())

    # average loss
    train_loss /= len(loader_t)

    # compute train metrics
    y_true = np.array(y_true_train)
    y_pred = np.array(y_pred_train)

    #print(tune_thr(y_pred, y_true))

    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    train_acc = (TP + TN) / (TP + TN + FP + FN)
    train_prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    train_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
    train_f1   = 2 * train_prec * train_rec / (train_prec + train_rec) if (train_prec + train_rec) > 0 else 0


    """# validation
    model.eval()
    val_loss = 0.0
    y_true_val, y_pred_val = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader_v:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            # store results for metrics
            preds = (torch.sigmoid(outputs) > traning_start_th).float()
            y_true_val.extend(y_batch.cpu().numpy().flatten())
            y_pred_val.extend(preds.cpu().numpy().flatten())

    val_loss /= len(loader_v)

    # compute validation metrics
    y_true = np.array(y_true_val)
    y_pred = np.array(y_pred_val)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    val_acc = (TP + TN) / (TP + TN + FP + FN)
    val_prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    val_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
    val_f1   = 2 * val_prec * val_rec / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0
"""
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(
        f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
    #print(
    #    f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
    print("-" * 70)

torch.save(model.state_dict(), "src/nn/models/20251104_neuron_event_det_cnn_dilation_mimic_noise.pt")

# load model and evaluate performance
model = SpikeNet().to(device)
model.load_state_dict(torch.load("src/nn/models/20251104_neuron_event_det_cnn_dilation_mimic_noise.pt"))
model.eval()

with torch.no_grad():
    X_sample = X_sample.unsqueeze(0)
    X_sample = X_sample.permute(0, 2, 1)
    outputs = model(X_sample.to(device))
    preds = (outputs > traning_start_th).float()
    print(len([x for x in preds.squeeze().tolist() if x != 0]))

scorecard_1 = []
missed_spk_c = 0
scorecard_0 = []
false_spk_c = 0

preds_lst = preds.squeeze().tolist()
for idx in range(len(preds_lst)):
    pred = preds_lst[idx]
    real = sample_dataset_idx_bin[idx]
    if real == 0:
        if pred == real:
            scorecard_0.append(1)
        else:
            scorecard_0.append(0)
            false_spk_c += 1

    if real == 1:
        if pred == real:
            scorecard_1.append(1)
        else:
            scorecard_1.append(0)
            missed_spk_c += 1

num_known_spikes = len([x for x in sample_dataset_idx_bin if x != 0])

print(f"{missed_spk_c}/{num_known_spikes} spike indexes are missed. "
      f"{1 - (missed_spk_c/num_known_spikes)}")
print(f"{false_spk_c}/{len(sample_dataset_raw_data) - num_known_spikes} zero indexes are misidentified as spikes. "
      f"{1 - (false_spk_c/(len(sample_dataset_raw_data) - num_known_spikes))}")

plot_sample_with_binary(sample_dataset_raw_data[-1000:], preds.squeeze().tolist()[-1000])
print()

"""
Loss:       How “off” the model’s predictions are numerically
Accuracy:   % of correct predictions
Precision:  When the model predicts 1, how often is it actually 1?
Recall:     Of all true X examples, how many did the model find?
F1:         2 × (Precision × Recall) / (Precision + Recall)
"""