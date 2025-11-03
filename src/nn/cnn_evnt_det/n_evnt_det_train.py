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
from src.nn.cnn_evnt_det.n_evnt_det_tuning import tune_thr
from src.nn.cnn_evnt_det.n_evnt_det_utils import plot_sample_with_binary, prep_set_train, prep_set_val

from sklearn.metrics import precision_recall_curve


os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(os.getcwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def norm_data(raw_data):
    """
    Norm the whole dataset between 1 and -1
    centered about zero
    """
    ret_val = copy.deepcopy(raw_data)
    raw_data_max = max(ret_val)
    raw_data_min = min(ret_val)
    ret_val = (2 * (ret_val - raw_data_min) /
               (raw_data_max - raw_data_min) - 1)
    return ret_val

data = loadmat('data\D1.mat')

raw_data = data['d'][0]
raw_data = norm_data(raw_data)
idx_lst = data['Index'][0]
tr_to_tst_r=0.8

num_pos = 0
num_neg = 0

labels_bin = []
for y in range(raw_data.shape[0]):
    if y in idx_lst:
        labels_bin.append(1)
        num_pos += 1
    else:
        labels_bin.append(0)
        num_neg += 1

pos_weight_value = num_neg/num_pos
pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)

split_index_raw = int(len(raw_data) * tr_to_tst_r)

# training data
raw_data_train = raw_data[:split_index_raw]
idx_bin_train = labels_bin[:split_index_raw]
X_t, y_t = prep_set_train(raw_data_train, idx_bin_train)
dataset_t = TensorDataset(X_t, y_t)
loader_t = DataLoader(dataset_t, batch_size=64, shuffle=True)

# val data
raw_data_val = raw_data[split_index_raw:]
idx_bin_val = labels_bin[split_index_raw:]
X_v, y_v = prep_set_val(raw_data_val, idx_bin_val)
dataset_v = TensorDataset(X_v, y_v)
loader_v = DataLoader(dataset_v, batch_size=64, shuffle=True)

# plotting sample data
sample_dataset_raw_data = raw_data
sample_dataset_idx_bin = labels_bin
X_sample = torch.tensor(sample_dataset_raw_data, dtype=torch.float32).unsqueeze(1)
y_sample = torch.tensor(sample_dataset_idx_bin, dtype=torch.float32)

# model
model = SpikeNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


traning_start_th = 0.6

# training
num_epochs = 500

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


    """
    # validation
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

torch.save(model.state_dict(), "src/nn/models/2025113_neuron_event_det_cnn_dilation.pt")

# load model and evaluate performance
model = SpikeNet().to(device)
model.load_state_dict(torch.load("src/nn/models/2025113_neuron_event_det_cnn_dilation.pt"))
model.eval()

with torch.no_grad():
    X_sample = X_sample.unsqueeze(0)
    X_sample = X_sample.permute(0, 2, 1)
    outputs = model(X_sample.to(device))
    preds = (outputs > traning_start_th).float()

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

TP = np.sum((preds_lst == 1) & (sample_dataset_idx_bin == 1))
TN = np.sum((preds_lst == 0) & (sample_dataset_idx_bin == 0))
FP = np.sum((preds_lst == 1) & (sample_dataset_idx_bin == 0))
FN = np.sum((preds_lst == 0) & (sample_dataset_idx_bin == 1))

val_acc = (TP + TN) / (TP + TN + FP + FN)
val_prec = TP / (TP + FP) if (TP + FP) > 0 else 0
val_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
val_f1   = 2 * val_prec * val_rec / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0

print(
    f"  Val   Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
print("-" * 70)

print(f"{missed_spk_c}/{len(idx_lst)} spike indexes are missed. {1 - (missed_spk_c/len(idx_lst))}")
print(f"{false_spk_c}/{len(raw_data) - len(idx_lst)} zero indexes are misidentified as spikes. {1 - (false_spk_c/(len(raw_data) - len(idx_lst)))}")


plot_sample_with_binary(sample_dataset_raw_data.tolist()[-1000:], preds.squeeze().tolist()[-1000])
print()

"""
Loss:       How “off” the model’s predictions are numerically
Accuracy:   % of correct predictions
Precision:  When the model predicts 1, how often is it actually 1?
Recall:     Of all true X examples, how many did the model find?
F1:         2 × (Precision × Recall) / (Precision + Recall)
"""