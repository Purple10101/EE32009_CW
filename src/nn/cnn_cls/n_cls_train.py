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

from src.ext.data_loader_cls import plot_sample
from src.nn.cnn_cls.n_cls import NeuronCNN
from src.nn.cnn_cls.n_cls_utils import noise_plt_example, RecordingTrain,  prep_training_set, degrade


os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(os.getcwd())

# data prep
print(os.getcwd())
data = loadmat('data\D1.mat')
data_inf = loadmat('data\D2.mat')

rec = RecordingTrain(data['d'][0], data['Index'][0], data['Class'][0])

# raw datasets


# show example of manual degradation
#noise_plt_example(rec, snr_lst)

dataloader = prep_training_set(rec)

# model and training
model = NeuronCNN(5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    total_batches = 0
    y_true_all, y_pred_all = [], []

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

        # Store predictions and true labels
        y_true_all.extend(y_batch.cpu().numpy())
        y_pred_all.extend(torch.argmax(preds, dim=1).cpu().numpy())

    avg_loss = epoch_loss / total_batches

    # cal and print performance metrics
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    num_classes = np.max(y_true_all) + 1

    total_TP, total_FP, total_FN = 0, 0, 0
    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        TP = np.sum((y_pred_all == c) & (y_true_all == c))
        FP = np.sum((y_pred_all == c) & (y_true_all != c))
        FN = np.sum((y_pred_all != c) & (y_true_all == c))

        total_TP += TP
        total_FP += FP
        total_FN += FN

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall  = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1   = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

        precisions.append(prec)
        recalls.append(recall)
        f1s.append(f1)

    macro_prec = np.mean(precisions)
    macro_rec  = np.mean(recalls)
    macro_f1   = np.mean(f1s)

    acc = np.sum(y_true_all == y_pred_all) / len(y_true_all)

    print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | "
          f"Acc: {acc:.4f} | Prec: {macro_prec:.4f} | "
          f"Rec: {macro_rec:.4f} | F1: {macro_f1:.4f}")

torch.save(model.state_dict(), "src/nn/models/20251103_neuron_total_norm_snr_to_0.pt")