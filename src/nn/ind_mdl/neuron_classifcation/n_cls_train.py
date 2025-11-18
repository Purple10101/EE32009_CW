# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls_train.py
 Description:
 Author:       Joshua Poole
 Created on:   20251114
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
import torch.nn as nn

from scipy.io import loadmat

from src.nn.ind_mdl.neuron_classifcation.n_cls_data_prep import *
from src.nn.ind_mdl.neuron_classifcation.n_cls_cnn import NeuronCNN, DualNormNeuronCNN

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


########################################################################################################################
# DATA #
########################################################################################################################

ground_truth_data = data1['d'][0]
ground_idx_list = data1['Index'][0]
ground_cls_list = data1['Class'][0]

target_data = data2['d'][0]

training_set = TrainingData(ground_truth_data, ground_idx_list, ground_cls_list)


print()
########################################################################################################################
# MODEL SETUP, TRAINING PARAMS #
########################################################################################################################

model = NeuronCNN(5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 200

print()
########################################################################################################################
# TRAINING LOOP #
########################################################################################################################

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    total_batches = 0
    y_true_all, y_pred_all = [], []

    for X_batch, y_batch in training_set.loader_t:
        y_batch -= 1
        optimizer.zero_grad()
        preds = model(X_batch.to(device))
        loss = criterion(preds, y_batch.to(device))
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

print()
########################################################################################################################
# .PT FILE SAVE #
########################################################################################################################

torch.save(model.state_dict(),
           "src/nn/ind_mdl/neuron_classifcation/models/20251118_cls_cnn_all.pt")

print()

