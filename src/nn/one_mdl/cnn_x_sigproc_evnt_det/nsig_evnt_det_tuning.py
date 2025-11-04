# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_evnt_det_tuning.py
 Description:
 Author:       Joshua Poole
 Created on:   20251103
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
from src.nn.cnn_evnt_det.n_evnt_det_utils import plot_sample_with_binary, prep_set_train, prep_set_val

from sklearn.metrics import precision_recall_curve

def tune_thr(y_probs, y_true):
    best_f1, best_thresh = 0, 0
    for t in np.linspace(0.1, 0.9, 17):
        preds = (y_probs > t).astype(int)
        f1 = f1_score(y_true, preds)
        #prec = precision_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_f1, best_thresh


"""
Loss:       How “off” the model’s predictions are numerically
Accuracy:   % of correct predictions
Precision:  When the model predicts 1, how often is it actually 1?
Recall:     Of all true X examples, how many did the model find?
F1:         2 × (Precision × Recall) / (Precision + Recall)
"""