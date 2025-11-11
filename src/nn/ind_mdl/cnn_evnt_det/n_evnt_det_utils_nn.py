    # -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_evnt_det_utils_nn.py
 Description:
 Author:       Joshua Poole
 Created on:   20251111
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

import numpy as np
import copy
import torch

from src.nn.ind_mdl.cnn_evnt_det.n_evnt_det_utils_sig import *


def widen_labels(labels, width=3):
    """Expand binary 1s in a 1D label array by ±width samples."""
    expanded = np.zeros_like(labels)
    idx = np.where(labels == 1)[0]
    for i in idx:
        start = max(0, i - width)
        end = min(len(labels), i + width + 1)
        expanded[start:end] = 1
    return expanded

def prep_set_train(data, labels, window_size=128, stride=1, window_interleave=1):
    """
    Package the series and indexes into tensors with respect to the
    required window size and stride length.

    It has a sliding window (hence stride) that means the output
    tensors will represent 2*input_size - window_size 1D datapoints
    """
    X, y = [], []
    np_lables = np.array(labels)
    noise_c = 0
    for i in range(0, len(data) - window_size, stride):
        # for each spike we want window_interleave windows of noise
        window_split_idx = [int(window_size * 0.5)]
        data_re_norm = norm_data(data[i:i + window_size])
        if any(np_lables[i + int(idx)] == 1 for idx in window_split_idx):
            X.append(data_re_norm)
            y.append(labels[i:i + window_size])
            #plot_sample_with_binary(data_re_norm, labels[i:i + window_size])
            noise_c = 0
        elif np_lables[i-window_size:i + window_size].mean() == 0 and noise_c < window_interleave:
            X.append(data_re_norm)
            y.append(labels[i:i + window_size])
            #plot_sample_with_binary(data_re_norm, labels[i:i + window_size])
            noise_c += 1

    X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
    y = torch.from_numpy(np.stack(y)).float()
    print(X.size(), y.size())
    return X, y

def prep_set_val(data, labels, window_size=128, stride=1):
    """
    This version id for preparing the inference tensors and still must
    side for the peak detection signal proc algo to sit inside it

    We then need to maintain dimensionality to build index list at the end
    """
    X, y = [], []
    index_map = []
    for i in range(0, len(data) - window_size, stride):
        data_re_norm = norm_data(data[i:i + window_size])
        X.append(data_re_norm)
        y.append(labels[i:i + window_size])
        index_map.append(np.arange(i, i + window_size))
    X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
    y = torch.from_numpy(np.stack(y)).float()
    index_map = np.stack(index_map)  # (N, window_size)
    print(X.size(), y.size())
    return X, y, index_map

def prep_set_inf(data, window_size=128, stride=1):
    """
    This version id for preparing the inference tensors and still must
    side for the peak detection signal proc algo to sit inside it

    We then need to maintain dimensionality to build index list at the end
    """
    X = []
    index_map = []
    for i in range(0, len(data) - window_size, stride):
        data_re_norm = norm_data(data[i:i + window_size])
        X.append(data_re_norm)
        index_map.append(np.arange(i, i + window_size))
    X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
    index_map = np.stack(index_map)  # (N, window_size)
    print(X.size())
    return X, index_map


def norm_data(raw_data):
    """
    Norm the whole dataset between 1 and -1
    centered about zero

    We norm the whole dataset then for spike detection
    we normalise each window again to boost small peaks
    """
    ret_val = copy.deepcopy(raw_data)
    raw_data_max = max(ret_val)
    raw_data_min = min(ret_val)
    ret_val = (2 * (ret_val - raw_data_min) /
               (raw_data_max - raw_data_min) - 1)
    return ret_val

def nonmax_rejection(preds, threshold, refractory=10):
    preds = np.array(preds)
    candidate_indices = np.where(preds > threshold)[0]
    final_spikes = []
    i = 0
    while i < len(candidate_indices):
        start = candidate_indices[i]
        region = candidate_indices[(candidate_indices >= start) &
                                   (candidate_indices < start + refractory)]
        best = region[np.argmax(preds[region])]
        final_spikes.append(best)
        i += len(region)
    labels_bin = np.isin(np.arange(len(preds)), final_spikes).astype(int)
    return labels_bin

def tolerant_binary_metrics(preds_bin, idx_bin_val, tol=50):
    preds_bin = np.asarray(preds_bin, dtype=int)
    idx_bin_val = np.asarray(idx_bin_val, dtype=int)
    assert len(preds_bin) == len(idx_bin_val), "Preds and labels must be same length"

    # Get indices of true positives and predicted positives
    true_idxs = np.where(idx_bin_val == 1)[0]
    pred_idxs = np.where(preds_bin == 1)[0]

    # --- Check predicted 1s ---
    tp = 0
    for p in pred_idxs:
        low, high = max(0, p - tol), min(len(idx_bin_val), p + tol + 1)
        if np.any(idx_bin_val[low:high] == 1):
            tp += 1

    fp = len(pred_idxs) - tp

    # --- Check missed true 1s ---
    fn = 0
    for t in true_idxs:
        low, high = max(0, t - tol), min(len(preds_bin), t + tol + 1)
        if not np.any(preds_bin[low:high] == 1):
            fn += 1

    # --- Compute zeros agreement (for info only) ---
    zeros_match = np.sum((preds_bin == 0) & (idx_bin_val == 0))
    zero_agreement = zeros_match / np.sum(idx_bin_val == 0)

    # --- Metrics ---
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    acc       = (tp + zeros_match) / len(preds_bin)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "TP": tp, "FP": fp, "FN": fn,
        "zero_agreement": zero_agreement
    }


