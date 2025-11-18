# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_event_det_data_prep.py
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
from torch.utils.data import TensorDataset, DataLoader
import copy
from src.nn.ind_mdl.noise_suppression.noise_suppression_utils import *


class TrainingData:
    def __init__(self, raw_80dB_data, raw_unknown_data, idx_bin, fs=25000):

        # index logic
        self.idx_ground_truth_bin = idx_bin
        self.expanded_idx_ground_truth_bin = self.widen_labels()

        # signal processing for the neuron data
        self.degraded_80dB_data = spectral_power_degrade(raw_80dB_data, raw_unknown_data, fs)
        # bandpass to eliminate large signal sway (simulated for inf)
        self.bandpass_degraded_80dB_data = bandpass_neurons(self.degraded_80dB_data)

        X_tensors, y_tensors = self.prep_set_train(self.bandpass_degraded_80dB_data,
                                                   self.expanded_idx_ground_truth_bin)
        self.dataset_t = TensorDataset(X_tensors, y_tensors)
        self.loader_t = DataLoader(self.dataset_t, batch_size=64, shuffle=True)

    def widen_labels(self, width=3):
        """
        Expand binary 1s in a 1D label array by ±width samples.
        We have decided that 3 is the best balance between learnability
        and the models ability to detect stacked spikes.
        """
        expanded = np.zeros_like(self.idx_ground_truth_bin)
        idx = np.where(self.idx_ground_truth_bin == 1)[0]
        for i in idx:
            start = max(0, i - width)
            end = min(len(self.idx_ground_truth_bin), i + width + 1)
            expanded[start:end] = 1
        return expanded

    def norm_data(self, raw_data):
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

    def prep_set_train(self, data, labels, window_size=128, stride=1, window_interleave=1):
        """
        Package the series and indexes into tensors with respect to the
        required window size and stride length.

        It has a sliding window (hence stride) that means the output
        tensors will represent 2*input_size - window_size 1D datapoints

        window_interleave > 0  break this whole thing if using a wavelet as many zeros!
        """
        X, y = [], []
        np_lables = np.array(labels)
        noise_c = 0
        for i in range(0, len(data) - window_size, stride):
            # for each spike we want window_interleave windows of noise
            window_split_idx = [int(window_size * 0.5)]
            data_re_norm = self.norm_data(data[i:i + window_size])
            if any(np_lables[i + int(idx)] == 1 for idx in window_split_idx):
                X.append(data_re_norm)
                y.append(labels[i:i + window_size])
                # plot_sample_with_binary(data_re_norm, labels[i:i + window_size])
                noise_c = 0
            elif np_lables[i - window_size:i + window_size].mean() == 0 and noise_c < window_interleave:
                X.append(data_re_norm)
                y.append(labels[i:i + window_size])
                # plot_sample_with_binary(data_re_norm, labels[i:i + window_size])
                noise_c += 1

        X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
        y = torch.from_numpy(np.stack(y)).float()
        print(X.size(), y.size())
        return X, y


class InferenceData:
    def __init__(self, raw_unknown_data):

        # bandpass to eliminate large signal sway (simulated for inf)
        self.bandpass_degraded_80dB_data = bandpass_neurons(raw_unknown_data)

        X_tensors, self.index_map = self.prep_set_inf(self.bandpass_degraded_80dB_data)
        self.dataset_i = TensorDataset(X_tensors)
        self.loader_i = DataLoader(self.dataset_i, batch_size=64, shuffle=False)


    def norm_data(self, raw_data):
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

    def prep_set_inf(self, data, window_size=128, stride=1):
        """
        This version id for preparing the inference tensors and still must
        side for the peak detection signal proc algo to sit inside it

        We then need to maintain dimensionality to build index list at the end
        """
        X = []
        index_map = []
        for i in range(0, len(data) - window_size, stride):
            data_re_norm = self.norm_data(data[i:i + window_size])
            X.append(data_re_norm)
            index_map.append(np.arange(i, i + window_size))
        X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
        index_map = np.stack(index_map)  # (N, window_size)
        print(X.size())
        return X, index_map


class ValidationData:
    def __init__(self, raw_80dB_data, raw_unknown_data, idx_bin, fs=25000):

        # index logic
        self.idx_ground_truth_bin = idx_bin
        self.expanded_idx_ground_truth_bin = self.widen_labels()

        # signal processing for the neuron data
        self.degraded_80dB_data = spectral_power_degrade(raw_80dB_data, raw_unknown_data, fs)
        # bandpass to eliminate large signal sway (simulated for inf)
        self.bandpass_degraded_80dB_data = bandpass_neurons(self.degraded_80dB_data)

        X_tensors, y_tensors, self.index_map = (
            self.prep_set_val(self.bandpass_degraded_80dB_data,
                              self.expanded_idx_ground_truth_bin))
        self.dataset_v = TensorDataset(X_tensors, y_tensors)
        self.loader_v = DataLoader(self.dataset_v, batch_size=64, shuffle=False)

    def widen_labels(self, width=3):
        """
        Expand binary 1s in a 1D label array by ±width samples.
        We have decided that 3 is the best balance between learnability
        and the models ability to detect stacked spikes.
        """
        expanded = np.zeros_like(self.idx_ground_truth_bin)
        idx = np.where(self.idx_ground_truth_bin == 1)[0]
        for i in idx:
            start = max(0, i - width)
            end = min(len(self.idx_ground_truth_bin), i + width + 1)
            expanded[start:end] = 1
        return expanded

    def norm_data(self, raw_data):
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

    def prep_set_val(self, data, labels, window_size=128, stride=1):
        """
        This version id for preparing the inference tensors and still must
        side for the peak detection signal proc algo to sit inside it

        We then need to maintain dimensionality to build index list at the end
        """
        X, y = [], []
        index_map = []
        for i in range(0, len(data) - window_size, stride):
            data_re_norm = self.norm_data(data[i:i + window_size])
            X.append(data_re_norm)
            y.append(labels[i:i + window_size])
            index_map.append(np.arange(i, i + window_size))
        X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
        y = torch.from_numpy(np.stack(y)).float()
        index_map = np.stack(index_map)  # (N, window_size)
        print(X.size(), y.size())
        return X, y, index_map


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

def plot_sample_with_binary(sample, binary_signal):
    """
    This plotting function will plot a time series
    accompanied by the binary decision signal
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