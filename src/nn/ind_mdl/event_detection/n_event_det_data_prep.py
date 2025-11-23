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


def prep_set_train_val(data, labels_idx, labels_bin, window_size=128):
    X = []
    y = []
    indices = []

    data_len = len(data)

    # The target spike position inside each window
    z = int(window_size * 0.2)

    labels_idx = np.array(labels_idx)

    for index in labels_idx:

        # spike windows
        window_start = index - z
        window_end = window_start + window_size

        if 0 <= window_start and window_end <= data_len:
            X.append(data[window_start:window_end])
            y.append(1.0)  # spike at the correct location
            indices.append(index)
            #plot_sample_with_binary(data[window_start:window_end], labels_bin[window_start:window_end])
            #print()

        # no spike windows
        while True:
            rand_idx = np.random.randint(z, data_len - (window_size - z))
            if labels_bin[rand_idx] == 0:
                break

        window_start = rand_idx - z
        window_end = window_start + window_size

        X.append(data[window_start:window_end])
        y.append(0.0)
        indices.append(index)
        plot_widow(data[window_start:window_end])
        print()

        # spike not at z windows
        # pick a spike close but NOT at z
        offset_choices = np.arange(-20, 21)
        offset_choices = offset_choices[offset_choices != 0]

        misaligned_idx = index + np.random.choice(offset_choices)

        if misaligned_idx >= 0 and misaligned_idx < data_len:

            window_start = misaligned_idx - z
            window_end = window_start + window_size

            if 0 <= window_start and window_end <= data_len:
                X.append(data[window_start:window_end])
                y.append(0.0)  # spike present but NOT at z
                indices.append(index)
                plot_widow(data[window_start:window_end])
                print()

    X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()  # (B,1,W)
    y = torch.tensor(y).float().unsqueeze(1)  # (B,1)

    return X, y, indices


class TrainingData:
    def __init__(self,
                 raw_80dB_data,
                 raw_unknown_data,
                 idx_list,
                 idx_bin,
                 target_id,
                 widen_labels,
                 fs=25000):

        # hyperparameters
        self.widen_labels_val = widen_labels

        # index logic
        self.idx_ground_truth_bin = idx_bin
        self.idx_groud_truth_list = idx_list
        self.expanded_idx_ground_truth_bin = self.widen_labels()

        # Data Processing Pipeline
        degraded_80dB_data = zscore(spectral_power_degrade(raw_80dB_data, raw_unknown_data, fs))
        degraded_80dB_resynth_data = zscore(spectral_power_suppress(degraded_80dB_data, raw_80dB_data, fs))
        wt_degraded_80dB_resynth_data = zscore(filter_wavelet(degraded_80dB_resynth_data))
        if target_id == 5:
            wt_degraded_80dB_resynth_data = zscore(add_colored_noise(wt_degraded_80dB_resynth_data, std=3))
        elif target_id == 6:
            wt_degraded_80dB_resynth_data = zscore(add_colored_noise(wt_degraded_80dB_resynth_data, std=5))
        bp_wt_degraded_80dB_resynth_data = zscore(bandpass_neurons(wt_degraded_80dB_resynth_data))

        self.data_proc = bp_wt_degraded_80dB_resynth_data

        plot_widow(self.data_proc)
        #plot_widow(self.data_proc[100_000:101_000])

        X_tensors, y_tensors, self.indices = prep_set_train_val(self.data_proc,
                                                   self.idx_groud_truth_list,
                                                   self.expanded_idx_ground_truth_bin)
        self.dataset_t = TensorDataset(X_tensors, y_tensors)
        self.loader_t = DataLoader(self.dataset_t, batch_size=64, shuffle=True)

    def widen_labels(self):
        """
        Expand binary 1s in a 1D label array by ±width samples.
        We have decided that 5 is the best balance between learnability
        and the models ability to detect stacked spikes.

        maybe unnecessary
        """
        expanded = np.zeros_like(self.idx_ground_truth_bin)
        idx = np.where(self.idx_ground_truth_bin == 1)[0]
        for i in idx:
            start = max(0, i - self.widen_labels_val)
            end = min(len(self.idx_ground_truth_bin), i + self.widen_labels_val + 1)
            expanded[start:end] = 1
        return expanded

class InferenceDataEvntDet:
    def __init__(self, raw_unknown_data, raw_80dB_data, fs=25_000):

        # Data Processing Pipeline
        spect_supress_data = zscore(spectral_power_suppress(raw_unknown_data, raw_80dB_data, fs))
        wt_spect_supress_data = zscore(filter_wavelet(spect_supress_data))
        bandpass_wt_spect_supress_data = zscore(bandpass_neurons(wt_spect_supress_data))
        self.data_proc = bandpass_wt_spect_supress_data

        X_tensors, self.index_map = self.prep_set_inf(self.data_proc)
        self.dataset_i = TensorDataset(X_tensors)
        self.loader_i = DataLoader(self.dataset_i, batch_size=64, shuffle=False)

    def prep_set_inf(self, data, window_size=128, stride=1):

        X = []
        index_map = []

        N = len(data)
        z = int(window_size * 0.2)  # same as training

        for i in range(0, N, stride):

            # compute window boundaries so that index i is at position z
            window_start = i - z
            window_end = window_start + window_size

            # skip edges where the window does not fit
            if window_start < 0 or window_end > N:
                continue

            # extract the window
            window = data[window_start:window_end]

            X.append(window)
            index_map.append(i)  # this window predicts: "Is there a spike at time i?"

        # convert to tensors
        X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
        index_map = np.array(index_map)

        print(X.shape, index_map.shape)
        return X, index_map

class ValidationData:
    def __init__(self,
                 raw_80dB_data,
                 raw_unknown_data,
                 idx_list,
                 idx_bin,
                 target_id,
                 widen_labels,
                 fs=25000):

        # hyperparameters
        self.widen_labels_val = widen_labels

        # index logic
        self.idx_ground_truth_bin = idx_bin
        self.idx_groud_truth_list = idx_list
        self.expanded_idx_ground_truth_bin = self.widen_labels()

        # Data Processing Pipeline
        degraded_80dB_data = zscore(spectral_power_degrade(raw_80dB_data, raw_unknown_data, fs))
        degraded_80dB_resynth_data = zscore(spectral_power_suppress(degraded_80dB_data, raw_80dB_data, fs))
        wt_degraded_80dB_resynth_data = zscore(filter_wavelet(degraded_80dB_resynth_data))
        if target_id == 5:
            wt_degraded_80dB_resynth_data = zscore(add_colored_noise(wt_degraded_80dB_resynth_data, std=3))
        elif target_id == 6:
            wt_degraded_80dB_resynth_data = zscore(add_colored_noise(wt_degraded_80dB_resynth_data, std=5))
        bp_wt_degraded_80dB_resynth_data = zscore(bandpass_neurons(wt_degraded_80dB_resynth_data))

        self.data_proc = bp_wt_degraded_80dB_resynth_data

        plot_widow(self.data_proc)
        # plot_widow(self.data_proc[100_000:101_000])

        X_tensors, y_tensors, self.indices = prep_set_train_val(self.data_proc,
                                                  self.idx_groud_truth_list,
                                                  self.expanded_idx_ground_truth_bin)
        self.dataset_v = TensorDataset(X_tensors, y_tensors)
        self.loader_v = DataLoader(self.dataset_v, batch_size=64, shuffle=False)

    def widen_labels(self, width=5):
        """
        Expand binary 1s in a 1D label array by ±width samples.
        We have decided that 5 is the best balance between learnability
        and the models ability to detect stacked spikes.
        """
        expanded = np.zeros_like(self.idx_ground_truth_bin)
        idx = np.where(self.idx_ground_truth_bin == 1)[0]
        for i in idx:
            start = max(0, i - width)
            end = min(len(self.idx_ground_truth_bin), i + width + 1)
            expanded[start:end] = 1
        return expanded


########################################################################################################################
# UTILS #
########################################################################################################################


def nms(probs, threshold, refractory=5):
    probs = np.asarray(probs)
    N = len(probs)

    # start with zeros
    spikes = np.zeros(N, dtype=int)

    # find all candidate peaks above threshold
    candidates = np.where(probs > threshold)[0]
    visited = np.zeros(N, dtype=bool)

    for idx in candidates:
        if visited[idx]:
            continue

        # define suppression window
        left  = max(0, idx - refractory)
        right = min(N, idx + refractory + 1)

        # find best peak in window
        window = np.arange(left, right)
        best_idx_in_window = window[np.argmax(probs[window])]

        # keep ONLY this index as a spike
        spikes[best_idx_in_window] = 1

        # mark the whole window as visited so no more spikes added here
        visited[left:right] = True

    return spikes

def tolerant_binary_metrics(preds_bin, idx_bin_val, tol=3):
    preds_bin = np.asarray(preds_bin, dtype=int)
    idx_bin_val = np.asarray(idx_bin_val, dtype=int)
    assert len(preds_bin) == len(idx_bin_val), "Preds and labels must be same length"

    # Indices of true positives (ground truth) and predicted positives
    true_idxs = np.where(idx_bin_val == 1)[0]
    pred_idxs = np.where(preds_bin == 1)[0]

    # --- One-to-one matching between true and predicted events ---
    tp = 0
    fn = 0

    # Track which predictions have been used to match a true event
    pred_used = np.zeros(len(pred_idxs), dtype=bool)

    for t in true_idxs:
        low, high = max(0, t - tol), min(len(preds_bin), t + tol + 1)
        # Candidates: predictions in the tolerance window and not used yet
        in_window = (pred_idxs >= low) & (pred_idxs < high) & (~pred_used)

        if np.any(in_window):
            # Pick the first unused prediction in the window (or nearest if you like)
            cand_idx = np.where(in_window)[0][0]
            pred_used[cand_idx] = True
            tp += 1
        else:
            fn += 1

    # Predictions that never got matched are false positives
    fp = np.sum(~pred_used)

    # --- Zeros agreement (for info) ---
    zeros_mask = (idx_bin_val == 0)
    zeros_total = np.sum(zeros_mask)
    zeros_match = np.sum((preds_bin == 0) & zeros_mask)
    zero_agreement = zeros_match / zeros_total if zeros_total > 0 else 0.0

    # --- Metrics ---
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc       = (tp + zeros_match) / len(preds_bin) if len(preds_bin) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
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