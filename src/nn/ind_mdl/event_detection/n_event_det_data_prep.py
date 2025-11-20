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
    def __init__(self, raw_80dB_data, raw_unknown_data, idx_bin, target_id, fs=25000):

        # index logic
        self.idx_ground_truth_bin = idx_bin
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

        #plot_widow(self.data_proc[10_000:])
        #plot_widow(self.data_proc[100_000:101_000])

        X_tensors, y_tensors = self.prep_set_train(self.data_proc,
                                                   self.expanded_idx_ground_truth_bin)
        self.dataset_t = TensorDataset(X_tensors, y_tensors)
        self.loader_t = DataLoader(self.dataset_t, batch_size=64, shuffle=True)

    def widen_labels(self, width=3):
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

    def prep_set_train(self, data, labels, window_size=128, stride=1, window_interleave=5):
        """
        Package the series and indexes into tensors with respect to the
        required window size and stride length.

        for D5 - D6 provide way more noise only windows to drive down FPs
        """
        X, y = [], []
        np_lables_no_exp = np.array(self.idx_ground_truth_bin)
        noise_c = 0
        num_random_positions = 4

        for i in range(0, len(data) - window_size, stride):
            # for each spike we want window_interleave windows of noise
            center = int(window_size * 0.5)
            window_split_idx = [center]
            data_re_norm = data[i:i + window_size]
            if any(np_lables_no_exp[i + int(idx)] == 1 for idx in window_split_idx):
                # find spike index inside global labels
                spike_global_idx = i + center

                # generate 4 random spike positions inside window
                random_positions = np.random.randint(
                    low=int(window_size * 0.2),
                    high=int(window_size * 0.8),
                    size=num_random_positions
                )
                for pos in random_positions:
                    new_i = spike_global_idx - pos
                    if new_i < 0 or new_i + window_size > len(data):
                        continue
                    X.append(data[new_i:new_i + window_size])
                    y.append(labels[new_i:new_i + window_size])
                # plot_sample_with_binary(data_re_norm, labels[i:i + window_size])
                noise_c = 0
            elif np_lables_no_exp[i - window_size:i + window_size].mean() == 0 and noise_c < window_interleave:
                X.append(data_re_norm)
                y.append(labels[i:i + window_size])
                # plot_sample_with_binary(data_re_norm, labels[i:i + window_size])
                noise_c += 1

        X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
        y = torch.from_numpy(np.stack(y)).float()
        print(X.size(), y.size())
        return X, y


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
        """
        This version id for preparing the inference tensors and still must
        side for the peak detection signal proc algo to sit inside it

        We then need to maintain dimensionality to build index list at the end
        """
        X = []
        index_map = []
        for i in range(0, len(data) - window_size, stride):
            data_re_norm = data[i:i + window_size]
            X.append(data_re_norm)
            #plot_widow(data_re_norm)
            index_map.append(np.arange(i, i + window_size))
        X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
        index_map = np.stack(index_map)  # (N, window_size)
        print(X.size())
        return X, index_map


class ValidationData:
    def __init__(self, raw_80dB_data, raw_unknown_data, idx_bin, target_id, fs=25000):

        # index logic
        self.idx_ground_truth_bin = idx_bin
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



        self.data_proc = zscore(bp_wt_degraded_80dB_resynth_data)

        X_tensors, y_tensors, self.index_map = (
            self.prep_set_val(self.data_proc,
                              self.idx_ground_truth_bin))
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

    def prep_set_val(self, data, labels, window_size=128, stride=1):
        """
        This version id for preparing the inference tensors and still must
        side for the peak detection signal proc algo to sit inside it

        We then need to maintain dimensionality to build index list at the end
        """
        X, y = [], []
        index_map = []
        for i in range(0, len(data) - window_size, stride):
            data_re_norm = data[i:i + window_size]
            X.append(data_re_norm)
            y.append(labels[i:i + window_size])
            index_map.append(np.arange(i, i + window_size))
        X = torch.from_numpy(np.stack(X)).unsqueeze(1).float()
        y = torch.from_numpy(np.stack(y)).float()
        index_map = np.stack(index_map)  # (N, window_size)
        print(X.size(), y.size())
        return X, y, index_map


########################################################################################################################
# UTILS #
########################################################################################################################


def nonmax_rejection(preds, threshold, refractory=3):
    # refactory needs a big increase for later datasets
    preds = np.asarray(preds)
    candidate_indices = np.where(preds > threshold)[0]

    if len(candidate_indices) == 0:
        return np.zeros_like(preds, dtype=int)

    final_spikes = []
    current_burst = [candidate_indices[0]]

    # Group candidate indices into bursts based on refractory distance
    for idx in candidate_indices[1:]:
        if idx - current_burst[-1] <= refractory:
            # same burst
            current_burst.append(idx)
        else:
            # burst ended → save best index
            best = current_burst[np.argmax(preds[current_burst])]
            final_spikes.append(best)
            current_burst = [idx]

    # handle last burst
    best = current_burst[np.argmax(preds[current_burst])]
    final_spikes.append(best)

    # produce binary output
    labels_bin = np.zeros_like(preds, dtype=int)
    labels_bin[final_spikes] = 1
    return labels_bin

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