# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls_data_prep.py
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
import torch
from torch.utils.data import TensorDataset, DataLoader
import copy
from src.nn.ind_mdl.noise_suppression.noise_suppression_utils import *


class TrainingData:
    def __init__(self, raw_80dB_data, idx_list, cls_list, fs=25000):
        # index and class logic
        split_index = int(len(raw_80dB_data) * 0.8)

        sort_order = np.argsort(idx_list)
        event_idx = idx_list[sort_order]
        event_cls = cls_list[sort_order]

        train_mask = event_idx < split_index
        val_mask = event_idx >= split_index
        # Event indexes
        self.idx_list_train = event_idx[train_mask]
        self.idx_list_val = event_idx[val_mask] - split_index  # shift for validation subset
        # Corresponding labels
        self.cls_list_train = event_cls[train_mask]
        self.cls_list_val = event_cls[val_mask]

        # signal processing for the neuron data
        # Spectral power degrade may be messing with cls?
        # We just cant degrade for classification bro
        # For now we will just train on d1 and try to bring everything up to its level of SNR

        self.data_80dB_global_norm = np.array(self.norm_data(raw_80dB_data))
        # split spikes for training
        self.spike_windows_train = self.split_spike(
            self.data_80dB_global_norm[:split_index], self.idx_list_train, self.cls_list_train)
        self.spike_windows_val = self.split_spike(
            self.data_80dB_global_norm[split_index:] , self.idx_list_val, self.cls_list_val)

        """
        # Uncomment for examples
        max_per_class = 3
        class_counts = {}
        for window in self.spike_windows_train:
            cls = window["Classification"]

            # Initialize count if class hasn't appeared yet
            if cls not in class_counts:
                class_counts[cls] = 0

            # Only print if we haven't hit the limit for this class
            if class_counts[cls] < max_per_class:
                plot_classification(window["Capture"], cls, cls)
                print(f"Printed example #{class_counts[cls] + 1} for class {cls}")
                class_counts[cls] += 1
        """


        # prep data loader
        self.loader_t = self.prep_set_train(self.spike_windows_train)
        self.loader_v = self.prep_set_train(self.spike_windows_val)

    def norm_data(self, raw_data):
        """
        Norm raw_data between 1 and -1
        centered about zero
        """
        ret_val = copy.deepcopy(raw_data)
        raw_data_max = max(ret_val)
        raw_data_min = min(ret_val)
        ret_val = (2 * (ret_val - raw_data_min) /
                   (raw_data_max - raw_data_min) - 1)
        return ret_val

    def split_spike(self, dataset, idx_list, cls_list, capture_width=64, capture_weight=0.80):
        """
        :param capture_width:  How many samples per capture
        :param capture_weight: A right bias value as the dataset
                               tends to house important data on the right of idx
        :return:
        """
        captures_list_all = []
        for classification_index, idx in enumerate(idx_list):
            cap = []
            for sample_idx in range(int(idx - (capture_width * (1 - capture_weight))),
                                    int(idx + (capture_width * capture_weight))):
                cap.append(dataset[sample_idx])
            #plot_classification(cap, cls_list[classification_index], cls_list[classification_index])
            captures_list_all.append({
                "Capture": cap,
                "Classification": cls_list[classification_index],
                "PeakIdx": capture_width * (1 - capture_weight)
            })
        return captures_list_all

    def prep_set_train(self, windows):
        cap_list = [d["Capture"] for d in windows]
        cls_list = [d["Classification"] for d in windows]

        X = np.array(cap_list, dtype=np.float32)
        y = np.array(cls_list, dtype=np.int64)
        X = np.expand_dims(X, axis=1)

        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return loader


class InferenceDataCls:
    def __init__(self, raw_unknown_data, ref_clean_data, idx_list):

        self.idx_list = idx_list
        # prep data loader
        raw_unknown_data_highpass = highpass_neurons(raw_unknown_data)
        raw_unknown_data_highpass_boosted_snr = spectral_power_suppress(raw_unknown_data_highpass,
                                                                        ref_clean_data, 25_000)
        inf_windows = self.split_spike(self.norm_data(raw_unknown_data_highpass_boosted_snr))
        self.loader_v = self.prep_set_inf(inf_windows)

    def norm_data(self, raw_data):
        """
        Norm raw_data between 1 and -1
        centered about zero
        """
        ret_val = copy.deepcopy(raw_data)
        raw_data_max = max(ret_val)
        raw_data_min = min(ret_val)
        ret_val = (2 * (ret_val - raw_data_min) /
                   (raw_data_max - raw_data_min) - 1)
        return ret_val

    def split_spike(self, raw_unknown_data, capture_width=64, capture_weight=0.80):
        """
        :param raw_unknown_data:
        :param capture_width:  How many samples per capture
        :param capture_weight: A right bias value as the dataset
                               tends to house important data on the right of idx
        :return:               A list of all captures in format {Capture Ch0=[], Capture Ch1=[], Classification=int}
        """
        captures_list_all = []
        for classification_index, idx in enumerate(self.idx_list):
            cap = []
            for sample_idx in range(int(idx - (capture_width * (1 - capture_weight))),
                                    int(idx + (capture_width * capture_weight))):
                cap.append(raw_unknown_data[sample_idx])
            # pad to ensure dimensionality
            # this only happens if a spike occurs too soon or too late
            if len(cap) < capture_width:
                pad_len = capture_width - len(cap)
                cap.extend([cap[-1]] * pad_len)
            captures_list_all.append({
                "Capture": cap,
                "PeakIdx": capture_width * (1 - capture_weight)
            })
        return captures_list_all

    def prep_set_inf(self, windows):
        cap_list = [d["Capture"] for d in windows]

        X = np.array(cap_list, dtype=np.float32)
        X = np.expand_dims(X, axis=1)

        X_tensor = torch.tensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return loader


def plot_classification(window, truth, infered):
    plt.figure(figsize=(10, 5))
    plt.plot(window, label='Signal', color='black')

    plt.plot([], [], ' ', label=f'Ground Truth: {truth}')
    plt.plot([], [], ' ', label=f'Predicted: {infered}')

    plt.title("Time Series with Ground Truth and Predicted Classification")
    plt.xlabel("Index")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)

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