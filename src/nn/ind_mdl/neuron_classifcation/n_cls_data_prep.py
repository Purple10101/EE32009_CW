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
    def __init__(self, raw_80dB_data, raw_unknown_data, idx_list, cls_list, fs=25000):

        # Two channels, one is recoding-wise norm oen is window-wise norm.

        # index and class logic
        split_index = int(len(idx_list) * 0.8)
        self.idx_ground_truth = idx_list
        self.cls_list = cls_list

        # signal processing for the neuron data
        self.degraded_80dB_data = np.array(spectral_power_degrade(raw_80dB_data, raw_unknown_data, fs))
        self.degraded_80dB_data_global_norm = np.array(self.norm_data(self.degraded_80dB_data))

        # split spikes for training
        self.spike_windows_train = self.split_spike()[:split_index]
        self.spike_windows_val = self.split_spike()[split_index:]
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

    def split_spike(self, capture_width=80, capture_weight=0.90):
        """
        :param capture_width:  How many samples per capture
        :param capture_weight: A right bias value as the dataset
                               tends to house important data on the right of idx
        :return:               A list of all captures in format {Capture Ch0=[], Capture Ch1=[], Classification=int}
        """
        captures_list_all = []
        for classification_index, idx in enumerate(self.idx_ground_truth):
            # We rely on the norm and non-norm sets being the same len
            cap_ch0 = [] # the global norm channel for nn
            cap_ch1 = [] # the window-wise channel for nn
            for sample_idx in range(int(idx - (capture_width * (1 - capture_weight))),
                                    int(idx + (capture_width * capture_weight))):
                cap_ch0.append(self.degraded_80dB_data_global_norm[sample_idx])
                cap_ch1.append(self.degraded_80dB_data[sample_idx])
            captures_list_all.append({
                "Capture Ch0": cap_ch0,
                "Capture Ch1": self.norm_data(cap_ch1),
                "Classification": self.cls_list[classification_index],
                "PeakIdx": capture_width * (1 - capture_weight)
            })
        return captures_list_all

    def prep_set_train(self, windows):
        ch0_list = [d["Capture Ch0"] for d in windows]
        ch1_list = [d["Capture Ch1"] for d in windows]
        cls_list = [d["Classification"] for d in windows]

        ch0 = np.array(ch0_list, dtype=np.float32)
        ch1 = np.array(ch1_list, dtype=np.float32)
        y = np.array(cls_list, dtype=np.int64)
        assert ch0.shape == ch1.shape, "Channel 0 and 1 shapes must match!"

        X = np.stack([ch0, ch1], axis=1)  # shape: (N, 2, T)
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return loader


class InferenceDataCls:
    def __init__(self, raw_unknown_data, idx_list):

        # Two channels, one is recoding-wise norm oen is window-wise norm.

        self.idx_list = idx_list
        # prep data loader
        inf_windows = self.split_spike(raw_unknown_data)
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

    def split_spike(self, raw_unknown_data, capture_width=80, capture_weight=0.90):
        """
        :param capture_width:  How many samples per capture
        :param capture_weight: A right bias value as the dataset
                               tends to house important data on the right of idx
        :return:               A list of all captures in format {Capture Ch0=[], Capture Ch1=[], Classification=int}
        """
        global_norm = np.array(self.norm_data(raw_unknown_data))
        captures_list_all = []
        for classification_index, idx in enumerate(self.idx_list):
            # We rely on the norm and non-norm sets being the same len
            cap_ch0 = [] # the global norm channel for nn
            cap_ch1 = [] # the window-wise channel for nn
            for sample_idx in range(int(idx - (capture_width * (1 - capture_weight))),
                                    int(idx + (capture_width * capture_weight))):
                cap_ch0.append(global_norm[sample_idx])
                cap_ch1.append(raw_unknown_data[sample_idx])
            captures_list_all.append({
                "Capture Ch0": cap_ch0,
                "Capture Ch1": self.norm_data(cap_ch1),
                "PeakIdx": capture_width * (1 - capture_weight)
            })
        return captures_list_all

    def prep_set_inf(self, windows):
        ch0_list = [d["Capture Ch0"] for d in windows]
        ch1_list = [d["Capture Ch1"] for d in windows]

        ch0 = np.array(ch0_list, dtype=np.float32)
        ch1 = np.array(ch1_list, dtype=np.float32)
        assert ch0.shape == ch1.shape, "Channel 0 and 1 shapes must match!"

        X = np.stack([ch0, ch1], axis=1)  # shape: (N, 2, T)
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