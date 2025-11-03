# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls_utils.py
 Description:
 Author:       Joshua Poole
 Created on:   20251029
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
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader


class RecordingTrain:

    def __init__(self, raw_data, idx_lst, cls_lst):
        # raw variables
        self.raw_data = raw_data
        self.idx_lst = idx_lst
        self.cls_lst = cls_lst

        raw_data_80 = norm_data(raw_data)
        raw_data_60 = norm_data(degrade(raw_data, 60))
        raw_data_40 = norm_data(degrade(raw_data, 40))
        raw_data_20 = norm_data(degrade(raw_data, 20))
        raw_data_0 = norm_data(degrade(raw_data, 0))
        raw_data_sub0 = norm_data(degrade(raw_data, -10))
        # the subzero one is making it really hard. Performance on inference is like 20%
        # you are NOT SHUFFLING WHEN SPLITTING INTO TRAINING AND VAL
        self.raw_data_lists = [raw_data_60]

        # a refined collection of peaks that don't overlap
        #self.idx_lst_ref = self.peak_idx_processing()

        # the bin series representing the idx_lst_ref
        self.idx_bin_ref = self.peak_idx_to_bin()

        # create the list of captures
        self.captures = self.split_spike()

        # compute 80% split index
        split_idx = int(0.8 * len(self.captures))
        # assign splits
        self.captures_train = self.captures[:split_idx]
        self.captures_val = self.captures[split_idx:]

    def peak_idx_processing(self, capture_width_pos=72):
        """
        If two peaks are within a frame of each-other, pair the index with
        its overlapping parents index to be later processed accordingly

        :param capture_width:  How many samples per capture
        :return:               A copy of the idx list with the overlapping
                               spikes removed.
        """
        idx_lst_loc = self.idx_lst.tolist()
        idx_lst_loc.sort()
        ret_val = []
        for i in range(len(idx_lst_loc)):
            if i < len(idx_lst_loc) - 1:
                if (idx_lst_loc[i + 1] - idx_lst_loc[i]) > capture_width_pos:
                    ret_val.append(idx_lst_loc[i])
                else:
                    print()
        return ret_val

    def peak_idx_to_bin(self):
        """
        Take the list of index as input
        Output a binary series where spikes occur 1 else 0
        """
        ret_val = []
        for i in range(len(self.raw_data)):
            if i in self.idx_lst:
                ret_val.append(1)
            else:
                ret_val.append(0)
        return ret_val

    def split_spike(self, capture_width=80, capture_weight=0.90):
        """
        :param capture_width:  How many samples per capture
        :param capture_weight: A right bias value as the dataset
                               tends to house important data on the right of idx
        :return:               A list of all captures in format {Capture=[], Classification=int}
        """
        captures_list_all = []
        for data in self.raw_data_lists:
            for classification_index, idx in enumerate(self.idx_lst):
                cap = []
                for sample_idx in range(int(idx-(capture_width * (1 - capture_weight))),
                                        int(idx+(capture_width * capture_weight))):
                    cap.append(data[sample_idx])
                captures_list_all.append({
                    "Capture": cap,
                    "Classification": self.cls_lst[classification_index],
                    "PeakIdx": capture_width * (1 - capture_weight)
                })
        return captures_list_all


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

def degrade(raw_data, snr_db):
    # Compute signal power and desired noise power
    signal_power = np.mean(raw_data**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=raw_data.shape)
    return raw_data + noise

def noise_plt_example(rec, snr_out):
    # plot the 0th capture with noise injection
    for snr in snr_out:
        noisy_un_norm = rec.noise_injection(rec.captures_training, snr)
        noisy_norm = rec.norm_data(noisy_un_norm)
        plot_sample(noisy_norm[0])

def prep_training_set(rec):

    captures = [d["Capture"] for d in rec.captures_train]
    cls = [d["Classification"] for d in rec.captures_train]

    X = np.array(captures, dtype=np.float32)
    y = np.array(cls, dtype=np.int64)
    X = np.expand_dims(X, axis=1)
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def plot_sample(sample):
    # Generate X values automatically: 0, 1, 2, ..., n-1
    x_axis = np.arange(len(sample["Capture"]))
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, sample["Capture"], marker='o', linestyle='-', color='b')
    plt.title("Discrete Dataset Plot")
    plt.xlabel("Index (0 to n)")
    plt.ylabel("Y Values")
    plt.text(0.05, 0.95, sample["Classification"],
             transform=plt.gca().transAxes,  # relative to plot axes
             fontsize=12, color='red', fontweight='bold',
             verticalalignment='top')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(block=False)