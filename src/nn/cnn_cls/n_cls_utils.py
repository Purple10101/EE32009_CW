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
import random
from scipy.io import loadmat
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d

import torch
from torch.utils.data import TensorDataset, DataLoader


class RecordingTrain:

    def __init__(self, raw_data, idx_lst, cls_lst):
        # raw variables
        self.raw_data = raw_data
        self.idx_lst = idx_lst
        self.cls_lst = cls_lst

        data2 = loadmat('data\D2.mat')
        data3 = loadmat('data\D3.mat')
        data4 = loadmat('data\D4.mat')
        data5 = loadmat('data\D5.mat')
        data6 = loadmat('data\D6.mat')

        raw_data_80 = norm_data(raw_data)
        raw_data_60 = norm_data(degrade(self.raw_data, data2['d'][0], 0.25))
        raw_data_40 = norm_data(degrade(self.raw_data, data3['d'][0], 0.4))
        raw_data_20 = norm_data(degrade(self.raw_data, data4['d'][0], 0.6))
        raw_data_0 = norm_data(degrade(self.raw_data, data5['d'][0], 0.8))
        raw_data_sub0 = norm_data(degrade(self.raw_data, data6['d'][0], 1))

        self.raw_data_lists = [raw_data_80, raw_data_60, raw_data_40, raw_data_20, raw_data_0, raw_data_sub0]

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
        random.shuffle(captures_list_all)
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

def degrade(raw_data, mimic_sig, noise_scale=1):
    """
    noise turning for different levels:
    60dB - noise_scale=0.25
    40dB - noise_scale=0.4
    20dB - noise_scale=0.6
    0dB - noise_scale=0.8
    sub 0dB - noise_scale=1
    """
    # --- Extract noise reference (zero-mean) ---
    noise_ref = mimic_sig - np.mean(mimic_sig)

    # --- Compute reference noise spectrum magnitude ---
    fft_noise = np.fft.rfft(noise_ref)
    mag_noise = np.abs(fft_noise)
    mag_noise_smooth = uniform_filter1d(mag_noise, size=15)

    # --- Generate random-phase noise with same spectral shape ---
    phase = np.exp(1j * 2 * np.pi * np.random.rand(len(mag_noise)))
    colored_noise = np.fft.irfft(mag_noise_smooth * phase)
    colored_noise -= np.mean(colored_noise)
    # --- Emphasize high frequencies to tighten noise around zero ---
    freqs = np.fft.rfftfreq(len(noise_ref))
    tilt = (freqs / freqs.max()) ** 0.5  # gentle boost toward high freq
    mag_shaped = mag_noise_smooth * tilt
    colored_noise = np.fft.irfft(mag_shaped * phase)

    # --- Scale noise strength ---
    target_rms = np.std(mimic_sig)
    current_rms = np.std(colored_noise)
    colored_noise *= (target_rms / (current_rms + 1e-12)) * noise_scale

    # --- Inject noise into clean signal ---
    noisy_out = raw_data + colored_noise

    return noisy_out

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