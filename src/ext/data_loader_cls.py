# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        data_loader_cls.py
 Description:
 Author:       Joshua Poole
 Created on:   20251024
 Version:      1.0
===========================================================

 Notes:
    - Loading in provided datasets of neuron activity
      For the purpose of classification
    - It is assumed we know a list of all peak orogin indexes

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
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random

class Recording:

    def __init__(self, raw_data, idx_lst, cls_lst):
        # raw variables
        self.raw_data = raw_data[0]
        self.idx_lst = idx_lst[0]
        self.cls_lst = cls_lst[0]

        # processed variables
        self.idx_lst_processed = self.peak_idx_processing()
        self.captures_all = self.split_spike()
        self.captures_training, self.captures_test = self.split_caps()

    def split_caps(self, tr_to_tst_r=0.8):
        split_index = int(len(self.captures_all) * tr_to_tst_r)
        return self.captures_all[:split_index], self.captures_all[split_index:]

    def noise_injection(self, dataset, snr_out, snr_in=80):
        ret_val = copy.deepcopy(dataset)
        for i, capture in enumerate(ret_val):
            sig = np.array(capture["Capture"], dtype=np.float32)
            sig_pwr = np.mean(sig ** 2)
            pwr_og = sig_pwr / (10 ** (snr_in / 10))
            pwr_tg = sig_pwr / (10 ** (snr_out / 10))
            pwr_add = pwr_tg - pwr_og
            noise = np.random.normal(0, np.sqrt(pwr_add), size=sig.shape)
            ret_val[i]["Capture"] = capture["Capture"] + noise
        return ret_val

    def split_spike(self, capture_width=60, capture_weight=0.9):
        """
        :param capture_width:  How many samples per capture
        :param capture_weight: A right bias value as the dataset
                               tends to house important data on the right of idx
        :return:               A list of all captures in format {Capture=[], Classification=int}
        """
        captures_list = []
        for classification_index, idx in enumerate(self.idx_lst_processed):
            cap = []
            if isinstance(idx, list):
                # This is my fancy magic eraser like they advertise for pictures on your phone
                # Only this time its for a second peak that's too damn close to the last one
                for sample_idx in range(int(idx[0]-(capture_width * (1 - capture_weight))),
                                        int(idx[0]+(capture_width * capture_weight))):
                    if sample_idx >= idx[1]:
                        # in this case we want to erase the second peak
                        # just keep rolling off with a lil bit of noise
                        # we add noise because else the CNN will go
                        # "bloody hell some of them have constant values after the spike!"
                        inj_value = cap[abs(idx[1] - idx[0]) - 1]
                        noisy_value = inj_value + (random.uniform(-1, 1) * abs(inj_value) * 0.25)
                        cap.append(noisy_value)
                    else:
                        cap.append(self.raw_data[sample_idx])
            else:
                for sample_idx in range(int(idx-(capture_width * (1 - capture_weight))),
                                        int(idx+(capture_width * capture_weight))):
                    cap.append(self.raw_data[sample_idx])
            captures_list.append({
                "Capture": cap,
                "Classification": self.cls_lst[classification_index],
                "PeakIdx": capture_width * (1 - capture_weight)
            })
        return captures_list

    def peak_idx_processing(self, capture_width=60):
        """
        If two peaks are within a frame of each-other, pair the index with
        its overlapping parents index to be later processed accordingly

        :param capture_width:  How many samples per capture
        :return:
        """
        ret_val = self.idx_lst.tolist()
        for i in range(len(ret_val)):
            for j in range(i + 1, len(self.idx_lst)):
                if abs(self.idx_lst[i] - self.idx_lst[j]) <= capture_width and self.idx_lst[i] < self.idx_lst[j]:
                    ret_val[i] = [ret_val[i], ret_val[j]]
        return ret_val


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
    plt.show()