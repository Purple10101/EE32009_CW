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

#os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))
#print(os.getcwd())

class RecordingTrain:

    def __init__(self, raw_data, idx_lst, cls_lst):
        # raw variables
        self.raw_data = raw_data[0]
        self.idx_lst = idx_lst[0]
        self.cls_lst = cls_lst[0]

        # normalised raw data
        self.norm_data = self.norm_data()

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

    def norm_data(self):
        """
        Norm the whole dataset between 1 and -1
        centered about zero
        """
        ret_val = copy.deepcopy(self.raw_data)
        raw_data_max = max(ret_val)
        raw_data_min = min(ret_val)
        ret_val = (2 * (ret_val - raw_data_min) /
            (raw_data_max - raw_data_min) - 1)
        return ret_val

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
        for i in range(len(self.norm_data)):
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
        for classification_index, idx in enumerate(self.idx_lst):
            cap = []
            for sample_idx in range(int(idx-(capture_width * (1 - capture_weight))),
                                    int(idx+(capture_width * capture_weight))):
                cap.append(self.norm_data[sample_idx])
            captures_list_all.append({
                "Capture": cap,
                "Classification": self.cls_lst[classification_index],
                "PeakIdx": capture_width * (1 - capture_weight)
            })
        return captures_list_all


class Recording:

    def __init__(self, raw_data, idx_lst=None, cls_lst=None):
        # raw variables
        self.raw_data = raw_data[0]
        self.idx_lst = idx_lst[0] if idx_lst is not None else None
        self.cls_lst = cls_lst[0] if cls_lst is not None else None

        # Noise gets added to the un-normalized time series
        # and then gets normalized

        # processed variables
        self.idx_lst_processed = self.peak_idx_processing()
        self.captures_all, self.captures_train = self.split_spike()
        # training has the stacked neurons removed
        self.captures_training, self.captures_test = self.split_caps()

        self.captures_training_norm = self.norm_data(self.captures_training)
        self.captures_test_norm = self.norm_data(self.captures_all)

    def split_caps(self, tr_to_tst_r=0.8):
        split_index = int(len(self.captures_train) * tr_to_tst_r)
        return self.captures_train[:split_index], self.captures_train[split_index:]

    def norm_data(self, dataset):
        ret_val = copy.deepcopy(dataset)
        for i, capture in enumerate(ret_val):
            raw_data_max = max(capture["Capture"])
            raw_data_min = min(capture["Capture"])
            ret_val[i]["Capture"] = (2 * (capture["Capture"] - raw_data_min) /
                (raw_data_max - raw_data_min) - 1)
        return ret_val

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

    def split_spike(self, capture_width=80, capture_weight=0.90):
        """
        :param capture_width:  How many samples per capture
        :param capture_weight: A right bias value as the dataset
                               tends to house important data on the right of idx
        :return:               A list of all captures in format {Capture=[], Classification=int}
        """
        captures_list_tr = []
        captures_list_all = []
        for classification_index, idx in enumerate(self.idx_lst_processed):
            cap = []
            if isinstance(idx, list):
                # This is my fancy magic eraser like they advertise for pictures on your phone
                # Only this time its for a second peak that's too damn close to the last one
                decay_rate = 1
                for sample_idx in range(int(idx[0]-(capture_width * (1 - capture_weight))),
                                        int(idx[0]+(capture_width * capture_weight))):
                    if sample_idx >= idx[1]:
                        # in this case we want to erase the second peak
                        # just keep rolling off with a lil bit of noise
                        # we add noise because else the CNN will go
                        # "bloody hell some of them have constant values after the spike!"
                        ##inj_value = cap[-1]
                        ##inj_value /= decay_rate
                        ##decay_rate += 0.001
                        ##noisy_value = inj_value + (random.uniform(-1, 1) * abs(inj_value) * 0.3)
                        ##cap.append(noisy_value)
                        cap.append(self.raw_data[sample_idx])
                    else:
                        cap.append(self.raw_data[sample_idx])
                # We will exclude this sample from training
                captures_list_all.append({
                    "Capture": cap,
                    "Classification": self.cls_lst[classification_index],
                    "PeakIdx": capture_width * (1 - capture_weight)
                })
            else:
                for sample_idx in range(int(idx-(capture_width * (1 - capture_weight))),
                                        int(idx+(capture_width * capture_weight))):
                    cap.append(self.raw_data[sample_idx])
                captures_list_tr.append({
                    "Capture": cap,
                    "Classification": self.cls_lst[classification_index],
                    "PeakIdx": capture_width * (1 - capture_weight)
                })
                captures_list_all.append({
                    "Capture": cap,
                    "Classification": self.cls_lst[classification_index],
                    "PeakIdx": capture_width * (1 - capture_weight)
                })
        return captures_list_all, captures_list_tr

    def peak_idx_processing(self, capture_width=60):
        """
        If two peaks are within a frame of each-other, pair the index with
        its overlapping parents index to be later processed accordingly

        :param capture_width:  How many samples per capture
        :return:
        """
        ret_val = copy.deepcopy(self.idx_lst.tolist())
        for i in range(len(ret_val)):
            for j in range(len(ret_val)):
                if abs(self.idx_lst[i] - self.idx_lst[j]) <= capture_width and self.idx_lst[i] < self.idx_lst[j]:
                    ret_val[i] = [self.idx_lst[i], self.idx_lst[j]]
                    break
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
    plt.show(block=False)

"""data = loadmat('data\D1.mat')
rec = RecordingTrain(data['d'], data['Index'], data['Class'])"""
