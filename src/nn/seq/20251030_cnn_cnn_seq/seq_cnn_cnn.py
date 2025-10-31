# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        seq_cnn_cnn.py
 Description:
 Author:       Joshua Poole
 Created on:   20251030
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
from scipy.io import loadmat, savemat
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import TensorDataset, DataLoader

from src.nn.cnn_cls.n_cls import NeuronCNN
from src.nn.cnn_evnt_det.n_evnt_det import SpikeNet

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))))
print(os.getcwd())

class Recording:

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


class RecordingInf:

    def __init__(self, dataset, dataset_id):

        self.dataset_id = dataset_id
        # load in models
        self.model_evnt_det = SpikeNet()
        self.model_evnt_det.load_state_dict(torch.load("src/nn/models/20251028_neuron_event_det_cnn.pt"))
        self.model_evnt_det.eval()

        self.model_cls = NeuronCNN(5)
        self.model_cls.load_state_dict(torch.load("src/nn/models/20251030_neuron_total_norm.pt"))
        self.model_cls.eval()

        # data prep
        self.raw_data = dataset['d'][0]
        self.data_norm = self.norm_data()

        # do forward pass of model_evnt_det with raw_data
        self.index_lst, self.index_bin = self.event_det_inf()
        plot_sample_with_binary(self.raw_data[:1000], self.index_bin[:1000])
        # do forward pass of model_cls with data_norm with respect to index_lst
        self.cls_lst = self.cls_inf()

        # export
        self.export_mat()

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

    def prep_event_det_inf(self, data, window_size=200):
        """
        This version is for preparing the inference tensors and will not slide!
        """
        X = []
        for i in range(0, len(data) - window_size + 1, window_size):
            X.append(data[i:i + window_size])
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, window_size)
        return X

    def split_spike(self, capture_width=80, capture_weight=0.90):
        """
        :param capture_width:  How many samples per capture
        :param capture_weight: A right bias value as the dataset
                               tends to house important data on the right of idx
        :return:               A list of all captures in format {Capture=[], Classification=int}
        """
        captures_list = []
        valid_indices = []

        left_offset = int(capture_width * (1 - capture_weight))
        right_offset = int(capture_width * capture_weight)

        for idx in self.index_lst:
            start = idx - left_offset
            end = idx + right_offset

            if start < 0 or end >= len(self.data_norm):
                continue

            cap = self.data_norm[start:end]
            captures_list.append({
                "Capture": cap,
                "Classification": None,
                "PeakIdx": left_offset
            })
            valid_indices.append(idx)

        self.index_lst = valid_indices
        return captures_list

    def event_det_inf(self):
        X_tensors = self.prep_event_det_inf(self.raw_data)
        T_dataset = TensorDataset(X_tensors)
        loader = DataLoader(T_dataset, batch_size=32, shuffle=False)

        infered_events_bin = []
        prediction_count = 0
        self.model_evnt_det.eval()
        with torch.no_grad():
            for X_batch in loader:
                outputs = self.model_evnt_det(X_batch[0])
                preds = (outputs > 0.5).float()
                infered_events_bin.extend(preds.flatten().tolist())
                prediction_count += preds.size(0) * preds.size(1)

        output_indexes = []
        for idx, val in enumerate(infered_events_bin):
            if val == 1:
                output_indexes.append(idx)
        return output_indexes, infered_events_bin

    def cls_inf(self):
        captures_list = self.split_spike()
        predictions_lst = []
        with torch.no_grad():
            for capture in captures_list:
                X = np.array(capture["Capture"], dtype=np.float32)
                X = np.expand_dims(X, axis=1)
                X_tensor = torch.tensor(X).T.unsqueeze(0)
                outputs = self.model_cls(X_tensor)
                predicted = torch.argmax(outputs) + 1  # classes 1-5 not 0-4
                predictions_lst.append(predicted.item())
        return predictions_lst

    def export_mat(self):
        export_data = {
            "d": self.raw_data,
            "Index": self.index_lst,
            "Class" : self.cls_lst
        }
        savemat(f"src/nn/seq/20251030_cnn_cnn_seq/outputs/vis/{self.dataset_id}_vis.mat", export_data)
        export_data_sub = {
            "Index": self.index_lst,
            "Class" : self.cls_lst
        }
        savemat(f"src/nn/seq/20251030_cnn_cnn_seq/outputs/sub/{self.dataset_id}.mat", export_data_sub)


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


dataset_80db = loadmat('data\D1.mat') # this was the training set
dataset_60db = loadmat('data\D2.mat')
dataset_40db = loadmat('data\D3.mat')
dataset_20db = loadmat('data\D4.mat')
dataset_0db = loadmat('data\D5.mat')
dataset_sub0db = loadmat('data\D6.mat')

recording_60db = RecordingInf(dataset_60db, "D2")
recording_40db = RecordingInf(dataset_40db, "D3")
recording_20db = RecordingInf(dataset_20db, "D4")
recording_0db = RecordingInf(dataset_0db, "D5")
recording_sub0db = RecordingInf(dataset_sub0db, "D6")
print()

