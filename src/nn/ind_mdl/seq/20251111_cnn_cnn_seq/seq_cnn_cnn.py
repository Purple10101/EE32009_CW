# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        seq_cnn_cnn.py
 Description:
 Author:       Joshua Poole
 Created on:   20251111
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
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.nn.ind_mdl.cnn_cls.n_cls import NeuronCNN
from src.nn.ind_mdl.cnn_evnt_det.n_evnt_det import SpikeNet
import src.nn.ind_mdl.cnn_evnt_det.n_evnt_det_utils_nn as evnt_det_utils

dir_name = "EE32009_CW"
p = Path.cwd()
while p.name != dir_name:
    if p.parent == p:
        raise FileNotFoundError(f"Directory '{dir_name}' not found above {Path.cwd()}")
    p = p.parent
os.chdir(p)
print(os.getcwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RecordingInf:

    def __init__(self, dataset, dataset_id):

        self.dataset_id = dataset_id
        # load in models
        self.model_evnt_det = SpikeNet().to(device)
        self.model_evnt_det.load_state_dict(torch.load(
            "src/nn/ind_mdl/models/D2/20251111_neuron_event_det_cnn_window_norm_d2.pt"))
        self.model_evnt_det.eval()

        self.model_cls = NeuronCNN(5)
        self.model_cls.load_state_dict(torch.load(
            "src/nn/ind_mdl/models/D2/20251104_neuron_total_norm_mimic_noise.pt"))
        self.model_cls.eval()

        # data prep
        self.raw_data = dataset['d'][0]
        self.data_norm = self.norm_data()

        # do forward pass of model_evnt_det with raw_data
        self.index_lst, self.index_bin = self.event_det_inf(0.7)
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

    def event_det_inf(self, threshold):

        X_i, index_map_i = evnt_det_utils.prep_set_inf(self.data_norm)
        dataset_i = TensorDataset(X_i)
        loader_i = DataLoader(dataset_i, batch_size=64, shuffle=False)

        self.model_evnt_det.eval()
        all_outputs = []
        with torch.no_grad():
            for X_batch, in loader_i:
                X_batch = X_batch.to(device)
                output = self.model_evnt_det(X_batch)  # shape: (batch_size, 1, window_size) or (batch_size, window_size)
                output = output.squeeze(1).cpu().numpy()  # shape: (batch_size, window_size)
                all_outputs.append(output)

        # Stack all batches back together
        outputs_i = np.concatenate(all_outputs, axis=0)  # (num_windows, window_size)
        # construct our outputs list
        n_total = len(self.data_norm)
        final_probs = np.zeros(n_total)
        counts = np.zeros(n_total)

        for i in range(outputs_i.shape[0]):
            final_probs[index_map_i[i]] += outputs_i[i]
            counts[index_map_i[i]] += 1

        # Average overlapping predictions
        final_probs /= np.maximum(counts, 1)
        preds = evnt_det_utils.nonmax_rejection(final_probs, 0.7)

        output_indexes = np.where(preds == 1)[0]
        return output_indexes, preds

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
        savemat(f"src/nn/ind_mdl/seq/20251111_cnn_cnn_seq/outputs/vis/{self.dataset_id}_vis.mat", export_data)
        export_data_sub = {
            "Index": self.index_lst,
            "Class" : self.cls_lst
        }
        savemat(f"src/nn/ind_mdl/seq/20251111_cnn_cnn_seq/outputs/sub/{self.dataset_id}.mat", export_data_sub)


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

