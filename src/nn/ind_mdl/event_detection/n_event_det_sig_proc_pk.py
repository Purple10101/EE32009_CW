# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_event_det_data_sig_proc_pk.py
 Description:
 Author:       Joshua Poole
 Created on:   20251120
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

from scipy.signal import find_peaks

def peak_detection(dataset, pk_height_s=2.0, distance_millis=0.05, fs=25_000):
    std = dataset.std(dtype=np.float64)
    height = pk_height_s * std.max() if std > 0 else None
    distance = int(max(1, distance_millis * 0.001 * fs))

    peaks, _ = find_peaks(dataset, height=height, distance=distance)
    return peaks


