# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_evnt_det_utils.py
 Description:
 Author:       Joshua Poole
 Created on:   20251028
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

from scipy.io import loadmat
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import uniform_filter1d
import pywt
from scipy.signal import butter, filtfilt

import torch


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

def filter_wavelet(signal, fs=25000):
    # High-pass filter to remove slow drift
    # This is only present in the final two datasets I think
    nyq = 0.5 * fs
    b, a = butter(3,   10 / nyq, btype='high')
    signal_hp = filtfilt(b, a, signal)

    # Wavelet denoising
    wavelet = 'db4'
    coeffs = pywt.wavedec(signal_hp, wavelet, level=5)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = 0.7 * sigma * np.sqrt(2 * np.log(len(signal_hp))) # tuned 0.7 try other else

    coeffs_thresh = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    clean_wavelet = pywt.waverec(coeffs_thresh, wavelet)

    return clean_wavelet

def widen_labels(labels, width=3):
    """Expand binary 1s in a 1D label array by ±width samples."""
    expanded = np.zeros_like(labels)
    idx = np.where(labels == 1)[0]
    for i in idx:
        start = max(0, i - width)
        end = min(len(labels), i + width + 1)
        expanded[start:end] = 1
    return expanded

def prep_set_train(data, labels, window_size=128, stride=1, window_interleave=1):
    """
    Package the series and indexes into tensors with respect to the
    required window size and stride length.

    It has a sliding window (hence stride) that means the output
    tensors will represent 2*input_size - window_size 1D datapoints
    """
    X, y = [], []
    np_lables = np.array(labels)
    noise_c = 0
    for i in range(0, len(data) - window_size, stride):
        # get windows with a 10 sample spike onset
        # for each spike we want window_interleave windows of noise
        window_split_idx = [int(window_size * 0.5)]
        if any(np_lables[i + int(idx)] == 1 for idx in window_split_idx):
            X.append(data[i:i + window_size])
            y.append(labels[i:i + window_size])
            #plot_sample_with_binary(data[i:i + window_size], labels[i:i + window_size])
            noise_c = 0
        elif np_lables[i-window_size:i + window_size].mean() == 0 and noise_c < window_interleave:
            X.append(data[i:i + window_size])
            y.append(labels[i:i + window_size])
            noise_c += 1

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, window_size)
    y = torch.tensor(y, dtype=torch.float32)  # (N, window_size)
    print(X.size(), y.size())
    return X, y

def prep_set_val(data, labels, window_size=64, stride=1):
    """
    Package the series and indexes into tensors with respect to the
    required window size and stride length.

    It has a sliding window (hence strid) that means the output
    tensors will represent 2*input_size - window_size 1D datapoints
    """
    X, y = [], []
    for i in range(0, len(data) - window_size, stride):
        X.append(data[i:i + window_size])
        y.append(labels[i:i + window_size])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, window_size)
    y = torch.tensor(y, dtype=torch.float32)  # (N, window_size)
    print(X.size(), y.size())
    return X, y

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

def prep_set_inf(data, labels, window_size=80):
    """
    This version id for preparing the inference tensors and will not slide!
    """
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, window_size):
        X.append(data[i:i + window_size])
        y.append(labels[i:i + window_size])

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, window_size)
    y = torch.tensor(y, dtype=torch.float32)  # (N, window_size)
    print(X.size(), y.size())
    return X, y

def nonmax_rejection(preds, threshold, refractory=10):
    preds = np.array(preds)
    candidate_indices = np.where(preds > threshold)[0]
    final_spikes = []
    i = 0
    while i < len(candidate_indices):
        start = candidate_indices[i]
        region = candidate_indices[(candidate_indices >= start) &
                                   (candidate_indices < start + refractory)]
        best = region[np.argmax(preds[region])]
        final_spikes.append(best)
        i += len(region)
    labels_bin = np.isin(np.arange(len(preds)), final_spikes).astype(int)
    return labels_bin

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
