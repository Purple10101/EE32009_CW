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

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter1d
import pywt
from scipy.signal import butter, filtfilt, find_peaks


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

def denoise_fft(noisy_signal, fs=25000, lowcut=4, highcut=1000):
    """
    aligns the noisy sig to the reference D1
    This only really works down to 20db
    """

    fft_vals = np.fft.fft(noisy_signal)
    fft_freqs = np.fft.fftfreq(len(noisy_signal), 1 / fs)

    band_mask = (np.abs(fft_freqs) >= lowcut) & (np.abs(fft_freqs) <= highcut)

    fft_filtered = np.zeros_like(fft_vals)
    fft_filtered[band_mask] = fft_vals[band_mask]

    filtered_signal = np.fft.ifft(fft_filtered)

    return filtered_signal.real

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

def degrade(raw_data, mimic_sig, noise_scale=1):
    """
    noise turning for different levels:
    60dB - noise_scale=0.25
    40dB - noise_scale=0.4
    20dB - noise_scale=0.6
    0dB - noise_scale=0.8
    sub 0dB - noise_scale=1
    """
    # Extract noise reference (zero-mean)
    noise_ref = mimic_sig - np.mean(mimic_sig)

    # Compute reference noise spectrum magnitude
    fft_noise = np.fft.rfft(noise_ref)
    mag_noise = np.abs(fft_noise)
    mag_noise_smooth = uniform_filter1d(mag_noise, size=15)

    # Generate random-phase noise with same spectral shape
    phase = np.exp(1j * 2 * np.pi * np.random.rand(len(mag_noise)))
    colored_noise = np.fft.irfft(mag_noise_smooth * phase)
    colored_noise -= np.mean(colored_noise)
    # Emphasize high frequencies to tighten noise around zero
    freqs = np.fft.rfftfreq(len(noise_ref))
    tilt = (freqs / freqs.max()) ** 0.5  # boost toward high freq
    mag_shaped = mag_noise_smooth * tilt
    colored_noise = np.fft.irfft(mag_shaped * phase)

    # Scale noise strength
    target_rms = np.std(mimic_sig)
    current_rms = np.std(colored_noise)
    colored_noise *= (target_rms / (current_rms + 1e-12)) * noise_scale

    # Inject noise into clean signal
    noisy_out = raw_data + colored_noise

    return noisy_out

def sig_peak_det(signal_in, k=3.9, distance=25, window_size=500, plot=None):
    signal_in = np.asarray(signal_in)
    t = np.arange(len(signal_in))

    # --- Compute rolling median and MAD efficiently ---
    from collections import deque
    medians = np.zeros_like(signal_in)
    mads = np.zeros_like(signal_in)

    window = deque(maxlen=window_size)
    for i, val in enumerate(signal_in):
        window.append(val)
        if len(window) == window_size:
            w = np.array(window)
            m = np.median(w)
            mad = 1.4826 * np.median(np.abs(w - m))
            medians[i] = m
            mads[i] = mad
        else:
            medians[i] = np.median(signal_in[:i+1])
            mad = 1.4826 * np.median(np.abs(signal_in[:i+1] - medians[i]))
            mads[i] = mad

    threshold = medians + k * mads
    if plot is not None:
        # --- Peak detection ---
        peaks, props = find_peaks(signal_in, height=threshold, distance=distance, prominence=0.2 * np.std(signal_in))
        fig = go.Figure()

        # Main signal trace
        fig.add_trace(go.Scatter(
            x=t, y=signal_in,
            mode='lines',
            name='Neural Signal',
            line=dict(color='blue')
        ))

        # Spike markers
        fig.add_trace(go.Scatter(
            x=t[peaks], y=signal_in[peaks],
            mode='markers',
            name='Detected Spikes',
            marker=dict(color='red', size=8, symbol='x')
        ))

        fig.update_layout(
            title="Neuron Spike Detection",
            xaxis_title="Time (samples)",
            yaxis_title="Amplitude",
            template="plotly_white",
            hovermode="x unified",
            width=1200,
            height=600
        )

        fig.show()
    return peaks

def val_peak_det(detected, ground_truth, tol=50):
    detected = np.sort(np.asarray(detected))
    ground_truth = np.sort(np.asarray(ground_truth))

    matched_pred = np.zeros(len(detected), dtype=bool)
    matched_true = np.zeros(len(ground_truth), dtype=bool)

    # For each true spike, find if there's a predicted one nearby
    for i, gt in enumerate(ground_truth):
        diffs = np.abs(detected - gt)
        close = np.where(diffs <= tol)[0]
        if len(close) > 0:
            matched_true[i] = True
            matched_pred[close[0]] = True  # mark the first match

    TP = np.sum(matched_true)
    FP = np.sum(~matched_pred)
    FN = np.sum(~matched_true)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }