# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        noise_suppression_utils.py
 Description:
 Author:       Joshua Poole
 Created on:   20251111
 Version:      1.0
===========================================================

 Notes:
    - Utility functions for wrestling with noise
      in neuron activation datasets
    - Explore wavelet filtering for the early datasets!

 Requirements:
    -

==================
"""
from scipy.io import loadmat
from pathlib import Path
import os

dir_name = "EE32009_CW"
p = Path.cwd()
while p.name != dir_name:
    if p.parent == p:
        raise FileNotFoundError(f"Directory '{dir_name}' not found above {Path.cwd()}")
    p = p.parent
os.chdir(p)
print(os.getcwd())

data1 = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

#### for research lets analyse D1 and try to pick out the freq components that cause the transients (spikes) ####
def d1_noise_analysis():
    import numpy as np
    from scipy.signal import welch
    import matplotlib.pyplot as plt
    d1_raw_data = data1['d'][0]
    d1_spike_idx = data1['Index'][0]

    x = d1_raw_data.astype(float)
    fs = 25000  # Hz
    spike_indices = np.array(d1_spike_idx)

    win_ms = 4  # total window length around each spike (e.g. 4 ms)
    half_win = int((win_ms / 1000) * fs / 2)

    # collect spikes
    P_spike_all = []
    for i in spike_indices:
        if i - half_win < 0 or i + half_win >= len(x):
            continue
        seg = x[i - half_win: i + half_win]
        f, Pxx = welch(seg, fs=fs, nperseg=len(seg))
        P_spike_all.append(Pxx)
    P_spike = np.mean(P_spike_all, axis=0)

    # collect some baselines
    rng = np.random.default_rng(0)
    n_baselines = len(P_spike_all)
    P_base_all = []

    # pick random windows that don't overlap spikes
    for _ in range(n_baselines * 2):
        j = rng.integers(half_win, len(x) - half_win)
        if np.any(np.abs(spike_indices - j) < half_win * 2):
            continue
        seg = x[j - half_win: j + half_win]
        f, Pxx = welch(seg, fs=fs, nperseg=len(seg))
        P_base_all.append(Pxx)
        if len(P_base_all) >= n_baselines:
            break
    P_base = np.mean(P_base_all, axis=0)

    # --- spectral contrast ---
    delta_db = 10 * np.log10((P_spike + 1e-12) / (P_base + 1e-12))

    # --- plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(f, 10 * np.log10(P_spike), label='Spike windows')
    plt.plot(f, 10 * np.log10(P_base), label='Baseline windows')
    plt.plot(f, delta_db, 'k--', label='Δ Power (spike-baseline)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power [dB]')
    plt.title('Spectral signature of spikes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# lets bandpass later datasets knowing the general spike freq
import numpy as np
from scipy.signal import butter, filtfilt, welch, fftconvolve
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# create spike_template
# maybe use this template to do detection?
def make_spike_template(x, spike_indices, fs, window_ms=2.0):
    """
    Compute average spike waveform template from known spike indices.

    x: 1D array, raw signal
    spike_indices: array of sample indices where spikes occur
    fs: sampling rate (Hz)
    window_ms: total window length around spike (e.g. 2 ms)
    """
    half_win = int((window_ms / 1000) * fs / 2)
    snippets = []

    for i in spike_indices:
        if i - half_win < 0 or i + half_win >= len(x):
            continue
        seg = x[i - half_win: i + half_win]
        snippets.append(seg)

    snippets = np.array(snippets)
    # Remove DC offset and normalize each snippet
    snippets = snippets - np.mean(snippets, axis=1, keepdims=True)
    snippets = snippets / (np.max(np.abs(snippets), axis=1, keepdims=True) + 1e-9)

    # Average across spikes
    template = np.mean(snippets, axis=0)
    template = template - np.mean(template)
    template = template / np.linalg.norm(template)

    t = np.linspace(-window_ms / 2, window_ms / 2, len(template))

    # visualize
    plt.figure(figsize=(6, 4))
    plt.plot(t, snippets.T, color='gray', alpha=0.3)
    plt.plot(t, template, 'k', lw=2, label='Template')
    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude (normalized)')
    plt.title('Average spike template')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return template

def filter_wavelet(signal, fs=25000):
    import pywt
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

def wiener_denoise(noisy, clean_spike_reference, fs, noise_segment=None):
    """
    Wiener filter denoising using reference clean spike data and noise PSD.
    """
    # --- Estimate signal PSD from clean dataset ---
    f_sig, P_sig = plt.psd(clean_spike_reference, NFFT=2048, Fs=fs)
    # --- Estimate noise PSD from baseline in noisy data ---
    if noise_segment is None:
        noise_segment = noisy[:fs]  # assume first 1 s baseline
    f_noise, P_noise = plt.psd(noise_segment, NFFT=2048, Fs=fs)

    # --- Interpolate to match frequency grid ---
    freqs = np.fft.rfftfreq(len(noisy), 1 / fs)
    S_s = np.interp(freqs, f_sig, P_sig)
    S_n = np.interp(freqs, f_noise, P_noise)

    # --- Compute Wiener filter ---
    H = S_s / (S_s + S_n + 1e-12)

    # --- Apply filter ---
    X = np.fft.rfft(noisy)
    X_hat = X * H
    s_hat = np.fft.irfft(X_hat)

    # --- Plot diagnostic ---
    plt.figure(figsize=(10, 5))
    plt.plot(noisy, lw=0.5, label="Noisy")
    plt.plot(s_hat, lw=1, label="Denoised")
    plt.legend();
    plt.title("Wiener denoised signal")
    plt.xlabel("Samples")
    plt.tight_layout()
    plt.show()

    return s_hat

def go_plot_time_series(signal):
    x = np.arange(len(signal))
    # Create interactive line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=signal,
        mode='lines+markers',
        name='Time Series',
        line=dict(width=2),
        marker=dict(size=6)
    ))

    # Add layout details
    fig.update_layout(
        title="Interactive Time Series (Index as X-axis)",
        xaxis_title="Index",
        yaxis_title="Value",
        template="plotly_white"
    )

    fig.show()

def plt_plotter_time_series(signal, binary_signal):
    x_axis = np.arange(len(signal))

    # Check that both have same length
    if len(signal) != len(binary_signal):
        raise ValueError("`sample` and `binary_signal` must have the same length.")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- Top plot: waveform ---
    ax1.plot(x_axis, signal, color='b', label='Signal')
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

#spike_template = make_spike_template(np.array(data1['d'][0]), np.array(data1['Index'][0]), 25000)
noisy_trace = data3['d'][0]

wv_filtered_d3 = filter_wavelet(noisy_trace)
plotting_sample_wv_filtered_d3 = wv_filtered_d3[-200_000:]
x = np.arange(len(plotting_sample_wv_filtered_d3))

plt_plotter_time_series(plotting_sample_wv_filtered_d3, plotting_sample_wv_filtered_d3)
plt_plotter_time_series(data3['d'][0][-200_000:], data3['d'][0][-200_000:])



print()


