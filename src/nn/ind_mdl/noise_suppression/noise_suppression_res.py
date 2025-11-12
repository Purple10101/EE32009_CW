# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        noise_suppression_res.py
 Description:
 Author:       Joshua Poole
 Created on:   20251112
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
import pickle

dir_name = "EE32009_CW"
p = Path.cwd()
while p.name != dir_name:
    if p.parent == p:
        raise FileNotFoundError(f"Directory '{dir_name}' not found above {Path.cwd()}")
    p = p.parent
os.chdir(p)
print(os.getcwd())


# lets bandpass later datasets knowing the general spike freq
import numpy as np
from scipy.signal import butter, filtfilt, welch, fftconvolve
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def plot_widow(window):
    x = np.arange(len(window))
    plt.figure(figsize=(8, 4))
    plt.plot(x, window)
    plt.xlabel("idx")
    plt.ylabel("Amplitude")
    plt.title("Raw plot of window")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def plot_widow_spect(freqs, power):
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, power)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Power Spectrum of Neuron Activity (One Window)")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def plot_window_with_spectrum(window, spect):

    freqs, power = spect
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # --- Time-domain plot ---
    x = np.arange(len(window))
    axes[0].plot(x, window)
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw Window")
    axes[0].grid(True)

    # --- Frequency-domain plot ---
    axes[1].plot(freqs, power)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].set_title("Power Spectrum")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show(block=False)

def degrade(raw_data, mimic_sig, noise_scale=1):
    """
    noise turning for different levels:
    60dB - noise_scale=0.25
    40dB - noise_scale=0.4
    20dB - noise_scale=0.6
    0dB - noise_scale=0.8
    sub 0dB - noise_scale=1
    """
    from scipy.ndimage import uniform_filter1d
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

def spectral_power_suppress(noisy, clean, fs, nperseg=2048):
    """
    Suppress the power of noisy signal to match that of clean signal across frequency bands.

    Parameters
    ----------
    noisy : array_like
        The noisy (0 dB SNR) signal.
    clean : array_like
        The clean (80 dB SNR) signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        FFT window length for Welch PSD estimate.

    Returns
    -------
    denoised : ndarray
        The noisy signal with frequency power suppressed to match the clean signal.
    G : ndarray
        The frequency-dependent gain curve applied.
    f : ndarray
        The frequency axis corresponding to G.
    """
    from scipy import signal
    # --- 1) Estimate PSDs ---
    f, P_noisy = signal.welch(noisy, fs=fs, nperseg=nperseg)
    _, P_clean = signal.welch(clean, fs=fs, nperseg=nperseg)

    # --- 2) Compute gain curve (avoid division by zero) ---
    eps = 1e-12
    G = np.sqrt((P_clean + eps) / (P_noisy + eps))

    # Smooth the gain a little to avoid spectral artifacts
    G = signal.savgol_filter(G, 31, 3)  # 31-pt window, cubic poly

    # --- 3) Apply gain curve in frequency domain ---
    # FFT of the noisy signal
    N = len(noisy)
    freqs = np.fft.rfftfreq(N, 1/fs)
    X = np.fft.rfft(noisy)

    # Interpolate gain to FFT bins
    G_interp = np.interp(freqs, f, G, left=G[0], right=G[-1])

    # Apply soft suppression
    X_suppressed = X * G_interp

    # Back to time domain
    denoised = np.fft.irfft(X_suppressed, n=N)

    # remove the low freq sway
    b, a = signal.butter(4, 3 / (fs / 2), btype='highpass')
    filtered = signal.filtfilt(b, a, denoised)

    return filtered


data1 = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

# raw datasets
data_80 = data1['d'][0]
data_60 = degrade(data1['d'][0], data2['d'][0],0.25)
data_40 = degrade(data1['d'][0], data3['d'][0], 0.4)
data_20 = degrade(data1['d'][0], data4['d'][0], 0.6)
data_0 = degrade(data1['d'][0], data5['d'][0], 0.8)
data_sub0 = degrade(data1['d'][0], data6['d'][0], 1)

supressed_d5 = spectral_power_suppress(data5['d'][0], data1['d'][0], 25000)

go_plot_time_series(supressed_d5[-1_000_000:-800_000])

fs = 25_000
f_clean, Pxx_clean = welch(data1['d'][0], fs=fs, nperseg=2048)
f_noisy_0, Pxx_noisy_0 = welch(data5['d'][0], fs=fs, nperseg=2048)
f_noisy_processed_0, Pxx_processed_0 = welch(supressed_d5, fs=fs, nperseg=2048)

fig = go.Figure()
fig.add_trace(go.Scatter(x=f_clean, y=Pxx_clean, mode='lines', name='D1 (80 dB)', line=dict(color='green')))
fig.add_trace(go.Scatter(x=f_noisy_0, y=Pxx_noisy_0, mode='lines', name='D5 (0 dB)', line=dict(color='gold')))
fig.add_trace(go.Scatter(x=f_noisy_processed_0, y=Pxx_processed_0, mode='lines', name='D5 processed', line=dict(color='gray')))
fig.update_layout(
    title="Power Spectral Density Comparison",
    xaxis_title="Frequency (Hz)",
    xaxis_range=[0, 12_000],
    yaxis_title="Power",
    yaxis_type="log",
    template="plotly_white",
    width=900, height=400
)
fig.show()

print()


