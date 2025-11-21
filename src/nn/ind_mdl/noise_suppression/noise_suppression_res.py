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
from src.nn.ind_mdl.noise_suppression.noise_suppression_utils import *
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

def plot_widow(window, name="Raw plot of window"):
    x = np.arange(len(window))
    plt.figure(figsize=(8, 4))
    plt.plot(x, window)
    plt.xlabel("idx")
    plt.ylabel("Amplitude")
    plt.title(name)
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

def plot_spect_comparason(ref, unknown, ref_map, fs=25000):
    f_ref, Pxx_ref = welch(ref, fs=fs, nperseg=2048)
    f_unknown, Pxx_unknown = welch(unknown, fs=fs, nperseg=2048)
    f_ref_map, Pxx_ref_map = welch(ref_map, fs=fs, nperseg=2048)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_ref, y=Pxx_ref, mode='lines', name='ref', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=f_unknown, y=Pxx_unknown, mode='lines', name='unknown', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=f_ref_map, y=Pxx_ref_map, mode='lines', name='ref remapped',
                             line=dict(color='gray')))
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

def spectral_power_degrade(clean, noisy, fs, nperseg=2048):
    """
    Apply spectral power shaping to make a clean signal sound like the noisy one.

    This does not add the low freq sway. to get around that you're gonna have to
    take local normalisation windows during both pk detection and classification.
    """
    from scipy import signal

    # Estimate PSDs
    f, P_clean = signal.welch(clean, fs=fs, nperseg=nperseg)
    _, P_noisy = signal.welch(noisy, fs=fs, nperseg=nperseg)
    # Compute *degradation* gain curve
    eps = 1e-12
    G = np.sqrt((P_noisy + eps) / (P_clean + eps))
    # Smooth the gain curve
    G = signal.savgol_filter(G, 31, 3)
    # Apply gain to clean signal
    N = len(clean)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    X = np.fft.rfft(clean)
    # Interpolate gain
    G_interp = np.interp(freqs, f, G, left=G[0], right=G[-1])
    # Apply gain
    X_degraded = X * G_interp
    degraded = np.fft.irfft(X_degraded, n=N)

    return degraded

########################################################################################################################
# OPERATION CHAIN FUNCTIONS #
########################################################################################################################


def wt_test(dn_signal, d1_signal, fs=25000):
    # We like this pipeline
    dn_ss = spectral_power_suppress(dn_signal, d1_signal, fs=fs)
    dn_ss_bandpass = bandpass_neurons(dn_ss, fs=fs)
    dn_ss_bandpass_wt = filter_wavelet(dn_ss_bandpass)
    x = np.arange(len(dn_signal))

    plot_widow(dn_ss_bandpass_wt[-400_000:-390_000])
    plot_widow(dn_signal[-400_000:-390_000])
    print()

def event_detection_train_pl(d1_signal, dn_signal, fs=25000):

    d1_dn = zscore(spectral_power_degrade(d1_signal, dn_signal, fs))
    d1_dn_d1 = zscore(spectral_power_suppress(d1_dn, d1_signal, fs))
    d1_dn_d1_wt = zscore(filter_wavelet(d1_dn_d1))
    # for D5 std=3, D6 std=5 else dont perform this step
    #d1_dn_d1_wt_added = zscore(add_colored_noise(d1_dn_d1_wt, std=5))
    d1_dn_d1_wt_added_bp = zscore(bandpass_neurons(d1_dn_d1_wt))

    dn_d1 = zscore(spectral_power_suppress(dn_signal, d1_signal, fs))
    dn_d1_wt = zscore(filter_wavelet(dn_d1))
    dn_d1_wt_bp = zscore(bandpass_neurons(dn_d1_wt))

    x = np.arange(len(dn_signal))
    # Plot all in one figure
    plt.figure(figsize=(13, 5))
    plt.plot(x, dn_d1_wt_bp, label="signal DN on inference")
    plt.plot(x, d1_dn_d1_wt_added_bp, label="Produced signal")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Event detection inf pipeline")
    plt.grid(True)
    plt.show(block=False)

    print()

def event_detection_inf_pl(dn_signal, d1_signal, fs=25000):
    dn_ss = spectral_power_suppress(dn_signal, d1_signal, fs=fs, gain_scalr=1)
    dn_ss_bandpass = bandpass_neurons(dn_ss, fs=fs)
    dn_ss_bandpass_wt = filter_wavelet(dn_ss_bandpass) #  this will be inference data

    plot_spect_comparason(d1_signal, dn_signal, dn_ss_bandpass)

    x = np.arange(len(dn_signal))[-400_000:-390_000]
    # Plot all in one figure
    plt.figure(figsize=(10, 5))

    plt.plot(x, zscore(dn_signal)[-400_000:-390_000], label="Raw signal DN")
    plt.plot(x, zscore(bandpass_neurons(d1_signal))[-400_000:-390_000], label="Reference signal")
    plt.plot(x, zscore(dn_ss)[-400_000:-390_000], label="signal DN after spec suppression")
    plt.plot(x, zscore(dn_ss_bandpass)[-400_000:-390_000], label="signal DN after spec suppression and bandpass")
    plt.plot(x, zscore(dn_ss_bandpass_wt)[-400_000:-390_000], label="signal DN after spec suppression, bandpass, wt")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Event detection inf pipeline")
    plt.grid(True)
    plt.show(block=False)
    print()

    plt.show()


def cls_train_pl(d1_signal):
    print()

def cls_inf_pl(dn_signal):
    print()



data1 = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

data = data3['d'][0]
data_ref = data1['d'][0]

wt_test(data, data_ref)
event_detection_train_pl(data_ref, data)
event_detection_inf_pl(data, data_ref)



plot_widow(data)
plot_widow(bandpass_neurons(data))

plot_widow(data[-1_000_000:-900_000])
plot_widow(bandpass_neurons(data)[-1_000_000:-900_000])



print()
