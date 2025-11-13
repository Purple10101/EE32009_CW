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
from pathlib import Path
import os
import numpy as np
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

dir_name = "EE32009_CW"
p = Path.cwd()
while p.name != dir_name:
    if p.parent == p:
        raise FileNotFoundError(f"Directory '{dir_name}' not found above {Path.cwd()}")
    p = p.parent
os.chdir(p)
print(os.getcwd())

########################################################################################################################
# PLOTTING FUNCTIONS #
########################################################################################################################

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

########################################################################################################################
# SIGNAL PROCESSING FUNCTIONS #
########################################################################################################################

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
    """
    from scipy import signal
    # Estimate PSDs
    f, P_noisy = signal.welch(noisy, fs=fs, nperseg=nperseg)
    _, P_clean = signal.welch(clean, fs=fs, nperseg=nperseg)
    # Compute gain curve (avoid division by zero)
    eps = 1e-12
    G = np.sqrt((P_clean + eps) / (P_noisy + eps))
    # Smooth the gain a little to avoid spectral artifacts
    G = signal.savgol_filter(G, 31, 3)  # 31-pt window, cubic poly
    # Apply gain curve in frequency domain
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
    # Add tiny noise to prevent zero-flat NaNs later
    eps = 1e-4 * np.random.randn(len(clean_wavelet))
    clean_wavelet = clean_wavelet + eps
    return clean_wavelet



