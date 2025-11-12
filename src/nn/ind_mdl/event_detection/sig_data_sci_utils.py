# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        sig_data_sci_utils.py
 Description:
 Author:       Joshua Poole
 Created on:   20251112
 Version:      1.0
===========================================================

 Notes:
    -

 Requirements:
    -

==================
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

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
