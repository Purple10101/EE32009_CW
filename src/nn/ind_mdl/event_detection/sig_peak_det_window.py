# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        sig_peak_det_window.py
 Description:
 Author:       Joshua Poole
 Created on:   20251112
 Version:      1.0
===========================================================

 Notes:
    - For windows that the NN determine to contain spikes
      get the index of that spike.
    - performance will break down as noise increases.

 Requirements:
    - torch.nn

==================
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter

def get_peak_idx(window, min_distance=5, smooth=True):
    """
    Returns the index of the most prominent peak in `window`.
    - min_distance: minimum samples between peaks
    - smooth: apply light Savitzky–Golay smoothing to reduce noise
    """
    x = np.asarray(window)

    # optional light smoothing (keeps the big bump, suppresses chatter)
    if smooth and len(x) >= 11:
        x = savgol_filter(x, window_length=11, polyorder=3)

    prom = 0.1 * np.ptp(x)  # 10% of peak-to-peak range
    peaks, props = find_peaks(x, prominence=prom, distance=min_distance)

    if peaks.size == 0:
        return int(np.argmax(x))        # just take the global max

    main = int(peaks[np.argmax(props["prominences"])])
    return main