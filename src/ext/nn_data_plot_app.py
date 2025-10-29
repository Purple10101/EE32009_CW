# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        nn_data_plot_app.py
 Description:
 Author:       Joshua Poole
 Created on:   20251029
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

import plotly.graph_objects as go
import numpy as np
from scipy.io import loadmat
import os

os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))
print(os.getcwd())

def plot_dataset(data_ts, data_spk=None, data_cls=None):
    fig = go.Figure()
    x = np.linspace(0, len(data_ts), num=len(data_ts))

    fig.add_trace(go.Scattergl(
        x=x,
        y=data_ts,
        mode='lines',
        line=dict(width=1),
        name='Signal'
    ))

    fig.update_layout(
        title='Interactive Time Series Explorer',
        xaxis=dict(
            title='Date',
            rangeslider=dict(visible=True),
            type='date'
        ),
        yaxis=dict(title='Value'),
        hovermode='x unified',
        template='plotly_white',
        height=700
    )

    fig.show()


data = loadmat('data\D1.mat')
raw_data = data['d'][0]
raw_data = raw_data.tolist()

plot_dataset(raw_data)
print()

