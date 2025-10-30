# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        nn_data_plot_app.py
 Description:  Takes a loaded .mat file and plots it.
 Author:
 Author:       Joshua Poole
 Created on:   20251029
 Version:      1.2
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
from plotly.subplots import make_subplots
import numpy as np
from scipy.io import loadmat
import os

# remove this dir stuff, this is just for my dir
"""os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))
print(os.getcwd())"""

def parse_n_dat(data_pk):
    """
    Takes data in the same format as the DX.mat files
    That data is accessible the by the following:
    data_pk['d'][0], data_pk['Index'][0], data_pk['Class'][0]
    """
    raw_dat, idx, cls = data_pk['d'][0], data_pk['Index'][0], data_pk['Class'][0]

    # parse the idx list into bin series
    idx_bin = []
    for i in range(len(raw_dat)):
        if i in idx:
            idx_bin.append(1)
        else:
            idx_bin.append(0)

    return raw_dat.tolist(), idx_bin, cls.tolist(), idx

def plot_dataset(data_pk):

    raw_dat, idx_bin, cls, idx = parse_n_dat(data_pk)
    x = np.arange(len(raw_dat))

    # Create a figure with 2 rows, shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=("Time Series", "Binary Signal")
    )

    # raw data plot
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=raw_dat,
            mode='lines',
            line=dict(width=1),
            name='Signal'
        ),
        row=1, col=1
    )

    # idx bin plot
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=idx_bin,
            mode='lines',
            line=dict(width=1, color='red'),
            name='Binary'
        ),
        row=2, col=1
    )

    # cls labels
    for i, label in enumerate(cls):
        fig.add_trace(
            go.Scattergl(
                x=[idx[i]],
                y=[raw_dat[idx[i]]],
                mode='markers+text',
                text=[str(label)],
                textposition='top center',
                marker=dict(size=8, symbol='circle', line=dict(width=1, color='black')),
                name=f'Class {label}',
                showlegend = False
            ),
            row=1, col=1
        )

    fig.update_layout(
        title='Interactive Time Series + Binary Signal',
        hovermode='x unified',
        template='plotly_white',
        height=800
    )
    # to try and improve dynamic performance plzzzzz so slow
    fig.update_layout(hovermode=False)

    fig.update_xaxes(title_text='Sample Index', row=2, col=1)
    fig.update_yaxes(title_text='Value', row=1, col=1)
    fig.update_yaxes(title_text='Binary', range=[-0.1, 1.1], row=2, col=1)

    fig.show()


"""
# optional example plot
data = loadmat('data\D1.mat')

plot_dataset(data)
print()"""

