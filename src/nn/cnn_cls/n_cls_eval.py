# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls_eval.py
 Description:
 Author:       Joshua Poole
 Created on:   20251024
 Version:      1.0
===========================================================

 Notes:
    - So things that could be hampering the performance:
        - Normalisation?
            - Right now you norm each capture independently.
              This could be an issue!
        - More variations of the degraded training set?
        - Shuffling the training set?
        - Criterion and optimiser?
        - Could you randomly deteriorate some samples?
          so that its being fed one at 80 then one at 0 then one at 60

        - !!!! Norm the whole thing. Different neurons have different amplitudes
          you might actually be stupid for not picking up on that bro icl

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

import torch

from src.ext.data_loader_cls import RecordingTrain, plot_sample
from src.nn.cnn_cls.n_cls import NeuronCNN
from src.nn.cnn_cls.n_cls_utils import noise_plt_example, prep_training_set

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(os.getcwd())

# data prep
print(os.getcwd())
data = loadmat('data\D1.mat')
data_inf = loadmat('data\D2.mat')

rec = RecordingTrain(data['d'], data['Index'], data['Class'])

# load model and evaluate performance
model = NeuronCNN(5)
model.load_state_dict(torch.load("src/nn/models/20251030_neuron_total_norm.pt"))
model.eval()

scorecard = []
predictions_lst = []

with torch.no_grad():
    for test_capture in rec.captures_val:
        X = np.array(test_capture["Capture"], dtype=np.float32)
        X = np.expand_dims(X, axis=1)
        X_tensor = torch.tensor(X).T.unsqueeze(0)
        outputs = model(X_tensor)
        predicted = torch.argmax(outputs) + 1 # classes 1-5 not 0-4
        predictions_lst.append(predicted.item())
        real_lb = test_capture["Classification"]
        if predicted == real_lb:
            scorecard.append(1)
        else:
            scorecard.append(0)
            #plot_sample(test_capture)
scorecard_array = np.array(scorecard)
print(f"performance = {scorecard_array.mean()*100}%")

