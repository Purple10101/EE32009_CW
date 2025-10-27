# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls_train.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.ext.data_loader_cls import Recording, plot_sample
from src.nn.cnn_cls.n_cls import NeuronCNN

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(os.getcwd())

def noise_plt_example(rec, snr_out):
    # plot the 0th capture with noise injection
    for snr in snr_out:
        noisy_un_norm = rec.noise_injection(rec.captures_training, snr)
        noisy_norm = rec.norm_data(noisy_un_norm)
        plot_sample(noisy_norm[0])

def prep_training_set(rec, snr_out=80):
    if snr_out == 80:
        captures = [d["Capture"] for d in rec.captures_training_norm]
        clss = [d["Classification"] for d in rec.captures_training_norm]
        # for example plot some normed time series
        for i in range(30):
            capture = rec.captures_training_norm[i]
            #if capture["Classification"] == 1:
                #plot_sample(capture)
    else:
        noisy_un_norm = rec.noise_injection(rec.captures_training, snr)
        noisy_norm = rec.norm_data(noisy_un_norm)
        captures = [d["Capture"] for d in noisy_norm]
        clss = [d["Classification"] for d in noisy_norm]

    X = np.array(captures, dtype=np.float32)
    y = np.array(clss, dtype=np.int64)
    X = np.expand_dims(X, axis=1)
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=True)


# data prep
print(os.getcwd())
data = loadmat('data\D1.mat')
data_inf = loadmat('data\D2.mat')

rec = Recording(data['d'], data['Index'], data['Class'])

snr_lst = [80] # add back 0 and -10

# show example of manual degradation
#noise_plt_example(rec, snr_lst)
# now create test sets for each SNR
training_sets = []
for snr in snr_lst:
    training_sets.append(prep_training_set(rec, snr))



# model and training
model = NeuronCNN(5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=10e-4)

num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    total_batches = 0

    for dataloader in training_sets:
        for X_batch, y_batch in dataloader:
            y_batch -= 1
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

    avg_loss = epoch_loss / total_batches
    print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

torch.save(model.state_dict(), "src/nn/models/20251025_neuron_noise_inj_cls.pt")

# separate training and inference, make n_cls_inf.py

# load model and evaluate performance
model = NeuronCNN(5)
model.load_state_dict(torch.load("src/nn/models/20251025_neuron_noise_inj_cls.pt"))
model.eval()

scorecard = []

with torch.no_grad():
    for test_capture in rec.captures_test_norm:
        X = np.array(test_capture["Capture"], dtype=np.float32)
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_norm = (X - X_min) / (X_max - X_min)
        X = np.expand_dims(X, axis=1)
        X_tensor = torch.tensor(X).T.unsqueeze(0)
        outputs = model(X_tensor)
        predicted = torch.argmax(outputs) + 1 # classes 1-5 not 0-4
        real_lb = test_capture["Classification"]
        if predicted == real_lb:
            scorecard.append(1)
        else:
            scorecard.append(0)
            #plot_sample(test_capture)
scorecard_array = np.array(scorecard)
print(f"performance = {scorecard_array.mean()*100}%")

scorecard = []

with torch.no_grad():
    noisy_un_norm = rec.noise_injection(rec.captures_training, 0)
    noisy_norm = rec.norm_data(noisy_un_norm)
    for test_capture in noisy_norm:
        X = np.array(test_capture["Capture"], dtype=np.float32)
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_norm = (X - X_min) / (X_max - X_min)
        X = np.expand_dims(X, axis=1)
        X_tensor = torch.tensor(X).T.unsqueeze(0)
        outputs = model(X_tensor)
        predicted = torch.argmax(outputs)
        real_lb = test_capture["Classification"]
        if predicted == real_lb:
            scorecard.append(1)
        else:
            scorecard.append(0)
scorecard_array = np.array(scorecard)
print(f"performance = {scorecard_array.mean()*100}%")