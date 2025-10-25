# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls.py
 Description:
 Author:       Joshua Poole
 Created on:   20251024
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

from scipy.io import loadmat
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.ext.data_loader_cls import Recording, plot_sample

os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))

class NeuronCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),   # keeps length 60
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# data prep
data = loadmat('data\D1.mat')
data_inf = loadmat('data\D2.mat')

rec = Recording(data['d'], data['Index'], data['Class'])

noisy = rec.noise_injection(rec.captures_all, 20)
print()
#plot_sample(noisy[1])
#plot_sample(rec.captures_all[1])

captures = [d["Capture"] for d in rec.captures_training]
clss = [d["Classification"] for d in rec.captures_training]

X = np.array(captures, dtype=np.float32)
y = np.array(clss, dtype=np.int64)
# normalize the captures
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)

X = np.expand_dims(X, axis=1)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model and training

"""model = NeuronCNN(6)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")

torch.save(model.state_dict(), "neuron_cls.pt")"""

# load model and evaluate performance
model = NeuronCNN(6)
model.load_state_dict(torch.load("src/nn/models/20251024_neuron_cls.pt"))
model.eval()

scorecard = []

with torch.no_grad():
    for test_capture in rec.captures_test:
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

with torch.no_grad():
    for test_capture in rec.noise_injection(rec.captures_test, 0):
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