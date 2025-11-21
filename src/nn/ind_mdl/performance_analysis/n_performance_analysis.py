# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_performance_analysis.py
 Description:
 Author:       Joshua Poole
 Created on:   20251121
 Version:      1.0
===========================================================

 Notes:
    -

 Requirements:
    -

==================
"""
import torch.nn.functional as F

from scipy.io import loadmat
import pickle

from src.nn.ind_mdl.neuron_classifcation.n_cls_data_prep import *
from src.nn.ind_mdl.neuron_classifcation.n_cls_cnn import NeuronCNN, DualNormNeuronCNN

dir_name = "EE32009_CW"
p = Path.cwd()
while p.name != dir_name:
    if p.parent == p:
        raise FileNotFoundError(f"Directory '{dir_name}' not found above {Path.cwd()}")
    p = p.parent
os.chdir(p)
print(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data1 = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

datasets = [data2, data3, data4, data5, data6]

dataset_id = 6

with open(f"src/nn/ind_mdl/event_detection/outputs/D{dataset_id}.pkl", "rb") as x:
    indexes = pickle.load(x)
with open(f"src/nn/ind_mdl/neuron_classifcation/outputs/D{dataset_id}.pkl", "rb") as y:
    classes = pickle.load(y)

# load inference set for cls to get what the input data to fp looks like

inference_set = InferenceDataCls(datasets[dataset_id-2]['d'][0], data1['d'][0], indexes)
data_at_inference = inference_set.inf_windows

class_counts = [0, 0, 0, 0, 0]
for idx, cls in enumerate(classes):
    if class_counts[cls-1] < 4:
        plot_classification(data_at_inference[idx]["Capture"], cls, cls)
        class_counts[cls-1] += 1

print()