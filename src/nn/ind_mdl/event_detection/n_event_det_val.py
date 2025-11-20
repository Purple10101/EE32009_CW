# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_event_det_val.py
 Description:
 Author:       Joshua Poole
 Created on:   20251113
 Version:      1.0
===========================================================

 Notes:
    -

 Requirements:
    -

==================
"""
from scipy.io import loadmat

from src.nn.ind_mdl.event_detection.n_event_det_data_prep import *
from src.nn.ind_mdl.event_detection.n_event_det_1d_cnn import SpikeNet

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

# Imma make a general training algo for d2 but it can be generalised for each dataset

########################################################################################################################
# DATA #
########################################################################################################################

# make index binary
idx_bin = np.isin(np.arange(data1['d'][0].shape[0]), data1['Index'][0]).astype(int)

# get an 20% val set
split_index = int(len(data1['d'][0]) * 0.8)

data1_val = data1['d'][0][split_index:]
# we need d2 to be the same size so unfortunately we lose some resolution
data_unknown_val = data2['d'][0][split_index:]
idx_train = idx_bin[split_index:]

val_set = ValidationData(data1_val, data_unknown_val, idx_train, 2)

plot_sample_with_binary(val_set.data_proc, val_set.idx_ground_truth_bin)

# now prep the d2 set for total inference.

data_inf = data2['d'][0]

inf_set = InferenceData(data_inf, data1['d'][0])

print()
########################################################################################################################
# MODEL SETUP, TRAINING PARAMS #
########################################################################################################################

model = SpikeNet().to(device)
model.load_state_dict(torch.load(
    "src/nn/ind_mdl/event_detection/models/D2/20251120_neuron_event_det_cnn.pt"))
model.eval()

print()
########################################################################################################################
# VALIDATION FORWARD PASS #
########################################################################################################################

# Do forward pass on the _val data using the index map to
# reconstruct the predictions
all_outputs = []
with torch.no_grad():  # disables gradient computation (saves memory)
    for X_batch, _ in val_set.loader_v:
        X_batch = X_batch.to(device)
        output = model(X_batch)  # shape: (batch_size, 1, window_size) or (batch_size, window_size)
        output = output.squeeze(1).cpu().numpy()  # shape: (batch_size, window_size)
        all_outputs.append(output)

# Stack all batches back together
outputs_v = np.concatenate(all_outputs, axis=0)  # (num_windows, window_size)
# construct our outputs list
n_total = len(val_set.data_proc)
final_probs = np.zeros(n_total)
counts = np.zeros(n_total)

for i in range(outputs_v.shape[0]):
    final_probs[val_set.index_map[i]] += outputs_v[i]
    counts[val_set.index_map[i]] += 1

# Average overlapping predictions
final_probs /= np.maximum(counts, 1)
preds = nonmax_rejection(final_probs, 0.9)
print(len([x for x in preds if x != 0]))
plot_sample_with_binary(val_set.data_proc[-11000:], preds[-11000:])

metrics = tolerant_binary_metrics(preds, val_set.idx_ground_truth_bin, tol=50)
print(f"Accuracy:  {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1:        {metrics['f1']:.3f}")
print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
print(f"Zero agreement: {metrics['zero_agreement']:.3f}")

"""
threshold at 0.9 compared to 0.7 at training
Accuracy:  0.999
Precision: 0.998
Recall:    1.000
F1:        0.999
TP: 409, FP: 1, FN: 0
"""

print()
########################################################################################################################
# INFERENCE FORWARD PASS #
########################################################################################################################

# Do forward pass on the _inf data using the index map to
# reconstruct the predictions
all_outputs = []
with torch.no_grad():  # disables gradient computation (saves memory)
    for X_batch, in inf_set.loader_i:
        X_batch = X_batch.to(device)
        output = model(X_batch)  # shape: (batch_size, 1, window_size) or (batch_size, window_size)
        output = output.squeeze(1).cpu().numpy()  # shape: (batch_size, window_size)
        all_outputs.append(output)

# Stack all batches back together
outputs_i = np.concatenate(all_outputs, axis=0)  # (num_windows, window_size)
# construct our outputs list
n_total = len(data_inf)
final_probs = np.zeros(n_total)
counts = np.zeros(n_total)

for i in range(outputs_i.shape[0]):
    final_probs[inf_set.index_map[i]] += outputs_i[i]
    counts[inf_set.index_map[i]] += 1

# Average overlapping predictions
final_probs /= np.maximum(counts, 1)

preds = nonmax_rejection(final_probs, 0.9, refractory=3)
print(len([x for x in preds if x != 0]))
plot_sample_with_binary(data_inf[-11000:], preds[-11000:])
plot_sample_with_binary(inf_set.data_proc[-11000:], preds[-11000:])

import pickle
with open("src/nn/ind_mdl/inference_pkl/D2.pkl", "wb") as f:
    pickle.dump(preds, f)

print()

"""
Prediction counts for all datasets

D2 = 3728 vs true 3985
D3 = 3095 vs true 3327
D4 = 3488 vs true 3031
D5 = 2637 vs true 2582 # colored noise with sd=3
D6 = 3609 vs true 3911 # colored noise with sd=5, th = 0.85

"""