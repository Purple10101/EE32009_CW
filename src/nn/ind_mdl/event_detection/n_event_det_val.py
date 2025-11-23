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
from src.nn.ind_mdl.event_detection.n_event_det_sig_proc_pk import *
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
data_unknown_val = data6['d'][0][split_index:]
idx_bin_val = idx_bin[split_index:]
idx_list_val = np.where(idx_bin_val == 1)[0]

val_set = ValidationData(raw_80dB_data=data1_val,
                         raw_unknown_data=data_unknown_val,
                         idx_list=idx_list_val,
                         idx_bin=idx_bin_val,
                         target_id=6,
                         widen_labels=1)

plot_sample_with_binary(val_set.data_proc, val_set.idx_ground_truth_bin)

# now prep the d2 set for total inference.

data_inf = data6['d'][0]

inf_set = InferenceDataEvntDet(data_inf, data1['d'][0])

print()
########################################################################################################################
# MODEL SETUP, TRAINING PARAMS #
########################################################################################################################

model = SpikeNet().to(device)
model.load_state_dict(torch.load(
    f"src/nn/ind_mdl/event_detection/models/D6/20251121_neuron_event_det_cnn.pt"))
model.eval()

threshold = 0.9

print()
########################################################################################################################
# VALIDATION FORWARD PASS #
########################################################################################################################

# Do forward pass on the _val data using the index map to
# reconstruct the predictions
all_preds = []
with torch.no_grad():
    for X_batch, _ in val_set.loader_v:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        probs = torch.sigmoid(output)
        preds = (probs > threshold).float().squeeze(1).cpu().numpy()
        all_preds.append(preds)

outputs_v = np.concatenate(all_preds, axis=0)  # (num_windows, window_size)
N = len(val_set.data_proc)
binary_spikes = np.zeros(N, dtype=int)

for idx, pred in zip(val_set.indices, outputs_v):
    if pred == 1:
        binary_spikes[int(idx)] = 1

print(len([x for x in binary_spikes if x != 0]))
plot_sample_with_binary(val_set.data_proc, binary_spikes)

metrics = tolerant_binary_metrics(binary_spikes, val_set.idx_ground_truth_bin, tol=50)
print(f"Accuracy:  {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1:        {metrics['f1']:.3f}")
print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
print(f"Zero agreement: {metrics['zero_agreement']:.3f}")


# now use the peak detector from sign processing
predicted_indexes = peak_detection(val_set.data_proc)
predicted_indexes_bin = np.isin(np.arange(val_set.data_proc.shape[0]), predicted_indexes).astype(int)

predicted_indexes_dn = peak_detection(inf_set.data_proc)
# this predicts 3853 for D6

metrics = tolerant_binary_metrics(predicted_indexes_bin, val_set.idx_ground_truth_bin, tol=50)
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
all_probs = []
with torch.no_grad():
    for X_batch in inf_set.loader_i:
        X_batch = X_batch[0].to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)

probs_full = np.zeros(len(inf_set.data_proc))
probs_full[inf_set.index_map] = all_probs

binary_spikes = nms(probs_full, threshold=0.9, refractory=5)
spikes = np.where(binary_spikes == 1)[0]

print(len(spikes))
plot_sample_with_binary(inf_set.data_proc, binary_spikes)

import pickle
with open("src/nn/ind_mdl/inference_pkl/D2.pkl", "wb") as f:
    output_indexes = np.where(preds == 1)[0]
    pickle.dump(output_indexes, f)

print()

"""
Prediction counts for all datasets

D2 = 3728 vs true 3985
D3 = 3095 vs true 3327
D4 = 3488 vs true 3031
D5 = 2637 vs true 2582 # colored noise with sd=3
D6 = 3609 vs true 3911 # colored noise with sd=5, th = 0.85

"""