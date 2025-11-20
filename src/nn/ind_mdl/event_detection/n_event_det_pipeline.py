# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_event_det_pipeline.py
 Description:
 Author:       Joshua Poole
 Created on:   20251120
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

def event_detection_forward_pass(dataN,
                                 dataset_id,
                                 model_name_conv,
                                 threshold,
                                 refractory):

    ####################################################################################################################
    # DATA #
    ####################################################################################################################

    data_inf = dataN['d'][0]
    inf_set = InferenceDataEvntDet(data_inf, data1['d'][0])

    ####################################################################################################################
    # MODEL SETUP, TRAINING PARAMS #
    ####################################################################################################################

    model = SpikeNet().to(device)
    model.load_state_dict(torch.load(
        f"src/nn/ind_mdl/event_detection/models/D{dataset_id}/{model_name_conv}.pt"))
    model.eval()

    ####################################################################################################################
    # INFERENCE FORWARD PASS #
    ####################################################################################################################

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

    preds = nonmax_rejection(final_probs, threshold, refractory=refractory)
    plot_sample_with_binary(data_inf[-11000:], preds[-11000:])
    plot_sample_with_binary(inf_set.data_proc[-11000:], preds[-11000:])

    import pickle
    with open(f"src/nn/ind_mdl/event_detection/outputs/D{dataset_id}.pkl", "wb") as f:
        output_indexes = np.where(preds == 1)[0]
        pickle.dump(output_indexes, f)

    print(f"Saved src/nn/ind_mdl/event_detection/outputs/D{dataset_id}.pkl")
    return len([x for x in preds if x != 0])

"""
Prediction counts for all datasets

D2 = 3728 vs true 3985
D3 = 3095 vs true 3327
D4 = 3488 vs true 3031
D5 = 2637 vs true 2582 # colored noise with sd=3
D6 = 3609 vs true 3911 # colored noise with sd=5, th = 0.85

latest outputs = [3382, 2915, 2884, 2521, 3486]
"""

data1 = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

datasets = [data2, data3, data4, data5, data6]
model_name = "20251121_neuron_event_det_cnn"

# some dataset wise hyperparams
#               D2   D3   D4   D5   D6
thresholds =   [0.9, 0.9, 0.9, 0.9, 0.85]
refractories = [3,   3,   3,   10,  10]

num_spikes = []


for i, dataset in enumerate(datasets):
    print(f"Processing dataset {i+2}...")
    num_spikes.append(event_detection_forward_pass(dataN=dataset,
                                                   dataset_id=i+2,
                                                   model_name_conv=model_name,
                                                   threshold=thresholds[i],
                                                   refractory=refractories[i])
                      )

print(num_spikes)
