# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls_pipeline.py
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
import torch.nn.functional as F

from scipy.io import loadmat

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

def cls_forward_pass(dataN,
                     dataset_id,
                     model_name_conv
                     ):

    ####################################################################################################################
    # DATA #
    ####################################################################################################################
    ground_truth_data = data1['d'][0]
    target_data = dataN['d'][0]

    import pickle
    with open(f"src/nn/ind_mdl/event_detection/outputs/D{dataset_id}.pkl", "rb") as f:
        preds_loaded = pickle.load(f)
    inference_set = InferenceDataCls(target_data, ground_truth_data, preds_loaded)

    ####################################################################################################################
    # MODEL SETUP, TRAINING PARAMS #
    ####################################################################################################################
    model = NeuronCNN(5).to(device)
    model.load_state_dict(torch.load(
        f"src/nn/ind_mdl/neuron_classifcation/models/D{dataset_id}/20251120_cls_cnn_all.pt"))
    model.eval()

    ####################################################################################################################
    # INFERENCE FORWARD PASS #
    ####################################################################################################################
    model_preds = []
    print_count = 0
    with torch.no_grad():
        for capture in inference_set.inf_windows:
            X = np.array(capture["Capture"], dtype=np.float32)
            X = np.expand_dims(X, axis=1)
            X_tensor = torch.tensor(X).T.unsqueeze(0).to(device)
            output = model(X_tensor)
            predicted = torch.argmax(output) + 1  # classes 1-5 not 0-4
            model_preds.append(predicted.item())

            # if predicted == 2 and print_count <10:
            #     plot_classification(capture["Capture"], predicted, predicted)
            #     print_count += 1

    with open(f"src/nn/ind_mdl/neuron_classifcation/outputs/D{dataset_id}.pkl", "wb") as f:
        pickle.dump(model_preds, f)

    print(f"Saved src/nn/ind_mdl/neuron_classifcation/outputs/D{dataset_id}.pkl")


data1 = loadmat('data\D1.mat')
data2 = loadmat('data\D2.mat')
data3 = loadmat('data\D3.mat')
data4 = loadmat('data\D4.mat')
data5 = loadmat('data\D5.mat')
data6 = loadmat('data\D6.mat')

datasets = [data2, data3, data4, data5, data6]
model_name = "20251121_neuron_event_det_cnn"

for i, dataset in enumerate(datasets):
    print(f"Processing dataset {i+2}...")
    cls_forward_pass(dataN=dataset,
                                 dataset_id=i+2,
                                 model_name_conv=model_name
                                 )