# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        20251114_seq.py
 Description:
 Author:       Joshua Poole
 Created on:   20251114
 Version:      1.0
===========================================================

 Notes:
    -

 Requirements:
    -

==================
"""
import torch
import torch.nn as nn

from scipy.io import loadmat, savemat
import os

from src.nn.ind_mdl.event_detection.n_event_det_data_prep import *
from src.nn.ind_mdl.event_detection.n_event_det_1d_cnn import SpikeNet

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


class RecordingInf:

    def __init__(self, datasets, dataset_ids):

        self.threshold = 0.9
        # load in models

        # event detection
        event_detection_models = []
        model_paths = [
            "src/nn/ind_mdl/event_detection/models/D2/20251118_neuron_event_det_cnn.pt",
            "src/nn/ind_mdl/event_detection/models/D3/20251118_neuron_event_det_cnn.pt",
            "src/nn/ind_mdl/event_detection/models/D4/20251118_neuron_event_det_cnn.pt",
            "src/nn/ind_mdl/event_detection/models/D5/20251118_neuron_event_det_cnn.pt",
            "src/nn/ind_mdl/event_detection/models/D6/20251118_neuron_event_det_cnn.pt"
        ]
        for path in model_paths:
            model = SpikeNet().to(device)
            model.load_state_dict(torch.load(path))
            event_detection_models.append(model)

        # classifcation
        neuron_classification_models = []
        model_paths = [
            "src/nn/ind_mdl/neuron_classifcation/models/20251118_cls_cnn_all.pt",
            "src/nn/ind_mdl/neuron_classifcation/models/20251118_cls_cnn_all.pt",
            "src/nn/ind_mdl/neuron_classifcation/models/20251118_cls_cnn_all.pt",
            "src/nn/ind_mdl/neuron_classifcation/models/20251118_cls_cnn_all.pt",
            "src/nn/ind_mdl/neuron_classifcation/models/20251118_cls_cnn_all.pt"
        ]
        for path in model_paths:
            model = NeuronCNN(5).to(device)
            model.load_state_dict(torch.load(path))
            neuron_classification_models.append(model)


        for i, dataset in enumerate(datasets):
            # We rely on the Dataset being aligned with all the model
            # lists and the dataset id

            # data prep
            raw_data = dataset['d'][0]

            # do forward pass of model_evnt_det with raw_data
            self.index_lst, self.index_bin = self.event_det_inf(raw_data, event_detection_models[i])
            # do forward pass of model_cls with data_norm with respect to index_lst
            self.cls_lst = self.cls_inf(raw_data, neuron_classification_models[i])

            # export
            self.export_mat(raw_data, dataset_ids[i])
            print(f"exported {dataset_ids[i]}")

    def event_det_inf(self, dataset, model):

        inference_dataset = InferenceData(dataset)

        model.eval()
        all_outputs = []
        with torch.no_grad():
            for X_batch, in inference_dataset.loader_i:
                X_batch = X_batch.to(device)
                output = model(X_batch)  # shape: (batch_size, 1, window_size) or (batch_size, window_size)
                output = output.squeeze(1).cpu().numpy()  # shape: (batch_size, window_size)
                all_outputs.append(output)

        # Stack all batches back together
        outputs_i = np.concatenate(all_outputs, axis=0)  # (num_windows, window_size)
        # construct our outputs list
        n_total = len(dataset)
        final_probs = np.zeros(n_total)
        counts = np.zeros(n_total)

        for i in range(outputs_i.shape[0]):
            final_probs[inference_dataset.index_map[i]] += outputs_i[i]
            counts[inference_dataset.index_map[i]] += 1

        # Average overlapping predictions
        final_probs /= np.maximum(counts, 1)
        preds = nonmax_rejection(final_probs, self.threshold)

        output_indexes = np.where(preds == 1)[0]
        return output_indexes, preds

    def cls_inf(self, dataset, model):
        import torch.nn.functional as F
        model.eval()
        inference_dataset = InferenceDataCls(dataset, data1['d'][0], self.index_lst)
        model_preds = []

        with torch.no_grad():
            for batch in inference_dataset.loader_v:
                # Unpack the batch
                (X_batch,) = batch  # shape: (B, 2, T)
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = F.softmax(logits, dim=1)

                # Check each sample for low-confidence predictions
                max_probs, _ = torch.max(probs, dim=1)
                for i, p in enumerate(max_probs):
                    if p.item() < 0.3:
                        print(f"[Warning] Sample {i} in this batch has very low confidence: max prob = {p.item():.4f}")

                preds = torch.argmax(probs, dim=1) + 1 # classes 1-5 not 0-4
                model_preds.extend(preds.cpu().numpy())
        return model_preds

    def export_mat(self, data, dataset_id):
        export_data = {
            "d": data,
            "Index": self.index_lst,
            "Class" : self.cls_lst
        }
        savemat(f"src/nn/ind_mdl/seq/20251118_seq/outputs/vis/{dataset_id}_vis.mat", export_data)
        export_data_sub = {
            "Index": self.index_lst,
            "Class" : self.cls_lst
        }
        savemat(f"src/nn/ind_mdl/seq/20251118_seq/outputs/sub/{dataset_id}.mat", export_data_sub)


dataset_80db = loadmat('data\D1.mat') # this was the training set
dataset_60db = loadmat('data\D2.mat')
dataset_40db = loadmat('data\D3.mat')
dataset_20db = loadmat('data\D4.mat')
dataset_0db = loadmat('data\D5.mat')
dataset_sub0db = loadmat('data\D6.mat')

datasets = [dataset_60db, dataset_40db, dataset_20db, dataset_0db, dataset_sub0db]
dataset_ids = ["D2", "D3", "D4", "D5", "D6"]

recordings = RecordingInf(datasets, dataset_ids)
print()

