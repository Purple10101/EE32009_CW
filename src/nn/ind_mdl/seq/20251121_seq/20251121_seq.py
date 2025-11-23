# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        20251120_seq.py
 Description:
 Author:       Joshua Poole
 Created on:   20251114
 Version:      1.0
===========================================================

 Notes:
    - New pipeline architecture

 Requirements:
    -

==================
"""


from scipy.io import loadmat, savemat
from pathlib import Path
import os
import numpy as np
import pickle

from src.nn.ind_mdl.event_detection.n_event_det_pipeline import event_detection_forward_pass
from src.nn.ind_mdl.neuron_classifcation.n_cls_pipeline import cls_forward_pass



dir_name = "EE32009_CW"
p = Path.cwd()
while p.name != dir_name:
    if p.parent == p:
        raise FileNotFoundError(f"Directory '{dir_name}' not found above {Path.cwd()}")
    p = p.parent
os.chdir(p)
print(os.getcwd())

def export_mat(data, index_lst, cls_lst, dataset_id):
    export_data = {
        "d": data,
        "Index": index_lst,
        "Class": cls_lst
    }
    savemat(f"src/nn/ind_mdl/seq/20251121_seq/outputs/vis/D{dataset_id}_vis.mat", export_data)
    print(f"Saved file src/nn/ind_mdl/seq/20251121_seq/outputs/vis/D{dataset_id}_vis.mat")
    export_data_sub = {
        "Index": index_lst,
        "Class": cls_lst
    }
    savemat(f"src/nn/ind_mdl/seq/20251121_seq/outputs/sub/D{dataset_id}.mat", export_data_sub)
    print(f"Saved file src/nn/ind_mdl/seq/20251121_seq/outputs/sub/D{dataset_id}_vis.mat")


def process_dataset(dataset,
                    dataset_id,
                    model_name_conv,
                    threshold,
                    refractory,
                    run_inference=False
                    ):
    """
    if run_inference then we will run inference manually
    else look for .pkl files
    if none exist we will run manually anyways
    """
    try:
        if run_inference:
            raise FileNotFoundError
        with open(f"src/nn/ind_mdl/event_detection/outputs/D{dataset_id}.pkl", "rb") as x:
            indexes = pickle.load(x)
        with open(f"src/nn/ind_mdl/neuron_classifcation/outputs/D{dataset_id}.pkl", "rb") as y:
            classes = pickle.load(y)
        export_mat(dataset['d'][0], indexes, classes, dataset_id)

    except FileNotFoundError:
        idx_count, vis_data_proc = event_detection_forward_pass(dataN=dataset,
                                                                dataset_id=dataset_id,
                                                                model_name_conv=model_name_conv,
                                                                threshold=threshold,
                                                                refractory=refractory
                                                                )
        with open(f"src/nn/ind_mdl/event_detection/outputs/D{dataset_id}.pkl", "rb") as x:
            indexes = pickle.load(x)
        cls_forward_pass(dataN=dataset,
                         dataset_id=dataset_id,
                         model_name_conv=model_name_conv
                         )
        with open(f"src/nn/ind_mdl/neuron_classifcation/outputs/D{dataset_id}.pkl", "rb") as y:
            classes = pickle.load(y)
        export_mat(vis_data_proc, indexes, classes, dataset_id)
    print(f"Exported D{dataset_id} outputs")


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

for i, dataset in enumerate(datasets):
    print(f"Processing dataset {i+2}...")
    process_dataset(dataset=dataset,
                    dataset_id=i + 2,
                    model_name_conv=model_name,
                    threshold=thresholds[i],
                    refractory=refractories[i],
                    run_inference=False
                    )



