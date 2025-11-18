# -*- coding: utf-8 -*-
"""
===========================================================
 Title:        n_cls_val.py
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

ground_truth_data = data1['d'][0]
ground_idx_list = data1['Index'][0]
ground_cls_list = data1['Class'][0]

target_data = data2['d'][0]

training_set = TrainingData(ground_truth_data, ground_idx_list, ground_cls_list)

print()
########################################################################################################################
# MODEL SETUP, TRAINING PARAMS #
########################################################################################################################

model = NeuronCNN(5).to(device)
model.load_state_dict(torch.load(
    "src/nn/ind_mdl/neuron_classifcation/models/20251118_cls_cnn_all.pt"))
model.eval()

print()
########################################################################################################################
# VALIDATION FORWARD PASS #
########################################################################################################################
from sklearn.metrics import confusion_matrix, classification_report, f1_score

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in training_set.loader_v:
        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        preds = outputs.argmax(dim=1) + 1 # classes 1-5 not 0-4

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        #X_sample = X[1].cpu().numpy()
        #y_true = y[1].item()
        #y_pred = preds[1].cpu().item()
        #if y_pred != y_true:
        #plot_classification(X_sample[0], y_true, y_pred)
        #print()


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = (all_preds == all_labels).mean()
f1 = f1_score(all_labels, all_preds, average="weighted")  # good for class imbalance
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=3)

print(f"Accuracy: {acc*100:.2f}%")
print(f"Weighted F1: {f1:.3f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

print()
########################################################################################################################
# INFERENCE FORWARD PASS #
########################################################################################################################

print()