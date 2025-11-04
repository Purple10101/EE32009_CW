from tkinter.font import names
import scipy.io as spio
import pandas as pd

def load_mat_datasets(names=None):
    if names is None:
        names = [f"D{i}" for i in range(1, 7)]
    datasets = {}
    for name in names:
        path = f"data/{name}.mat"
        m = spio.loadmat(path, squeeze_me=True)
        data = m.get("d")
        datasets[name] = pd.DataFrame(data)
    return datasets