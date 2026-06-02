import numpy as np
import pandas as pd
import os

DATASET_URLS = {
    "heart_failure": "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip",
    "breast_cancer": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    "wine_quality": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "avila": "https://archive.ics.uci.edu/static/public/459/avila.zip",
    "magic_telescope": "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data",
    "room_occupancy": "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip",
    "skin_nonskin": "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt",
    "miniboone": "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt",
    "hepmass": "https://archive.ics.uci.edu/static/public/347/hepmass.zip"
}

def load_dataset(data_path = "./data", subfolder = "room-occupancy"):
    """
    Loads a dataset from a relative data path and subfolder, expecting X.csv and y.csv inside.
    Returns (X, y) as numpy arrays (no column names).
    Assumes y contains only -1 and 1. Data is already preprocessed.
    """
    subfolder_path = os.path.join(data_path, subfolder)
    x_path = os.path.join(subfolder_path, "X.csv")
    y_path = os.path.join(subfolder_path, "y.csv")

    X = pd.read_csv(x_path, header = 0).values
    y = pd.read_csv(y_path, header = 0).values.ravel()
    y = y.astype(np.int32)

    return X, y

def load_all_datasets(data_path="./data"):
    all_data = {}

    for entry in os.listdir(data_path):
        subfolder_path = os.path.join(data_path, entry)
        if not os.path.isdir(subfolder_path):
            continue

        x_path = os.path.join(subfolder_path, "X.csv")
        y_path = os.path.join(subfolder_path, "y.csv")
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            continue

        X, y = load_dataset(data_path=data_path, subfolder=entry)
        all_data[entry] = (X, y)

    return all_data

# X, y = load_dataset("breast_cancer")
# X_wine, y_wine = load_dataset("wine_quality")