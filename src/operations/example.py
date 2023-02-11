import os
import pickle

import pandas as pd


def convert_data(data):
    with open(os.getenv("SCALER"), "rb") as f:
        scaler = pickle.load(f)
    data = pd.DataFrame.from_dict(data, orient="index")
    data = data.T
    scaler.transform(data)
    return scaler.transform(data)
