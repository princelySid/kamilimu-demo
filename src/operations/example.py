import os
import pickle

import pandas as pd


def convert_data(data):
    scaler = pickle.load(os.getenv("SCALER"))
    data = pd.DataFrame.from_dict(data).T
    scaler.transform(data)
    return scaler.transform(data)
