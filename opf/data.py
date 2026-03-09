# data.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Diff2SPDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert x.shape[0] == y.shape[0]
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_csv_data(data_dir: str, merged_csv: str, label_csv: str, n_rows: int):
    merged_path = os.path.join(data_dir, merged_csv)
    label_path = os.path.join(data_dir, label_csv)

    # label
    dfy = pd.read_csv(label_path, nrows=n_rows)
    # Compatibility: the label column may be named new_label or something else.
    if "new_label" in dfy.columns:
        y = dfy["new_label"].values
    else:
        y = dfy.iloc[:, 0].values
    y = y.astype(np.int64) - 1  # 1~16 -> 0~15

    # merged_data: read the header first, then drop the first Time column.
    header = pd.read_csv(merged_path, nrows=0)
    cols = list(header.columns)
    if len(cols) < 2:
        raise ValueError("merged_data.csv has an invalid number of columns; it should contain at least Time + 18 features.")
    feat_cols = cols[1:]  # drop Time

    dfx = pd.read_csv(merged_path, usecols=feat_cols, nrows=n_rows)
    x = dfx.values.astype(np.float32)

    if x.shape[0] != y.shape[0]:
        n = min(x.shape[0], y.shape[0])
        x, y = x[:n], y[:n]

    if x.shape[1] != 18:
        raise ValueError(
            f"Loaded feature dimension is {x.shape[1]}, but it is expected to be 18. "
            "Please check whether merged_data.csv has exactly 18 columns after dropping Time."
        )

    return x, y
