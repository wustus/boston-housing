
from torch.utils.data import Dataset

import torch
import pandas as pd


class BostonHousingDataset(Dataset):

    def __init__(self, path, filter=[], test=False):

        self.data = []
        self.labels = []

        df = pd.read_csv(path, sep=r"\s+")
        for col in df:
            if col == "MEDV":
                continue

            if col in filter:
                del df[col]
                continue

            if col == "CHAS":
                continue

            df[col] -= df[col].mean()
            df[col] /= df[col].std()

        def proc(r):
            self.data.append(torch.tensor(r.to_numpy()[:-1]).float())
            self.labels.append(torch.tensor(r.to_numpy()[-1]).float())

        df.apply(proc, axis=1)


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
