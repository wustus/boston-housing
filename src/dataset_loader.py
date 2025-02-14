
from torch.utils.data import Dataset

import torch
import numpy as np
import pandas as pd


class BostonHousingDataset(Dataset):

    def __init__(self, path, transformer, filter=[], test=False):

        self.data = []
        self.labels = []

        df = pd.read_csv(path, sep=r"\s+")

        train = not test

        for col in df:
            if col == "MEDV":
                continue

            if col in filter:
                df.drop(columns=[col], inplace=True)
                continue

            if col == "CHAS":
                continue

            if col in ["CRIM", "ZN", "B"]:
                df[col] = np.log1p(df[col])

            if train:
                df[col] = transformer.fit_transform(df[col], col).flatten()
            else:
                df[col] = transformer.transform(df[col], col).flatten()


        self.df = df

        def proc(r):
            self.data.append(torch.tensor(r.to_numpy()[:-1]).float())
            self.labels.append(torch.tensor(r.to_numpy()[-1]).float())

        df.apply(proc, axis=1)


    def features(self):
        return self.df.columns[:-1]


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
