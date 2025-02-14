
from torch.utils.data import Dataset
from sklearn import preprocessing

import torch
import numpy as np
import pandas as pd


class BostonHousingDataset(Dataset):

    def __init__(self, path, filter=[], test=False):

        self.data = []
        self.labels = []

        df = pd.read_csv(path, sep=r"\s+")
        pt = preprocessing.PowerTransformer(method="yeo-johnson", standardize=False)

        for col in df:
            if col == "MEDV":
                continue

            if col in filter:
                del df[col]
                continue

            if col == "CHAS" or col == "RMCHAS":
                continue

            if col in ["CRIM", "ZN", "B"]:
                df[col] = np.log1p(df[col])

            if col in ["NOX", "AGE", "LSTAT", "DIS", "TAX"]:
                df[col] = pt.fit_transform(df[[col]])

            df[col] -= df[col].mean()
            df[col] /= df[col].std()

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
