
from sklearn import preprocessing


class DatasetTransformer:

    def __init__(self):
        self.feat_transformer = {}
        self.feat_mean = {}
        self.feat_std = {}


    def fit_transform(self, X, col):

        if col in ["NOX", "AGE", "LSTAT", "DIS", "TAX"]:
            pt = preprocessing.PowerTransformer(method="yeo-johnson", standardize=False)
            X_trans = pt.fit_transform(X.values.reshape(-1, 1))
            self.feat_transformer[col] = pt
        else:
            X_trans = X.values.reshape(-1, 1)

        mean = X_trans.mean()
        std = X_trans.std()
        self.feat_mean[col] = mean
        self.feat_std[col] = std

        return (X_trans - mean) / std


    def transform(self, X, col):

        if col in self.feat_transformer:
            pt = self.feat_transformer[col]
            X_trans = pt.transform(X.values.reshape(-1, 1))
        else:
            X_trans = X.values.reshape(-1, 1)

        mean = self.feat_mean[col]
        std = self.feat_std[col]
        
        return (X_trans - mean) / std
