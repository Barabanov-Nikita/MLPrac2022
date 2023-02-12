import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        column_keys = [(col, np.unique(X[col])) for col in X]
        self.onehot = {col: {elem: np.eye(1, len(unique), k=idx)[0] for idx, elem in enumerate(unique)}
                       for col, unique in column_keys}

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        encoded = [np.asarray([self.onehot[col][elem] for elem in X[col]]) for col in X]
        return np.concatenate(encoded, axis=1)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        column_keys = [(col, np.unique(X[col])) for col in X]
        self.counters = {col: {elem: [Y[X[col] == elem].mean(), len(Y[X[col] == elem]) / len(X)] for elem in unique}
                         for col, unique in column_keys}

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        for col in self.counters:
            for elem in self.counters[col]:
                c = self.counters[col][elem]
                c.append((c[0] + a)/(c[1] + b))

        encoded = [np.asarray([self.counters[col][elem] for elem in X[col]]) for col in X]
        return np.concatenate(encoded, axis=1)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        idxs = group_k_fold(len(X), self.n_folds, seed)
        column_keys = [(col, np.unique(X[col])) for col in X]
        print(idxs)
        self.counters = {}
        for idx in idxs:
            X0, Y0 = X.iloc[idx[1]], Y.iloc[idx[1]]
            info = {col: {elem: [Y0[X0[col] == elem].mean(), len(Y0[X0[col] == elem]) / len(X0), 0] for elem in unique}
                    for col, unique in column_keys}
            for _ in idx[0]:
                self.counters[_] = info

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        for _ in self.counters:
            print(self.counters[_])
        for row in self.counters:
            for col in self.counters[row]:
                for elem in self.counters[row][col]:
                    c = self.counters[row][col][elem]
                    c[2] = (c[0] + a) / (c[1] + b)

        for _ in self.counters:
            print(self.counters[_])
        encoded = [np.asarray([self.counters[idx][col][elem] for idx, elem in enumerate(X[col])]) for col in X]
        return np.concatenate(encoded, axis=1)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    uniques = np.unique(x)
    weights = np.array([np.count_nonzero(y[x == val]) / len(x[x == val]) for val in uniques])
    return weights
