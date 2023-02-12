import numpy as np


class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.x_max = data.max(axis=0)
        self.x_min = data.min(axis=0)
        print(self.x_max, self.x_min)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        minmaxnormalizers = [lambda x: (x - self.x_min[i]) / (self.x_max[i] - self.x_min[i])
                             for i in range(data.shape[1])]
        return np.array([list(map(minmaxnormalizers[i], data[:, i])) for i in range(data.shape[1])]).T

class StandardScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.std = np.std(data, axis=0)
        self.mean = np.mean(data, axis=0)
        print(self.std, self.mean)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        standardnormalizers = [lambda x: (x - self.mean[i]) / self.std[i] for i in range(data.shape[1])]
        return np.array([list(map(standardnormalizers[i], data[:, i])) for i in range(data.shape[1])]).T



