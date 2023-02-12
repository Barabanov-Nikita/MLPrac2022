import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    idxs = [_ for _ in range(num_objects)]
    fold_size = num_objects // num_folds
    folds = [idxs[i: None if i == (num_folds - 1) * fold_size else i + fold_size]
             for i in range(0, num_folds * fold_size, fold_size)]
    split = [(np.array([x for y in folds[:i] + folds[i+1:] for x in y]), folds[i]) for i in range(len(folds))]
    return split


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    params_suite = [(n, m, w, norm)
                    for n in parameters["n_neighbors"]
                    for m in parameters["metrics"]
                    for w in parameters["weights"]
                    for norm in parameters["normalizers"]]
    stat_dict = {}
    for params in params_suite:
        knn = knn_class(params[0], metric=params[1], weights=params[2])
        norm = params[3][0]
        stat = []
        for fold in folds:
            if norm is not None:
                if len(X.shape) == 1:
                    norm.fit(X[fold[0]])
                else:
                    norm.fit(X[fold[0], :])
                X_norm = norm.transform(X)
            else:
                X_norm = X
            knn.fit(X_norm[fold[0], :], y[fold[0]])
            stat.append(score_function(y[fold[1]], knn.predict(X_norm[fold[1], :])))

        stat_dict[(params[3][1], params[0], params[1], params[2])] = np.mean(stat)
    return stat_dict
