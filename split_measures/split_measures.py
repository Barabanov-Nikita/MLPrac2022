import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    n = len(sample)
    _, counts = np.unique(sample, return_counts=True)
    error = 1 - max(counts) / n
    entropy = -sum(x / n * np.log(x / n) for x in counts)
    gini = 1 - sum((x / n) ** 2 for x in counts)

    measures = {'gini': gini, 'entropy': entropy, 'error': error}
    return measures


print(evaluate_measures([]))
