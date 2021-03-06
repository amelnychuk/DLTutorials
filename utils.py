import numpy as np
import pandas as pd


def get_data():
    data = pd.read_csv(
        'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/ann_logistic_extra/ecommerce_data.csv')
    data_matrix = data.values

    X = data_matrix[:, :-1]
    Y = data_matrix[:, -1]

    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]

    # onehot encode
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:, (D - 1)].astype(np.int32)] = 1
    X2[:, -4:] = Z

    return X2, Y


def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]

    return X2, Y2


def every(N: int, iteration: int) -> bool:
    """Returns true for every N of iteration.

    :param N:
        number to modulo by
    :param iteration:
        iteration of a loop
    :return:
    """
    return iteration % N == 0
