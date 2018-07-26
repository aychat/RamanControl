from multiprocessing import Pool
import numpy as np


def f(args):
    x, y = args
    return x*x


if __name__ == '__main__':

    from itertools import product

    X = np.linspace(-1, 1, 10)
    Y = np.linspace(-1, 1, 10)

    p = Pool(4)
    results = p.map(f, product(X, Y))

    results = np.array(results)
    results = results.reshape(X.size, Y.size).T

    import matplotlib.pyplot as plt

    plt.imshow(results)
    plt.show()
