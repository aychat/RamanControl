from multiprocessing import Pool
import numpy as np


class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


def f(args1, args2, args):
    x, y = args
    print x*x + args1.a - args2.b
    return


if __name__ == '__main__':
    params = ADict(
        a=1.,
        b=2.
    )

    from functools import partial

    X = np.linspace(-1, 1, 10)
    Y = np.linspace(-1, 1, 10)

    p = Pool(4)
    results = p.map(partial(f, params, params), zip(X, Y))
